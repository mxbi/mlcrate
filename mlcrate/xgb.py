import os
from .time import Timer

def get_importances(model, features):
    """Get XGBoost feature importances from an xgboost model and list of features.

    Keyword arguments:
    model -- a trained xgboost.Booster object
    features -- a list of feature names corresponding to the features the model was trained on.

    Returns:
    importance -- A list of (feature, importance) tuples representing sorted importance
    """

    for feature in features:
        assert '\n' not in feature and '\t' not in feature, "\\n and \\t cannot be in feature names"

    outfile = open('mlcrate_xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i += 1
    outfile.close()

    importance = model.get_fscore(fmap='mlcrate_xgb.fmap')
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    os.remove('mlcrate_xgb.fmap')

    return importance

def train_kfold(params, x_train, y_train, x_test=None, folds=5, stratify=None, random_state=1337, skip_checks=False, print_imp='final'):
    """Trains a set of XGBoost models with chosen parameters on a KFold split dataset, returning full out-of-fold
    training set predictions (useful for stacking) as well as test set predictions and the models themselves.
    Test set predictions are generated by averaging predictions from all the individual fold models - this means
    1 model fewer has to be trained and from my experience performs better than retraining a single model on the full set.

    Optionally, the split can be stratified along a passed array. Feature importances are also computed and summed across all folds for convenience.

    Keyword arguments:
    params -- Parameters passed to the xgboost model, as well as ['early_stopping_rounds', 'nrounds', 'verbose_eval'], which are passed to xgb.train()
              Defaults: early_stopping_rounds = 50, nrounds = 100000, verbose_eval = 1
    x_train -- The training set features
    y_train -- The training set labels
    x_test (optional) -- The test set features
    folds (default: 5) -- The number of folds to perform
    stratify (optional) -- An array to stratify the splits along
    random_state (default: 1337) -- Random seed for splitting folds
    skip_checks -- By default, this function tries to reorder the test set columns to match the order of the training set columns. Set this to disable this behaviour.
    print_imp -- One of ['every', 'final', None] - 'every' prints importances for every fold, 'final' prints combined importances at the end, None does not print importance

    Returns:
    models -- a list of trained xgboost.Booster objects
    p_train -- Out-of-fold training set predictions (shaped like y_train)
    p_test -- Mean of test set predictions from the models
    imps -- dict with \{feature: importance\} pairs representing the sum feature importance from all the models.
    """

    from sklearn.model_selection import KFold, StratifiedKFold  # Optional dependencies
    from collections import defaultdict
    import numpy as np
    import xgboost as xgb

    assert print_imp in ['every', 'final', None]

    # If it's a dataframe, we can take column names, otherwise just use column indices (eg. for printing importances).
    if hasattr(x_train, 'columns'):
        columns = x_train.columns.values
        columns_exists = True
    else:
        columns = np.arange(x_train.shape[1])
        columns_exists = False

    x_train = np.asarray(x_train)
    y_train = np.array(y_train)

    if x_test is not None:
        if columns_exists and not skip_checks:
            try:
                x_test = x_test[columns]
            except Exception as e:
                print('[mlcrate] Could not coerce x_test columns to match x_train columns. Set skip_checks=True to run anyway.')
                raise e

        x_test = np.asarray(x_test)
        d_test = xgb.DMatrix(x_test)

    if not skip_checks and x_test is not None:
        assert x_train.shape[1] == x_test.shape[1], "x_train and x_test have different numbers of features."

    print('[mlcrate] Training {} {}XGBoost models on training set {} {}'.format(folds, 'stratified ' if stratify is not None else '',
            x_train.shape, 'with test set {}'.format(x_test.shape) if x_test is not None else 'without a test set'))

    # Init a timer to get fold durations
    t = Timer()

    if stratify is not None:
        kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        splits = kf.split(x_train, stratify)
    else:
        kf = KFold(n_splits=folds, shuffle=True, random_state=4242)
        splits = kf.split(x_train)

    p_train = np.zeros_like(y_train, dtype=np.float32)
    ps_test = []
    models = []
    scores = []
    imps = defaultdict(int)

    fold_i = 0
    for train_kf, valid_kf in splits:
        print('[mlcrate] Running fold {}, {} train samples, {} validation samples'.format(fold_i, len(train_kf), len(valid_kf)))
        d_train = xgb.DMatrix(x_train[train_kf], label=y_train[train_kf])
        d_valid = xgb.DMatrix(x_train[valid_kf], label=y_train[valid_kf])

        # Start a timer for the fold
        t.add('fold{}'.format(fold_i))

        # Metrics to print
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        mdl = xgb.train(params, d_train, params.get('nrounds', 100000), watchlist,
                        early_stopping_rounds=params.get('early_stopping_rounds', 50), verbose_eval=params.get('verbose_eval', 1), feval=params.get('feval'))

        scores.append(mdl.best_score)

        print('[mlcrate] Finished training fold {} - took {} - running score {}'.format(fold_i, t.format_elapsed('fold{}'.format(fold_i)), np.mean(scores)))

        # Get importances for this model and add to global importance
        imp = get_importances(mdl, columns)
        if print_imp == 'every':
            print('Fold {} importances:'.format(fold_i), imp)

        for f, i in imp:
            imps[f] += i

        # Get predictions from the model
        p_valid = mdl.predict(d_valid, ntree_limit=mdl.best_ntree_limit)
        if x_test is not None:
            p_test = mdl.predict(d_test, ntree_limit=mdl.best_ntree_limit)

        p_train[valid_kf] = p_valid

        ps_test.append(p_test)
        models.append(mdl)

        fold_i += 1

    if x_test is not None:
        p_test = np.mean(ps_test, axis=0)

    print('[mlcrate] Finished training {} XGBoost models, took {}'.format(folds, t.format_elapsed(0)))

    if print_imp in ['every', 'final']:
        print('[mlcrate] Overall feature importances:', sorted(imps.items(), key=lambda x: x[1], reverse=True))

    if x_test is None:
        p_test = None

    return models, p_train, p_test, imps
