import os
from .time import Timer

def get_importances(model, features):
    """Get XGBoost feature importances from an xgboost model and list of features.

    Keyword arguments:
    model -- a trained XGBoost Booster object
    features -- a list of feature names corresponding to the features the model was trained on.
    """

    for feature in features:
        assert '\n' not in feature and '\t' not in feature, "\\n and \\t cannot be in feature names"

    outfile = open('mlcrate_xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

    importance = model.get_fscore(fmap='mlcrate_xgb.fmap')
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    os.remove('mlcrate_xgb.fmap')

    return importance

def train_kfold(params, x_train, y_train, x_test=None, folds=5, stratify=None, random_state=1337, skip_checks=False, print_imp='final'):
    from sklearn.model_selection import KFold, StratifiedKFold  # Optional dependencies
    from collections import defaultdict
    import numpy as np
    import xgboost as xgb

    assert print_imp in ['every', 'final', None]

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

    if not skip_checks:
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

    p_train = np.zeros_like(y_train)
    ps_test = []
    models = []
    scores = []
    imps = defaultdict(int)

    fold_i = 0
    for train_kf, valid_kf in splits:
        print('[mlcrate] Running fold {}, {} train samples, {} validation samples'.format(fold_i, len(train_kf), len(valid_kf)))
        d_train = xgb.DMatrix(x_train[train_kf], label=y_train[train_kf])
        d_valid = xgb.DMatrix(x_train[valid_kf], label=y_train[valid_kf])

        t.add('fold{}'.format(fold_i))

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        mdl = xgb.train(params, d_train, params.get('nrounds', 100000), watchlist,
                        early_stopping_rounds=params.get('early_stopping_rounds', 50), verbose_eval=params.get('verbose_eval', 1))

        scores.append(mdl.best_score)

        print('[mlcrate] Finished training fold {} - took {} seconds - running score {}'.format(fold_i, t.elapsed('fold{}'.format(fold_i)), np.mean(scores)))

        # Get importances for this model and add to global importance
        imp = get_importances(mdl, columns)
        if print_imp == 'every':
            print('Fold {} importances:'.format(fold_i), imp)

        for f, i in imp:
            imps[f] += i

        p_valid = mdl.predict(d_valid, ntree_limit=mdl.best_ntree_limit)
        if x_test is not None:
            p_test = mdl.predict(d_test, ntree_limit=mdl.best_ntree_limit)

        p_train[valid_kf] = p_valid
        ps_test.append(p_test)
        models.append(mdl)

        fold_i += 1

    if x_test is not None:
        p_test = np.mean(ps_test, axis=0)

    print('[mlcrate] Finished training {} XGBoost models, took {} seconds'.format(folds, t.elapsed(0)))

    if print_imp in ['every', 'final']:
        print('[mlcrate] Overall feature importances:', sorted(imps.items(), key=lambda x: x[1], reverse=True))

    if x_test is None:
        p_test = None

    return models, p_train, p_test, imps
