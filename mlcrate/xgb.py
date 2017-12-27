import os

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

    importance = gbm.get_fscore(fmap='mlcrate_xgb.fmap')
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    os.remove('mlcrate_xgb.fmap')

    return importance
