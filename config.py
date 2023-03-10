# PATH_DATA = './data/train.csv'
PATH_DATA = './data/train_data.csv'
PATH_SCORES = './model/scores.csv'
PATH_MODEL = './model/best_estimator.pickle'
PATH_TEST_DATA = './data/hidden_test.csv'
# PATH_TEST_DATA = './data/test_data.csv'
PATH_PRED = './model/test_prediction.csv'
PATH_MODEL = './model/best_estimator.pickle'

TARGET_COLUMN_NAME = 'target'
COLUMNS_TO_DROP = ['8']

PIPE_PARAMS={
		# 'ridge__fit_intercept':[True],
		# 'ridge__alpha':[5,10],
		# 'ridge__solver':[ 'svd']

		# "n_estimators"      : [10,20,30],
  #       "max_features"      : ["auto", "sqrt", "log2"],
  #       "min_samples_split" : [2,4,8],
  #       "bootstrap": [True, False],
  		"randomforestregressor__n_estimators"      : [1, 5, 20],
        "randomforestregressor__max_features"      : ["auto"],
        "randomforestregressor__min_samples_split" : [2, 4],
        "randomforestregressor__bootstrap": [True],
}
