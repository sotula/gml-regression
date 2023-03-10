import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
import joblib
import logging
import config

logger = logging
logger.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def main():
	# Read data
	path_data = config.PATH_DATA 
	path_scores = config.PATH_SCORES 
	path_model = config.PATH_MODEL 

	logger.info('Start reading data')
	df = pd.read_csv(path_data)
	logger.info('Finish reading data')

	logger.info('Start data preprocessing')
	# Drop binary feature 8
	df = df.drop(config.COLUMNS_TO_DROP, axis=1)

	# Split data on train and test dataset
	X_train, X_test, y_train, y_test = train_test_split(df.drop([config.TARGET_COLUMN_NAME], axis=1), 
		df[config.TARGET_COLUMN_NAME], test_size=0.2, random_state=1)
	logger.info('Finish data preprocessing')

	logger.info('Start model training')
	#Setting up a pipeline
	pipe = make_pipeline(
		StandardScaler(),
		# Ridge(),
		RandomForestRegressor()
		)

	#define your rmse 
	mse = make_scorer(mean_squared_error,greater_is_better=False)

	#setting up the grid search
	gs=GridSearchCV(
		pipe,
		config.PIPE_PARAMS,
		n_jobs=-1,
		cv=5,
		# scoring='neg_root_mean_squared_error', 
		scoring=mse,
		verbose=True)
	#fitting gs to training data
	gs.fit(X_train, y_train)

	df_cv_scores=pd.DataFrame(gs.cv_results_).sort_values(by='rank_test_score')
	# Save scores
	df_cv_scores.to_csv(path_scores, index=False)
	logger.info('Finish model training')


	logger.info('Start model saving')
	joblib.dump(gs.best_estimator_, path_model)
	logger.info('Finish model saving')


if __name__ == "__main__":
	main()