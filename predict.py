import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
import joblib
import logging
import config

logger = logging
logger.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def main():
	# Read data
	path_test_data = config.PATH_TEST_DATA
	path_pred = config.PATH_PRED  
	path_model = config.PATH_MODEL 

	logger.info('Start reading data')
	df = pd.read_csv(path_test_data)
	logger.info('Finish reading data')

	logger.info('Start data preprocessing')
	# Drop binary feature 8
	df = df.drop(config.COLUMNS_TO_DROP, axis=1)
	logger.info('Finish data preprocessing')

	logger.info('Start load model')
	model = joblib.load(path_model)
	logger.info('Finish load model')	

	logger.info('Start get model prediction')
	prediction = model.predict(df)
	result = pd.DataFrame({'prediction': prediction})
	logger.info('Finish get model prediction')


	logger.info('Start saving scores')
	result.to_csv(path_pred, index=False)
	logger.info('Finish saving scores')


if __name__ == "__main__":
	main()