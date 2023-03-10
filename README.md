# gml-regression
Building regression model

Project contains:
- Data_analysis.ipynb - jupyter notebook with exploratory data analysis;
- config.py - configuration file for choosing model parameters for GridSearchCV, path to data and where save results; 
- train.py - python script for model training. Use GridSearchCV for choosing model;
- predict.py - python script for model inference on test data. Path to test data you can write in config.py;
- requirements.txt - file with requirements for virtual environment;
- /model/test_prediction.csv - file with prediction results for hidden_test.csv;
- /model/scores.csv - file with scores from GridSearchCV;
- /model/test_prediction_for_test_data_from_2.csv - test scores what were used for choosing model.

Add model to zip archive for (/model/best_estimator.7z). 