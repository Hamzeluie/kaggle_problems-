import pandas as pd
import numpy as np
from settings import *
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from preprocess import dataset


data_test = dataset[dataset['SalePrice'] == 0.]
data_test_id = data_test['Id']
data_test = data_test[[col for col in data_test.keys() if col not in ['SalePrice', 'Id']]]


data_train = dataset[dataset['SalePrice'] > 0.]
y_data = data_train.SalePrice
x_data = data_train[[col for col in data_train.keys() if col not in ['SalePrice', 'Id']]]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=.20)
xg_model = XGBRegressor()
xg_model.fit(x_train, y_train)
xg_train_score = xg_model.score(x_train, y_train)
xg_test_score = xg_model.score(x_test, y_test)
print(f'XGB Model Evaluation :train score: {xg_train_score} and test score: {xg_test_score}')


rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(x_train, y_train)
rf_train_score = rf_model.score(x_train, y_train)
rf_test_score = rf_model.score(x_test, y_test)
print(f'RandomForest Model Evaluation :train score: {rf_train_score} and test score: {rf_test_score}')

pd.DataFrame({'Id': data_test_id, 'SalePrice': xg_model.predict(data_test)}).to_csv('../data/xgb_submit.csv')
pd.DataFrame({'Id': data_test_id, 'SalePrice': rf_model.predict(data_test)}).to_csv('../data/rf_submit.csv')