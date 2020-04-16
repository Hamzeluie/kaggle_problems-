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


data_test = pd.read_csv(PATH_TEST)
data_test_id = data_test['Id'].values
data_test = data_test.drop(['Id'], axis=1)

data_train = pd.read_csv(PATH_TRAIN).drop(['Id'], axis=1)
data_y = data_train.SalePrice
data_x = data_train.loc[:, data_train.columns != 'SalePrice']


def transformer(x, p=15):
    """
    fill nan categorical value with most frequency value
    and fill nan numerical value with mean value
    x:pandas Dataframe
    p:percentage to valid be nan
    output:DataFrame of filled nan
    """
    def drop_col(p, col_null_count, col_count):
        threshold = p * col_count // 100
        if col_null_count > threshold:
            return True
        else:
            return False

    x = pd.DataFrame({col: x[col].replace('None', np.nan) for col in x.columns})

    col_has_null = [col for col in x.columns if x[col].isnull().any()]
    col_types_dict = {col: (x[col].values.dtype, x[col].isnull().sum()) for col in col_has_null}
    for col in col_has_null:
        col_type, col_null_count = col_types_dict[col]
        if drop_col(p, col_null_count, x[col].__len__()):
            x = x.drop([col], axis=1)
            continue

        if col_type == object:
            x[col] = x[col].fillna(x[col].mode().values[0])
        elif col_type == int:
            x[col] = x[col].fillna(x[col].mean())
        elif col_type == float:
            x[col] = x[col].fillna(round(x[col].mean(), 4))
    return x


def category_encoder(x):
    """
    category encode for categorical columns
    x:pandas Dataframe
    output:DataFrame of filled nan
    """
    col_categories = [col for col in x.columns if x[col].values.dtype == object]
    for col in col_categories:
        transform_dict = {cat: id + 1 for id, cat in enumerate(np.unique(x[col]))}
        value_template = [transform_dict[val] for val in x[col]]
        x[col] = value_template
    return x


data_x = transformer(data_x)
data_x = category_encoder(data_x)

data_test = transformer(data_test)
data_test = category_encoder(data_test)

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=.20)
#
# transformer = Pipeline(steps=[
#     'imputer', SimpleImputer(),
#     'onehot', OneHotEncoder()
# ])
# preprocess = ColumnTransformer(transformers=[
#     ('num', SimpleImputer()),
#     ('cat', transformer)])
xg_model = XGBRegressor()
# my_pipline = Pipeline(steps=[
#     ('preprocess', preprocess),
#     ('model', model)
#     ])
# my_pipline.fit(x_train, y_train)
# pred = my_pipline.predict(x_test)

xg_model.fit(x_train, y_train)

xg_pred = xg_model.predict(x_test)


xg_score = mae(y_test, xg_pred)

pd.DataFrame({'Id': data_test_id,
              'SalePrice': [i for i in xg_model.predict(data_test)]})\
    .to_csv('../data/xgb_submission2.csv')

xg_df = pd.DataFrame({'actual': np.array(y_test.values), 'predicted': np.array(xg_pred)})

df1 = xg_df.head(25)
df1.plot(figsize=(16, 10), style='o')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

exit()


print([i for i in data_train.columns if data_train[i].isnull().any()].__len__())
print(data_train.keys().__len__())
print(data_train.describe(include='object').keys().__len__())
print(data_train.describe(include=['int', 'float']).keys().__len__())