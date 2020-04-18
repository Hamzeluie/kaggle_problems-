import pandas as pd
import numpy as np
from settings import *  # contain data path
import seaborn as sns
import matplotlib.pyplot as plt

interest_features = ['Id', 'GarageCond', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual',
'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual',
'MasVnrArea', 'MasVnrType', 'Electrical', 'Utilities', 'YearRemodAdd',
'MSSubClass', 'Foundation', 'ExterCond', 'ExterQual', 'Exterior2nd',
'Exterior1st', 'RoofMatl', 'RoofStyle', 'YearBuilt', 'LotConfig',
'OverallCond', 'OverallQual', 'HouseStyle', 'BldgType', 'Condition2',
'BsmtFinSF1', 'MSZoning', 'LotArea', 'Street', 'Condition1',
'Neighborhood', 'LotShape', 'LandContour', 'LandSlope', 'SalePrice',
'HeatingQC', 'BsmtFinSF2', 'EnclosedPorch', 'Fireplaces', 'GarageCars',
'GarageArea', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', '3SsnPorch',
'BsmtUnfSF', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold',
'SaleType', 'Functional', 'TotRmsAbvGrd', 'KitchenQual', 'KitchenAbvGr',
'BedroomAbvGr', 'HalfBath', 'FullBath', 'BsmtHalfBath', 'BsmtFullBath',
'GrLivArea', 'LowQualFinSF', '2ndFlrSF', '1stFlrSF', 'CentralAir',
'SaleCondition', 'Heating', 'TotalBsmtSF']

train_data = pd.read_csv(PATH_TRAIN)
test_data = pd.read_csv(PATH_TEST)
test_data['SalePrice'] = np.zeros([test_data.__len__(), 1])
dataset = pd.concat([train_data, test_data])

# get feature names with less than 15% null value
# total = dataset.isnull().sum().sort_values(ascending=False)
# percent = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)
# missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print(missing_data[missing_data['Percent'] < .15].index)
dataset = dataset[interest_features]
# print(dataset.describe(include=[object]).info())
obj_null_count = pd.DataFrame(dataset.__len__() - dataset.describe(include=[object]).loc['count'].sort_values(ascending=False))
obj_null_zeros_cols = ['Heating', 'MSZoning', 'Utilities', 'Foundation', 'ExterCond', 'ExterQual', 'Exterior2nd',
                       'Exterior1st', 'RoofMatl', 'SaleCondition', 'LotConfig', 'HouseStyle', 'BldgType', 'Condition2',
                       'RoofStyle', 'Street', 'PavedDrive', 'Condition1', 'KitchenQual', 'Functional', 'SaleType',
                       'CentralAir', 'HeatingQC', 'LandSlope', 'LandContour', 'LotShape', 'Neighborhood']
# convert category to numerical
dataset['Heating'] = dataset['Heating'].map({'GasA': 1, 'GasW': 2, 'Grav': 3, 'Wall': 4, 'OthW': 5, 'Floor': 6})
dataset['Foundation'] = dataset['Foundation'].map({'PConc': 1, 'CBlock': 2, 'BrkTil': 3, 'Wood': 4, 'Slab': 5, 'Stone': 6})
dataset['RoofMatl'] = dataset['RoofMatl'].map({'CompShg': 1, 'WdShngl': 2, 'Metal': 3, 'WdShake': 4, 'Membran': 5, 'Tar&Grv': 6, 'Roll': 7, 'ClyTile': 8})
dataset['SaleCondition'] = dataset['SaleCondition'].map({'Normal': 1, 'Abnorml': 2, 'Partial': 3, 'AdjLand': 4, 'Alloca': 5, 'Family': 6})
dataset['LotConfig'] = dataset['LotConfig'].map({'Inside': 1, 'FR2': 2, 'Corner': 3, 'CulDSac': 4, 'FR3': 5})
dataset['HouseStyle'] = dataset['HouseStyle'].map({'2Story': 1, '1Story': 2, '1.5Fin': 3, '1.5Unf': 4, 'SFoyer': 5, 'SLvl': 6, '2.5Unf': 7, '2.5Fin': 8})
dataset['BldgType'] = dataset['BldgType'].map({'1Fam': 1, '2fmCon': 2, 'Duplex': 3, 'TwnhsE': 4, 'Twnhs': 5})
dataset['Street'] = dataset['Street'].map({'Pave': 1, 'Grvl': 2})
dataset['RoofStyle'] = dataset['RoofStyle'].map({'Gable': 1, 'Hip': 2, 'Gambrel': 3, 'Mansard': 4, 'Flat': 5, 'Shed': 6})
dataset['LandSlope'] = dataset['LandSlope'].map({'Gtl': 1, 'Mod': 2, 'Sev': 3})
dataset['Neighborhood'] = dataset['Neighborhood'].map({city: k + 1 for k, city in enumerate(
    ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 'NWAmes', 'OldTown', 'BrkSide', 'Sawyer',
     'NridgHt', 'NAmes', 'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert', 'StoneBr', 'ClearCr',
     'NPkVill', 'Blmngtn', 'BrDale', 'SWISU', 'Blueste'])})

dataset['PavedDrive'] = dataset['PavedDrive'].map({'Y': 1, 'N': 2, 'P': 3})
dataset['HeatingQC'] = dataset['HeatingQC'].map({'Ex': 1, 'Gd': 2, 'TA': 3, 'Fa': 4, 'Po': 5})
dataset['CentralAir'] = dataset['CentralAir'].map({'Y': 1, 'N': 2})
dataset['LandContour'] = dataset['LandContour'].map({'Lvl': 1, 'Bnk': 2, 'Low': 3, 'HLS': 4})
dataset['LotShape'] = dataset['LotShape'].map({'Reg': 1, 'IR1': 2, 'IR2': 3, 'IR3': 4})
# fillna and convertion
dataset['SaleType'] = dataset.SaleType.fillna(dataset.SaleType.dropna().mode()[0])
dataset['SaleType'] = dataset['SaleType'].map({v: k + 1 for k, v in enumerate(
    ['WD', 'New', 'COD', 'ConLD', 'ConLI', 'CWD', 'ConLw', 'Con', 'Oth'])})

# fillna and convertion
dataset['KitchenQual'] = dataset.KitchenQual.fillna(dataset.KitchenQual.dropna().mode()[0])
dataset['KitchenQual'] = dataset['KitchenQual'].map({'Gd': 1, 'TA': 2, 'Ex': 3, 'Fa': 4})

# fillna and convertion
dataset['Electrical'] = dataset.Electrical.fillna(dataset.Electrical.dropna().mode()[0])
dataset['Electrical'] = dataset['Electrical'].map({'SBrkr': 1, 'FuseF':2, 'FuseA': 3, 'FuseP': 4, 'Mix': 5})
# fillna and convertion
dataset['Functional'] = dataset.Functional.fillna(dataset.Functional.dropna().mode()[0])
dataset['Functional'] = dataset['Functional'].map({'Typ': 1, 'Min1': 2, 'Maj1': 3, 'Min2': 4, 'Mod': 5, 'Maj2': 6, 'Sev': 7})
# fillna and convertion
dataset['MSZoning'] = dataset.MSZoning.fillna(dataset.MSZoning.dropna().mode()[0])
dataset['MSZoning'] = dataset['MSZoning'].map({'RL': 1, 'RM': 2, 'C (all)': 3, 'FV': 4, 'RH': 5})

# fillna and convertion
dataset['Utilities'] = dataset.Utilities.fillna(dataset.Utilities.dropna().mode()[0])
dataset['Utilities'] = dataset['Utilities'].map({'AllPub': 1, 'NoSeWa': 2})
# create has Garage and drop Garage s features
dataset['HasGarage'] = 0
dataset.loc[dataset['GarageCars'] > 0, 'HasGarage'] = 1
dataset = dataset.drop(['GarageCond', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCars',
                        'GarageArea'], axis=1)

def map_merger_column(col1, col2, new_col_name, dataset):
    state = list(
        dataset[[col1, col2]].groupby(by=[col1, col2]).nunique().index)
    dict = {v: k + 1 for k, v in enumerate(state)}
    dataset[new_col_name] = 0
    for (c1, c2), v in dict.items():
        dataset.loc[(dataset[col1] == c1) & (dataset[col2] == c2), new_col_name] = v
    dataset = dataset.drop([col1, col2], axis=1)
    return dataset

# create Condition  and drop Condition1 and Condition2
dataset = map_merger_column('Condition1', 'Condition2', 'Conditions', dataset)
# create Exter  and drop ExterQual and ExterCond

dataset = map_merger_column('ExterQual', 'ExterCond', 'Exter', dataset)
# Exterior1st and Exterior2nd have data entry error first fix them
# fill Exterior1st and Exterior2nd nan and
# create Exterior  and drop Exterior1st and Exterior2nd
# [i for i in dataset['Exterior2nd'].unique() if i not in dataset['Exterior1st'].unique()]
for k, v in {'Wd Sdng': 'WdShing', 'Wd Shng': 'WdShing', 'CmentBd': 'CemntBd', 'Brk Cmn': 'BrkComm'}.items():
    dataset['Exterior1st'] = dataset['Exterior1st'].replace(to_replace=k, value=v)
    dataset['Exterior2nd'] = dataset['Exterior2nd'].replace(to_replace=k, value=v)

dataset['Exterior1st'] = dataset.Exterior1st.fillna(dataset.Exterior1st.dropna().mode()[0])
dataset['Exterior2nd'] = dataset.Exterior2nd.fillna(dataset.Exterior2nd.dropna().mode()[0])

dataset = map_merger_column('Exterior1st', 'Exterior2nd', 'Exterior', dataset)
"""Basement analyz"""
basement_features = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2',
                     'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
# fill nan 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1' and 'BsmtFinType2'
dataset['BsmtQual'] = dataset.BsmtQual.fillna(dataset.BsmtQual.dropna().mode()[0])
dataset['BsmtCond'] = dataset.BsmtCond.fillna(dataset.BsmtCond.dropna().mode()[0])
dataset['BsmtExposure'] = dataset.BsmtExposure.fillna(dataset.BsmtExposure.dropna().mode()[0])
dataset['BsmtFinType1'] = dataset.BsmtFinType1.fillna(dataset.BsmtFinType1.dropna().mode()[0])
dataset['BsmtFinType2'] = dataset.BsmtFinType2.fillna(dataset.BsmtFinType2.dropna().mode()[0])
# map BsmtExposure
dataset['BsmtExposure'] = dataset.BsmtExposure.map({'No': 1, 'Gd': 2, 'Mn': 3, 'Av': 4})
# merge and map BsmtQual , BsmtCond and create BsmtQC
dataset = map_merger_column('BsmtQual', 'BsmtCond', 'BsmtQC', dataset)
# merge and map BsmtFinType1 , BsmtFinType2 and create BsmtFinType
dataset = map_merger_column('BsmtFinType1', 'BsmtFinType2', 'BsmtFinType', dataset)
# drop 'BsmtFullBath', 'BsmtHalfBath' and create 'BsmtBath'
dataset = map_merger_column('BsmtFullBath', 'BsmtHalfBath', 'BsmtBath', dataset)
# these 4 column are related to each other directly and have just one nan value so randomly fill that one
# 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF'
dataset['BsmtFinSF1'] = dataset['BsmtFinSF1'].fillna(706.)
dataset['BsmtFinSF2'] = dataset['BsmtFinSF2'].fillna(0.)
dataset['BsmtUnfSF'] = dataset['BsmtUnfSF'].fillna(150.)
dataset['TotalBsmtSF'] = dataset['TotalBsmtSF'].fillna(856.)

"""End of Basement"""
# fill na MasVnrType and create HasMasVnr and finally drop MasVnrType and MasVnrArea
dataset['MasVnrType'] = dataset.MasVnrType.fillna(dataset.MasVnrType.dropna().mode()[0])

dataset['HasMasVnr'] = 0
dataset.loc[dataset['MasVnrType'] != 'None', 'HasMasVnr'] = 1
dataset = dataset.drop(['MasVnrType', 'MasVnrArea'], axis=1)
# ==================================numerical cols analyzing
new_cols = ['Id', 'HasGarage', 'Conditions', 'Exter', 'Exterior', 'BsmtQC', 'BsmtFinType', 'HasMasVnr', 'BsmtBath']
num_cols = [col for col in dataset.keys() if col not in list(obj_null_count.index) + new_cols]

dataset.loc[dataset['YearRemodAdd'] <= 1950, 'YearRemodAdd'] = 0
dataset.loc[(dataset['YearRemodAdd'] > 1950) & (dataset['YearRemodAdd'] <= 1970), 'YearRemodAdd'] = 1
dataset.loc[(dataset['YearRemodAdd'] > 1970) & (dataset['YearRemodAdd'] <= 1990), 'YearRemodAdd'] = 2
dataset.loc[(dataset['YearRemodAdd'] > 1990) & (dataset['YearRemodAdd'] <= 2010), 'YearRemodAdd'] = 3
dataset['YearRemodAdd'] = dataset['YearRemodAdd'].astype(int)

num_null_count = dataset[num_cols].describe(include=[int])
