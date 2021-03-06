import pandas as pd
import numpy as np
from settings import *  # contain data path
import seaborn as sns
import matplotlib.pyplot as plt

# step 1
train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)
dataset = pd.concat([train_data, test_data])
print(dataset.keys())
print(dataset.describe())
print(dataset.describe(include=[object]))
print(dataset.info())

null_percentage = lambda x: 100*x.isnull().sum()//x.__len__()
for col in dataset.keys():
    print(f'{col}: {null_percentage(dataset[col])}')
print(dataset.describe(include=[int, float]))
# step 2
dataset = dataset.drop(['Cabin'], axis=1)
dataset = dataset.drop(['PassengerId', 'Name'], axis=1)

train_data = train_data.drop(['Ticket'], axis=1)
freq_port = train_data.Embarked.dropna().mode()[0]
train_data['Embarked'] = train_data['Embarked'].fillna(freq_port)
train_data['Embarked'] = train_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

grid = sns.FacetGrid(train_data, col='Sex', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()
train_data['Sex'] = train_data['Sex'].map({'female': 1, 'male': 0}).astype(int)

for sex in range(0, 2):
    for pclass in range(0, 3):
        target_ages = train_data[(train_data['Sex'] == sex) & \
                           (train_data['Pclass'] == pclass + 1)]['Age'].dropna()
        age_guess = target_ages.median()
        train_data.loc[(train_data.Age.isnull()) & (train_data.Sex == sex) & (train_data.Pclass == pclass + 1), \
                       'Age'] = age_guess

train_data['Age'] = train_data['Age'].astype(int)

train_data['AgeBand'] = pd.cut(train_data['Age'], 5)
train_data.loc[ train_data['Age'] <= 16, 'Age'] = 0
train_data.loc[(train_data['Age'] > 16) & (train_data['Age'] <= 32), 'Age'] = 1
train_data.loc[(train_data['Age'] > 32) & (train_data['Age'] <= 48), 'Age'] = 2
train_data.loc[(train_data['Age'] > 48) & (train_data['Age'] <= 64), 'Age'] = 3
train_data.loc[ train_data['Age'] > 64, 'Age'] = 4

train_data = train_data.drop(['AgeBand'], axis=1)

train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
train_data['IsAlone'] = 0
train_data.loc[train_data['FamilySize'] == 1, 'IsAlone'] = 1
train_data = train_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
train_data['FareBand'] = pd.qcut(train_data['Fare'], 4)
train_data.loc[ train_data['Fare'] <= 7.91, 'Fare'] = 0
train_data.loc[(train_data['Fare'] > 7.91) & (train_data['Fare'] <= 14.454), 'Fare'] = 1
train_data.loc[(train_data['Fare'] > 14.454) & (train_data['Fare'] <= 31), 'Fare']   = 2
train_data.loc[ train_data['Fare'] > 31, 'Fare'] = 3
train_data['Fare'] = train_data['Fare'].astype(int)

train_data = train_data.drop(['FareBand'], axis=1)
