import pandas as pd
import numpy as np
from settings import *  # contain data path


train_data = pd.read_csv(TRAIN_PATH)
train_data = train_data.drop(['Cabin'], axis=1)
train_data = train_data.drop(['PassengerId', 'Name'], axis=1)

train_data = train_data.drop(['Ticket'], axis=1)
freq_port = train_data.Embarked.dropna().mode()[0]
train_data['Embarked'] = train_data['Embarked'].fillna(freq_port)
train_data['Embarked'] = train_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

train_data['Sex'] = train_data['Sex'].map({'female': 1, 'male': 0}).astype(int)
guess_ages = np.zeros((2, 3))
for i in range(0, 2):
    for j in range(0, 3):

        guess_df = train_data[(train_data['Sex'] == i) & \
                           (train_data['Pclass'] == j + 1)]['Age'].dropna()

        age_guess = guess_df.median()

        # Convert random age float to nearest .5 age
        guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

for i in range(0, 2):
    for j in range(0, 3):
        train_data.loc[(train_data.Age.isnull()) & (train_data.Sex == i) & (train_data.Pclass == j + 1), \
                    'Age'] = guess_ages[i, j]

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
