import pandas as pd
import numpy as np
from settings import *  # contain data path
import seaborn as sns
import matplotlib.pyplot as plt

# read dataset from TRAIN_PATH
train_data = pd.read_csv(TRAIN_PATH)
print(train_data)

# Which features are available in the dataset?
print(train_data.columns)
# Which features are numerical?
print(train_data.describe(include=[int, float]))
# Which features are categorical?
print(train_data.describe(include=[object]))
# Which features contain blank, null or empty values?
print(train_data.info())
#as you can see cabin has so many null value ~77% so you can drop cabin
train_data = train_data.drop(['Cabin'], axis=1)
# PassengerId, Name have cont contribute to goal of problem(survived) so we can also drop them
train_data = train_data.drop(['PassengerId', 'Name'], axis=1)
print(train_data.describe(include=[object]))

#Ticket has 681 unique value which is so many.
# so we have struggle to encode that to numerical.
# so i think it is better to drop it too.
train_data = train_data.drop(['Ticket'], axis=1)
# this is our final dataset till now
print(train_data.info())
# Age and Embarked have null value. so we sould deal with them.
# fill Embarked with most the most common occurance.
freq_port = train_data.Embarked.dropna().mode()[0]
train_data['Embarked'] = train_data['Embarked'].fillna(freq_port)
# now convert Embarked to numerical
train_data['Embarked'] = train_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# fill Age with respect of other feature correlation.
grid = sns.FacetGrid(train_data, col='Sex', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
plt.show()
# with respect of the above charts you can see correlation between age, sex and pclass.
# so we can fill age null values with respect of pclass and sex.
# but before that we should convert categorical value of sex to numerical
train_data['Sex'] = train_data['Sex'].map({'female': 1, 'male': 0}).astype(int)
# so now we can fill null values of age
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
# We may want to create new feature for Age bands. This turns a continous numerical feature into an ordinal categorical feature.
train_data['AgeBand'] = pd.cut(train_data['Age'], 5)
print(train_data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand',
                                                                                                  ascending=True))
train_data.loc[ train_data['Age'] <= 16, 'Age'] = 0
train_data.loc[(train_data['Age'] > 16) & (train_data['Age'] <= 32), 'Age'] = 1
train_data.loc[(train_data['Age'] > 32) & (train_data['Age'] <= 48), 'Age'] = 2
train_data.loc[(train_data['Age'] > 48) & (train_data['Age'] <= 64), 'Age'] = 3
train_data.loc[ train_data['Age'] > 64, 'Age'] = 4

train_data = train_data.drop(['AgeBand'], axis=1)
# We may want to create a new feature called Family based on Parch and SibSp to get total count of family members on board.
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
# now lets have some analyze
print(train_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived',
                                                                                                      ascending=False))
# We can create another feature called IsAlone.
train_data['IsAlone'] = 0
train_data.loc[train_data['FamilySize'] == 1, 'IsAlone'] = 1
# lets have some analyze on IsAlone
print(train_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
# Let us drop Parch, SibSp, and FamilySize features in favor of IsAlone.
train_data = train_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
# We may also want to create a Fare range feature if it helps our analysis.
train_data['FareBand'] = pd.qcut(train_data['Fare'], 4)
print(train_data[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand',
                                                                                                  ascending=True))
train_data.loc[ train_data['Fare'] <= 7.91, 'Fare'] = 0
train_data.loc[(train_data['Fare'] > 7.91) & (train_data['Fare'] <= 14.454), 'Fare'] = 1
train_data.loc[(train_data['Fare'] > 14.454) & (train_data['Fare'] <= 31), 'Fare']   = 2
train_data.loc[ train_data['Fare'] > 31, 'Fare'] = 3
train_data['Fare'] = train_data['Fare'].astype(int)

train_data = train_data.drop(['FareBand'], axis=1)


# split data to TRAIN and TEST
from sklearn.model_selection import train_test_split
y_data = train_data.Survived
x_data = train_data[[i for i in train_data.keys() if i != 'Survived']]
xtrain, xtest, ytrain, ytest = train_test_split(x_data, y_data, test_size=.2)
# now feed data to model
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(xtrain, ytrain)
acc_random_forest = round(random_forest.score(xtrain, ytrain) * 100, 2)
print(acc_random_forest)
print(round(random_forest.score(xtest, ytest) * 100, 2))
# you can use Pipline to have cleaner code
from sklearn.pipeline import Pipeline

my_pipeline = Pipeline(steps=[
    ('model', acc_random_forest)
])
my_pipeline.fit(xtrain, ytrain)
print(my_pipeline.score(xtest, ytest))

# neural network model in this tutorial i do not work on NN so much
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.metrics import binary_crossentropy, binary_accuracy
from keras.optimizers import Adam, SGD
from keras.regularizers import l1, l2


model = Sequential()
model.add(Dense(128, input_shape=(6,), activation='relu', kernel_regularizer=l2(.01), bias_regularizer=l1(.2)))
model.add(Dropout(.3))
model.add(Dense(1028, activation='relu', kernel_regularizer=l2(.01), bias_regularizer=l1(.2)))
model.add(Dropout(.5))
model.add(Dense(1, activation='sigmoid'))

sgd = Adam(lr=0.001)
model.compile(loss=binary_crossentropy, optimizer=sgd,
                  metrics=[binary_crossentropy, binary_accuracy])

model.fit(xtrain, ytrain,  epochs=700)
score = binary_accuracy(ytest, model.predict(xtest))
print(score)