from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras.backend as K
from keras.metrics import binary_crossentropy, binary_accuracy
from keras.optimizers import Adam, SGD
from keras.callbacks import  LearningRateScheduler
from keras.regularizers import l1, l2
from keras.callbacks import Callback
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from settings import *
from sklearn.model_selection import train_test_split
import os


def schedual(epoch, lr):
    decay_rate = 0.1
    if epoch == 500:
        return lr * decay_rate
    elif epoch == 700:
        return lr * decay_rate
    else:
        return lr


class SGDLearningRateTracker(Callback):
    def __init__(self, model):
        self.model = model

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr)
        print('\nLR: {:.6f}\n'.format(lr))
lr_schedualer = LearningRateScheduler(schedual)
train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
ffeatures = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp',
            'Parch', 'Embarked']
ttest_data = test_data[ffeatures]
test_data = test_data[features]


y_data = train_data.Survived
x_data = train_data[features]
x_data['Embarked'] = x_data['Embarked'].fillna('S')

test_data['Embarked'] = test_data['Embarked'].fillna('S')

embarked_dict = {'C': 0, 'Q': 1, 'S': 2}
embarked = [embarked_dict[i] for i in x_data['Embarked']]
test_embarked = [embarked_dict[i] for i in test_data['Embarked']]

x_data['Embarked'] = embarked
test_data['Embarked'] = test_embarked

sex_dict = {'female': 0, 'male': 1}
sex = [sex_dict[i] for i in x_data['Sex']]
test_sex = [sex_dict[i] for i in test_data['Sex']]

x_data['Sex'] = sex
test_data['Sex'] = test_sex

x_data['Age'] = x_data['Age'].fillna(29.)
test_data['Age'] = test_data['Age'].fillna(29.)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=.25)


model = Sequential()
model.add(Dense(128, input_shape=(6,), activation='relu', kernel_regularizer=l2(.01), bias_regularizer=l1(.2)))
model.add(Dropout(.3))
model.add(Dense(1028, activation='relu', kernel_regularizer=l2(.01), bias_regularizer=l1(.2)))
model.add(Dropout(.5))
model.add(Dense(1, activation='sigmoid'))

sgd = Adam(lr=0.001)

def model_fn():
    model = Sequential()
    model.add(Dense(128, input_shape=(6,), activation='relu', kernel_regularizer=l2(.01), bias_regularizer=l1(.2)))
    model.add(Dropout(.3))
    model.add(Dense(1028, activation='relu', kernel_regularizer=l2(.01), bias_regularizer=l1(.2)))
    model.add(Dropout(.5))
    model.add(Dense(1, activation='sigmoid'))

    sgd = Adam(lr=0.001)

    model.compile(loss=binary_crossentropy, optimizer=sgd,
                  metrics=[binary_crossentropy, binary_accuracy])
    return model


clf = KerasClassifier(build_fn=model_fn)
# just create the pipeline
pipeline = Pipeline([
    ('clf', clf)
])



pipeline.fit(x_train, y_train,  epochs=700, callbacks=[lr_schedualer, SGDLearningRateTracker(model)])

score = binary_accuracy(y_test, pipeline.predict(x_test))
"""
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score, KFold
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
seed = 1

diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]

def baseline_model():
    model = Sequential()
    model.add(Dense(10, input_dim=10, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=100, verbose=False)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

estimator.fit(X, y)
prediction = estimator.predict(X)
accuracy_score(y, prediction)
"""