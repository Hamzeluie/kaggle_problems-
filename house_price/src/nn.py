import pandas as pd
import numpy as np
from settings import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras.backend as K
from keras.metrics import binary_crossentropy, binary_accuracy, mean_squared_error
from keras.optimizers import Adam, SGD
from keras.callbacks import  LearningRateScheduler
from keras.regularizers import l1, l2
from keras.callbacks import Callback
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
from preprocess import dataset


data_test = dataset[dataset['SalePrice'] == 0.]
data_test_id = data_test['Id']
data_test = data_test[[col for col in data_test.keys() if col not in ['SalePrice', 'Id']]]


data_train = dataset[dataset['SalePrice'] > 0.]
y_data = data_train.SalePrice
x_data = data_train[[col for col in data_train.keys() if col not in ['SalePrice', 'Id']]]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=.20)


def schedual(epoch, lr):
    decay_rate = 0.1
    threshold = 120
    if epoch % threshold == 0 and epoch > 0:
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


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=.20)

input_shape = x_train.shape[1]

model = Sequential()
model.add(Dense(512, input_shape=(input_shape,), activation='relu', kernel_regularizer=l2(.01), bias_regularizer=l1(.2)))
model.add(Dropout(.3))
model.add(Dense(2048, activation='relu', kernel_regularizer=l2(.01), bias_regularizer=l1(.2)))
model.add(Dropout(.5))
model.add(Dense(512, activation='relu', kernel_regularizer=l2(.01), bias_regularizer=l1(.2)))
model.add(Dropout(.5))
model.add(Dense(1, kernel_initializer='normal'))

sgd = Adam(lr=0.001)

model.compile(loss='mean_squared_error', optimizer=sgd,
                  metrics=[mean_squared_error])

history = model.fit(x_train, y_train, epochs=1, callbacks=[lr_schedualer, SGDLearningRateTracker(model)])
pred = model.predict(x_test)
score = mean_squared_error(y_test, model.predict(x_test))
print(score)

pd.DataFrame({'Id': data_test_id,
              'SalePrice': [i[0] for i in model.predict(data_test)]}).to_csv('../data/nn_submit.csv')

