from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras.backend as K
from keras.metrics import binary_crossentropy, binary_accuracy
from keras.optimizers import Adam, SGD
from keras.callbacks import  LearningRateScheduler
from keras.regularizers import l1, l2
from keras.callbacks import Callback
from keras.wrappers.scikit_learn import KerasClassifier

from preprocess import train_data
from sklearn.pipeline import Pipeline
import pandas as pd
from settings import *
from sklearn.model_selection import train_test_split


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

y_data = train_data.Survived
x_data = train_data[[i for i in train_data.keys() if i != 'Survived']]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=.25)


model = Sequential()
model.add(Dense(128, input_shape=(6,), activation='relu', kernel_regularizer=l2(.01), bias_regularizer=l1(.2)))
model.add(Dropout(.3))
model.add(Dense(1028, activation='relu', kernel_regularizer=l2(.01), bias_regularizer=l1(.2)))
model.add(Dropout(.5))
model.add(Dense(1, activation='sigmoid'))

sgd = Adam(lr=0.001)
model.compile(loss=binary_crossentropy, optimizer=sgd,
                  metrics=[binary_crossentropy, binary_accuracy])

model.fit(x_train, y_train,  epochs=700, callbacks=[lr_schedualer, SGDLearningRateTracker(model)])

print(f'train accuracy: {model.score(x_train, y_train)}')
print(f'test accuracy: {model.score(x_test, y_test)}')