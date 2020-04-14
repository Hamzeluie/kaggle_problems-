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


data_train = pd.read_csv(PATH_TRAIN).drop(['Id'], axis=1)
data_test = pd.read_csv(PATH_TEST)
data_test_id = data_test['Id'].values
data_test = data_test.drop(['Id'], axis=1)
data_y = data_train.SalePrice
data_x = data_train.loc[:, data_train.columns != 'SalePrice']


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


lr_schedualer = LearningRateScheduler(schedual)

data_x = transformer(data_x)
data_x = category_encoder(data_x)

data_test = transformer(data_test[data_x.keys()], p=100)
data_test = category_encoder(data_test)

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=.20)

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

history = model.fit(x_train, y_train, epochs=1000, callbacks=[lr_schedualer, SGDLearningRateTracker(model)])
pred = model.predict(x_test)
score = mean_squared_error(y_test, model.predict(x_test))
print(score)


df = pd.DataFrame({'actual': np.array(y_test.values), 'predicted': np.array([i[0] for i in pred])})


plt.plot(range(history.history['loss'].__len__()), np.array(history.history['loss']), 'r--')
# plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()



df1 = df.head(25)
df1.plot(figsize=(16, 10), style='o')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
pd.DataFrame({'Id': data_test_id,
              'SalePrice': [i[0] for i in model.predict(data_test)]}).to_csv('../data/submission2.csv')
exit()
def model_fn():
    model = Sequential()
    model.add(Dense(128, input_shape=(6,), activation='relu', kernel_regularizer=l2(.01), bias_regularizer=l1(.2)))
    model.add(Dropout(.3))
    model.add(Dense(1028, activation='relu', kernel_regularizer=l2(.01), bias_regularizer=l1(.2)))
    model.add(Dropout(.5))
    model.add(Dense(1))

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