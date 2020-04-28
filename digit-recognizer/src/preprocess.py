import pandas as pd
from settings import *
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical


train = pd.read_csv(PATH_TRAIN)
submit = pd.read_csv(PATH_TEST)
y_data = train.label
x_data = train.drop(['label'], axis=1)

del train
y_data = to_categorical(y_data, num_classes=10)



x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=.25)
x_train = x_train.values.reshape(-1, 28, 28, 1)
x_test = x_test.values.reshape(-1, 28, 28, 1)
submit = submit.values.reshape(-1, 28, 28, 1)



