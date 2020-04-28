from settings import *
from preprocess import x_train, y_train, x_test, y_test, submit
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy, accuracy
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import Callback, EarlyStopping


class ModelManipulate(Callback):

    def __init__(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs={}):
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

opt = Adam(lr=.001)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, kernel_size=(5, 5), padding='Same', activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.compile(optimizer=opt, loss=categorical_crossentropy, metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, callbacks=[learning_rate_reduction, es,
                                                           ModelManipulate(x_test=x_test, y_test=y_test)])
