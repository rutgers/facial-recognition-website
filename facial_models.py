
import numpy as np
np.random.seed(0)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
K.set_image_dim_ordering('tf')

# from MNIST example
def model0sex(nb_filters=32,
              kernel_size=(3,3),
              input_shape=(64,64,1),
              pool_size=(2,2),
              nb_classes=2):
    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    return model

# AlexNet
def alexnet(version=0):
    model = Sequential()

    if version == 0:
        model.add(Convolution2D(nb_filter=64, nb_row=11, nb_col=11, border_mode='same', subsample=(4, 4),
                                input_shape=(128,128,3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

        model.add(Convolution2D(nb_filter=128, nb_row=7, nb_col=7, border_mode='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

        model.add(Convolution2D(nb_filter=192, nb_row=3, nb_col=3, border_mode='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Convolution2D(nb_filter=256, nb_row=3, nb_col=3, border_mode='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(8, activation='relu'))
        model.add(Dropout(0.05))
        model.add(Dense(2, activation='relu'))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adagrad',
                      metrics=['accuracy'])

    return model