
import numpy as np
import pandas as pd
import scipy.io as sio
import os
np.random.seed(0)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import load_model
from keras import backend as K
K.set_image_dim_ordering('tf')

# training configuration
batch_size = 128
nb_epoch = 40

# model choice
fold = 0
model_file = 'model0sex_fold'+str(fold)+'.h5'
new_model = model_file not in os.listdir('.')
nb_classes = 2

label_file = 'adience/labels.csv'
image_file = 'adience/images64g.mat'
img_rows, img_cols = 64, 64

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)


# the data, shuffled and split between train and test sets
image_mat = sio.loadmat(image_file)
label_df = pd.read_csv(label_file)
image_mat = image_mat['images'][label_df.use==1]
label_df = label_df[label_df.use==1]

Y_train = np.transpose(np.asarray((label_df['sex_m'], label_df['sex_f'])))[label_df.fold!=fold]
Y_test = np.transpose(np.asarray((label_df['sex_m'], label_df['sex_f'])))[label_df.fold==fold]

X_train = image_mat[label_df.fold!=fold]
X_test = image_mat[label_df.fold==fold]
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# build and compile model
if new_model:
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
else:
    model = load_model(model_file)


# train model
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# save model
model.save(model_file)
print('Saved to: ', model_file)