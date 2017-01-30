
import numpy as np
import pandas as pd
import scipy.io as sio
import os
import facial_models
np.random.seed(0)

from keras.models import load_model
from keras import backend as K
K.set_image_dim_ordering('tf')

# training configuration
batch_size = 256
nb_epoch = 10
nb_loops = 5

# model choice
fold = 0
model_num = 0
model_file = 'alex'+str(model_num)+'sex_nobabes_fold'+str(fold)+'.h5'
new_model = model_file not in os.listdir('.')

label_file = 'adience/labels_nobabes.csv'
image_file = 'adience/images128rgb.mat'

# the data, shuffled and split between train and test sets
image_mat = sio.loadmat(image_file)
label_df = pd.read_csv(label_file)
image_mat = image_mat['images'][label_df.use==1]
label_df = label_df[label_df.use==1]

Y_train = np.transpose(np.asarray((label_df['sex_m'], label_df['sex_f'])))[label_df.fold!=fold]
Y_test = np.transpose(np.asarray((label_df['sex_m'], label_df['sex_f'])))[label_df.fold==fold]

X_train = image_mat[label_df.fold!=fold]
X_test = image_mat[label_df.fold==fold]
#X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
#X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# build and compile model
if new_model:
    model = facial_models.alexnet(model_num)
else:
    model = load_model(model_file)


# training loop
for i in range(nb_loops):
    print('BEGIN LOOP ', i)

    # train model
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # save model
    model.save(model_file)
    print('Saved to: ', model_file)

    print('END LOOP ', i)