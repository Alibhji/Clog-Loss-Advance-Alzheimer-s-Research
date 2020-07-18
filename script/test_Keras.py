# The Code written by Ali Babolhaveji @ 6/9/2020


import sys
import yaml
import  pickle
import os
import pandas as pd


package_path = '..'
# package_path = '../../script2/Result/nuScenes/exp3/sources/'
if not package_path in sys.path:
    sys.path.append(package_path)

from Klib import  DataSet


config = './config.yml'
with open (config , 'rb') as f:
    config = yaml.load(f ,Loader=yaml.FullLoader)

# print(config)



# for data in tqdm(dataset.data_generator(dataset.df_dataset,batch_size=10)):
#     print(data[0].shape ,data[1].shape)



from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import *
from keras.layers import LSTM
from keras.datasets import imdb
from keras import backend as K


from keras.utils import multi_gpu_model
from keras.optimizers import Adam, SGD

import tensorflow as tf
import keras.backend.tensorflow_backend as tfback

from keras.callbacks import ModelCheckpoint
from keras import models

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)


def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed

#
#
# model = Sequential()
# model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), activation='relu', padding='same'),
#                           input_shape=(100, 224, 224, 1)))
# model.add(TimeDistributed(Conv2D(32, (3, 3), kernel_initializer="he_normal", activation='relu')))
# model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
#
# model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
# model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
# model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
#
# model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu')))
# model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu')))
# model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
#
# model.add(TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu')))
# model.add(TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu')))
# model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
#
# model.add(TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu')))
# model.add(TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu')))
# model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
#
# model.add(TimeDistributed(Flatten()))
#
# model.add(Dropout(0.5))
# model.add(LSTM(256, return_sequences=False, dropout=0.5))
# model.add(Dense(2, activation='softmax'))
# model.summary()

#weights_path = '/home/mjamali/proj/B/Clog/script/weights-improvement-09-0.59.hdf5'
#weights_path = '/home/mjamali/proj/B/Clog/script/weights-improvement-04-0.88.hdf5'  # -->3
# weights_path = "/home/mjamali/proj/B/Clog/script/weights-improvement-05-0.92.hdf5"   # -->4
# weights_path = "/home/mjamali/proj/B/Clog/script/weights-improvement-02-0.74.hdf5"   # -->5
# weights_path = "/home/mjamali/proj/B/Clog/script/result_6th/weights-improvement-10-0.97.hdf5"   # -->6
weights_path = "/home/mjamali/proj/B/Clog/script/result_6th/weights-improvement-08-0.97.hdf5"   # -->7


model = models.load_model(weights_path , custom_objects={'FocalLoss': focal_loss, 'focal_loss_fixed': focal_loss()})

model.summary()

data = DataSet(config , datatype = 'test')
# X_train, X_test, y_train, y_test = data.split_train_test()
X_test = data.df_dataset


print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)
#
def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]


tfback._get_available_gpus = _get_available_gpus

#tf.config.experimental_list_devices()
print(tf.config.list_logical_devices())







# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]



batch_size = 8

size = (224, 224)

# train_steps = len(X_train) / batch_size
test_steps = len(X_test) / batch_size


parallel_model = multi_gpu_model(model, gpus=4)

ynew  =model.predict_generator(data.data_generator(X_test, 'standard', size=size, batch_size=batch_size)
              , steps= test_steps ,verbose=1)

print(ynew)

with open('model_output.out','wb') as file:
    pickle.dump(ynew, file, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Model estimation is saved at model_output.out")




submit = pd.read_csv('../../data/submission_format.csv')
submit['stalled'] = ynew[:,0]
submit['stalled']  = submit.apply(lambda row: [1,0] [row.stalled>=0.5] ,axis=1)
folder = 'result_6th'
save_csv= os.path.join(folder,os.path.splitext(os.path.basename(weights_path))[0]+'.csv')
submit.to_csv(save_csv,index=False)
# parallel_model.compile(optimizer=Adam(lr=0.00005), loss='binary_crossentropy', metrics = ['accuracy'])

# parallel_model.fit_generator(data.data_generator(X_train, 'standard', size=size, batch_size=batch_size),
#                     train_steps, epochs=epochs, callbacks=callbacks_list, verbose=1,
#                     validation_data=data.data_generator(X_test, 'standard', size=size, batch_size=batch_size),
#                     validation_steps=valid_steps) #, workers=30 ,use_multiprocessing= True)

# model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
#
# model.fit_generator(data.data_generator(X_train, 'standard', size=size, batch_size=batch_size),
#                     train_steps, epochs=epochs, verbose=1,
#                     validation_data=data.data_generator(X_test, 'standard', size=size, batch_size=batch_size),
#                     validation_steps=valid_steps)
