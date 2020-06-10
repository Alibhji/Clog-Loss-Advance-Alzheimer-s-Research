# The Code written by Ali Babolhaveji @ 6/9/2020


import sys
import yaml
from tqdm import tqdm
import os
import gc
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
dataset = DataSet(config)



# for data in tqdm(dataset.data_generator(dataset.df_dataset,batch_size=10)):
#     print(data[0].shape ,data[1].shape)



from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import *
from keras.layers import LSTM
from keras.datasets import imdb

model = Sequential()
model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), activation='relu', padding='same'),
                          input_shape=(100, 224, 224, 1)))
model.add(TimeDistributed(Conv2D(32, (3, 3), kernel_initializer="he_normal", activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu')))
model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

model.add(TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu')))
model.add(TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

model.add(TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu')))
model.add(TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

model.add(TimeDistributed(Flatten()))

model.add(Dropout(0.5))
model.add(LSTM(256, return_sequences=False, dropout=0.5))
model.add(Dense(2, activation='softmax'))
model.summary()


data = DataSet(config)
X_train, X_test, y_train, y_test = data.split_train_test()

from keras.utils import multi_gpu_model
from keras.optimizers import Adam, SGD

import tensorflow as tf
import keras.backend.tensorflow_backend as tfback

print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

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




parallel_model = multi_gpu_model(model, gpus=4)
parallel_model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics = ['accuracy'])

batch_size = 32
epochs = 10
size = (224, 224)

train_steps = len(X_train) / batch_size
valid_steps = len(X_test) / batch_size

parallel_model.fit_generator(data.data_generator(X_train, 'standard', size=size, batch_size=batch_size),
                    train_steps, epochs=epochs, verbose=1,
                    validation_data=data.data_generator(X_test, 'standard', size=size, batch_size=batch_size),
                    validation_steps=valid_steps)

# model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
#
# model.fit_generator(data.data_generator(X_train, 'standard', size=size, batch_size=batch_size),
#                     train_steps, epochs=epochs, verbose=1,
#                     validation_data=data.data_generator(X_test, 'standard', size=size, batch_size=batch_size),
#                     validation_steps=valid_steps)