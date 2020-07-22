# The Code written by Ali Babolhaveji @ 7/18/2020
#https://towardsdatascience.com/prototyping-an-anomaly-detection-system-for-videos-step-by-step-using-lstm-convolutional-4e06b7dcdd29

import sys
import yaml
from tqdm import tqdm
import os
import gc
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

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


from keras.utils import multi_gpu_model
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras import models
import tensorflow as tf
from keras.models import Sequential,Model

import keras.backend.tensorflow_backend as tfback

from keras.callbacks import ModelCheckpoint
from keras_layer_normalization import LayerNormalization



# physical_devices = tf.config.list_logical_devices()
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
#
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)








data = DataSet(config)
X_train, X_test, y_train, y_test = data.split_train_test()
# X_test = data.df_dataset


print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)
#
# def _get_available_gpus():
#     """Get a list of available gpu devices (formatted as strings).
#
#     # Returns
#         A list of available GPU devices.
#     """
#     #global _LOCAL_DEVICES
#     if tfback._LOCAL_DEVICES is None:
#         devices = tf.config.list_logical_devices()
#         tfback._LOCAL_DEVICES = [x.name for x in devices]
#     return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]
#
#
# tfback._get_available_gpus = _get_available_gpus

#tf.config.experimental_list_devices()
# print(tf.config.list_logical_devices())







# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_acc:.3f}.hdf5"
# filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"

# checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]


def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed

batch_size = 16
epochs = 20
size = (56, 56)

train_steps = len(X_train) / batch_size
valid_steps = len(X_test) / batch_size

weights_path = "/home/mjamali/proj/B/Clog/script/result_6th/weights-improvement-08-0.97.hdf5"   # -->5
# model = models.load_model(weights_path).save("my_h5_model.h5")
from keras import initializers, regularizers, constraints

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


def get_model():
    HIDDEN_UNITS = 512
    expected_frames = 60
    num_input_tokens = 56 * 56
    nb_classes = 2

    input_shape = (60, 56, 56)

    # inputs = Input(input_shape)

    model = Sequential()
    model.add(Reshape((expected_frames, -1), input_shape=input_shape))
    model.add(Bidirectional(LSTM(units=HIDDEN_UNITS, return_sequences=True)))
    model.add(Bidirectional(LSTM(10)))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))

    model.add(Activation('softmax'))



    # input_shape = (60,56, 56)
    # nb_classes = 2
    #
    # model = Sequential()
    # model.add(Conv2D(filters=32, input_shape=input_shape, padding='same', kernel_size=(3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(filters=32, padding='same', kernel_size=(3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Dropout(rate=0.25))
    #
    # model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(filters=64, padding='same', kernel_size=(3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Dropout(rate=0.25))
    #
    # model.add(Flatten())
    # model.add(Dense(units=512))
    # model.add(Activation('relu'))
    # model.add(Dropout(rate=0.5))
    # model.add(Dense(units=nb_classes))
    # model.add(Activation('softmax'))

    # model = Sequential()
    # model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), return_sequences=False, data_format="channels_last",
    #                      input_shape=(60, 56, 56, 1)))
    # # model.add(LayerNormalization())
    # model.add(Dropout(0.2))
    # model.add((AveragePooling2D(pool_size=(25, 25))))
    #
    # model.add(Flatten())
    #
    # model.add(Dense(4096, activation="relu"))
    # # model.add(LayerNormalization())
    # model.add(Dropout(0.3))
    # model.add(Dense(2048, activation="relu"))
    # # model.add(LayerNormalization())
    # model.add(Dropout(0.3))
    # model.add(Dense(1024, activation="relu"))
    # model.add(Dropout(0.3))
    # model.add(Dense(256, activation="relu"))
    # model.add(Dense(2, activation="softmax"))

    # #######
    # inp = Input(shape=(60, 224, 224, 1))
    #
    # x = ConvLSTM2D(filters=60, kernel_size=(3, 3), return_sequences=False, data_format="channels_last")(inp)
    # x = Dropout(0.2)(x)
    # x = AveragePooling2D(pool_size=(25, 25))(x)
    # x = Reshape((-1, 60))(x)
    # x = Bidirectional(LSTM(60, dropout=0.4, recurrent_dropout=0.4, activation='relu', return_sequences=True))(x)
    # x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    # x = Attention(64)(x)
    # x = Dense(256, activation="relu")(x)
    # # x = Dropout(0.25)(x)
    # x = Dense(2, activation="sigmoid")(x)
    #
    # model = Model(inputs=inp, outputs=x)
    # #########


    # ### doesn't train
    # model = Sequential()
    # model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu', input_shape=(60, 224, 224, 1)))
    # model.add(MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2)))
    # model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2)))
    # model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2)))
    # # model.add(Conv3D(256, (3,3,3), activation='relu'))
    # # model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
    # model.add(Conv3D(256, (3, 3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2)))
    # model.add(Conv3D(128, (3, 3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2)))
    # model.add(Conv3D(128, (3, 3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2)))
    # model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu'))
    #
    # model.add(Reshape((60, -1)))
    # model.add(BatchNormalization(input_shape=(60, 256)))
    # model.add(Bidirectional(LSTM(256, dropout=0.4, recurrent_dropout=0.4, activation='relu', return_sequences=True)))
    # model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True)))
    # model.add(Attention(60))
    # model.add(Dense(2, activation='sigmoid'))
    # # model.summary()
    # ### doesn't train

    # seq = Sequential()
    # # seq.add(TimeDistributed(Conv2D(128, (11, 11), strides=4, padding="same"), batch_input_shape=(None, 100, 224, 224, 1)))
    # seq.add(TimeDistributed(Conv2D(128, (11, 11), strides=4, padding='same'),
    #                         input_shape=(100, 224, 224, 1)))
    # seq.add(LayerNormalization())
    # seq.add(TimeDistributed(Conv2D(64, (5, 5), strides=2, padding="same")))
    # seq.add(LayerNormalization())
    # # # # # #
    # seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
    # seq.add(LayerNormalization())
    # seq.add(ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True))
    # seq.add(LayerNormalization())
    # seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
    # seq.add(LayerNormalization())
    # # # # #
    # seq.add(TimeDistributed(Conv2DTranspose(64, (5, 5), strides=2, padding="same")))
    # seq.add(LayerNormalization())
    # seq.add(TimeDistributed(Conv2DTranspose(128, (11, 11), strides=4, padding="same")))
    # seq.add(LayerNormalization())
    # seq.add(TimeDistributed(Conv2D(1, (11, 11), activation="sigmoid", padding="same")))
    # seq.add(LayerNormalization())
    # seq.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
    # seq.add(TimeDistributed(Conv2D(1, (5, 5), activation="sigmoid", padding="same")))
    # seq.add(LayerNormalization())
    # seq.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
    # seq.add(TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu')))
    #
    # seq.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
    # seq.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
    #
    # seq.add(TimeDistributed(AveragePooling2D(pool_size=(25, 25))))
    #
    # seq.add(TimeDistributed(Flatten()))
    # seq.add(Reshape((-1,)))
    #
    # # seq.add(Dropout(0.4))
    # # # seq.add(LSTM(512, return_sequences=False, dropout=0.4))
    # seq.add(Dense(512, activation='sigmoid'))
    # seq.add(LayerNormalization())
    # seq.add(Dropout(0.4))
    # seq.add(Dense(128, activation='sigmoid'))
    # seq.add(LayerNormalization())
    # # seq.add(Reshape((-1,)))
    # seq.add(Dense(2, activation='sigmoid'))
    return model

# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():

model = get_model()
# parallel_model = multi_gpu_model(model, gpus=4)

# parallel_model.load_weights('my_h5_model.h5')
model.compile(optimizer=Adam(lr=0.00001), loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# parallel_model.compile(optimizer=Adam(lr=0.0001), loss=[focal_loss(alpha=.25, gamma=2)], metrics=['accuracy'])
model.fit_generator(data.data_generator(X_train, 'standard', size=size, batch_size=batch_size),
                                 train_steps, epochs=epochs, callbacks=callbacks_list, verbose=1,
                                 validation_data=data.data_generator(X_test, 'standard', size=size,
                                                                     batch_size=batch_size),
                                 validation_steps=valid_steps , workers=38, initial_epoch=0 ,use_multiprocessing= True ,shuffle=True)




# model = model.load_weights(weights_path )
# parallel_model.compile(optimizer=Adam(lr=0.00001), loss='binary_crossentropy', metrics = ['accuracy'])
# #parallel_model.compile(optimizer=Adam(lr=0.0001), loss=[focal_loss(alpha=.25, gamma=2)],metrics = ['accuracy'])
#
# parallel_model.fit_generator(data.data_generator(X_train, 'standard', size=size, batch_size=batch_size),
#                     train_steps, epochs=epochs, callbacks=callbacks_list, verbose=1,
#                     validation_data=data.data_generator(X_test, 'standard', size=size, batch_size=batch_size),
#                     validation_steps=valid_steps , workers=39 ,initial_epoch=2)#,use_multiprocessing= True)

# model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
#
# model.fit_generator(data.data_generator(X_train, 'standard', size=size, batch_size=batch_size),
#                     train_steps, epochs=epochs, verbose=1,
#                     validation_data=data.data_generator(X_test, 'standard', size=size, batch_size=batch_size),
#                     validation_steps=valid_steps)
