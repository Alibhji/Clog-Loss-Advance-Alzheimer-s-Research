import numpy as np
import threading
from tensorflow import keras

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# ===============================================================
# Modify this:
# ===============================================================
USE_GENERATOR = True
NUM_WORKERS = 2
# ===============================================================
batch_size = 128
num_classes = 10
num_epochs = 5
# ===============================================================
class ThreadSafeIterator:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()

def thread_safe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe."""

    def g(*a, **kw):
        return ThreadSafeIterator(f(*a, **kw))

    return g

@thread_safe_generator
def create_generator(x, y):
    assert x.shape[0] == y.shape[0]
    i = 0
    while True:
        batch_slice = np.arange(i, i + batch_size) % x.shape[0]
        x_batch = x[batch_slice, ...]
        y_batch = y[batch_slice, ...]
        i = i + 1
        yield x_batch, y_batch
# ===============================================================
(x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()
# ===============================================================
x_train = x_train.reshape(60000, 28 * 28).astype('float32')/255
x_val = x_val.reshape(10000, 28 * 28).astype('float32')/255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
# ===============================================================
model = keras.models.Sequential()
model.add(keras.layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# ===============================================================
if USE_GENERATOR:
    train_gen = create_generator(x_train, y_train)
    steps_per_epoch = int(np.ceil(x_train.shape[0] / batch_size))
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=num_epochs,
                                  validation_data=(x_val, y_val),
                                  workers=NUM_WORKERS)
else:
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        validation_data=(x_val, y_val),
                        workers=NUM_WORKERS)
# ===============================================================
score = model.evaluate(x_val, y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
