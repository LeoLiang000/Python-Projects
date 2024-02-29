import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import time


plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
import sys

# Model / data parameters
num_classes = 10
input_shape = (32, 32, 3)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
n = 5000 * 10
x_train = x_train[1:n]
y_train = y_train[1:n]
# x_test=x_test[1:500]; y_test=y_test[1:500]

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
print("orig x_train shape:", x_train.shape)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# for w in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0, 1, 5, 10]:
optional = True
if not optional:
    for w in [1e-4]:
        model = keras.Sequential()
        model.add(Conv2D(16, (3, 3), padding='same', input_shape=x_train.shape[1:], activation='relu'))

        # ii(c).i replace stride with max pooling
        # model.add(Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu'))
        model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))  # add max pooling

        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))

        # ii(c).i replace stride with max pooling
        # model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))  # add max pooling 2x2
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1(w)))

else:
    model = keras.Sequential()
    model.add(Conv2D(8, (3, 3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
    model.add(Conv2D(8, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1(1e-4)))

    # try weights [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0, 1] default 1e-4

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
model.summary()


batch_size = 128
epochs = 100

# record start time
t1 = time.time()
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# record end time
t2 = time.time()

delta_t = format(t2 - t1, '.4f')
print(f'Time Taken: {delta_t}')

model.save("cifar.model")
plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title(f'Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title(f'Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.tight_layout()
plt.show()

preds = model.predict(x_train)
y_pred = np.argmax(preds, axis=1)
y_train1 = np.argmax(y_train, axis=1)
print(f'========================Training Data========================')
print(classification_report(y_train1, y_pred))
print(confusion_matrix(y_train1, y_pred))

preds = model.predict(x_test)
y_pred = np.argmax(preds, axis=1)
y_test1 = np.argmax(y_test, axis=1)
print(f'========================Test Data========================')
print(classification_report(y_test1, y_pred))
print(confusion_matrix(y_test1, y_pred))

# Determine the most common label in the training data
most_common_label = np.argmax(np.bincount(np.argmax(y_train, axis=1)))

# Evaluate the baseline classifier for training data
baseline_predictions_train = np.full((len(x_train),), most_common_label)
print("========================Baseline Classifier Training Data========================")
print(classification_report(y_train1, baseline_predictions_train))
print(confusion_matrix(y_train1, baseline_predictions_train))

# Evaluate the baseline classifier for test data
baseline_predictions_test = np.full((len(x_test),), most_common_label)
print("========================Baseline Classifier Test Data========================")
print(classification_report(y_test1, baseline_predictions_test))
print(confusion_matrix(y_test1, baseline_predictions_test))





