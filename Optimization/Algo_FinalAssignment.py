import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True

# Model & Data parameters
num_classes = 10
input_shape = (32, 32, 3)

# split data into train and test sets
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()

n_train = 5000
n_test = 500
x_train = x_train_full[1:n_train]
y_train = y_train_full[1:n_train]
x_test = x_test[1:n_test]
y_test = y_test[1:n_test]

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# split validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

OVERFIT_BATCH_SIZE = 128  # overfitting batch size
OVERFIT_EPOCH = 50  # overfitting epoch number


def pretrain_overfit_cnn_model():
    model = keras.Sequential()
    model.add(Conv2D(16, (3, 3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
    model.add(Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    model.summary()

    history = model.fit(x_train, y_train, batch_size=OVERFIT_BATCH_SIZE, epochs=OVERFIT_EPOCH, validation_split=0.1)
    model.save("overfitCNN.model")

    plot_(history)


# Investigate the role of mini-batch size on overfitting
def validate_minibatch_size():
    batch_sizes = [128, 256, len(x_train)]  # Includes full batch gradient descent
    # batch_sizes = [5, 16, 32, 64, 128, 256, 1024, len(x_train)]  # Includes full batch gradient descent
    results = {}

    for batch in batch_sizes:
        tmp_model = keras.models.load_model("overfitCNN.model")
        history = tmp_model.fit(x_train, y_train,
                                batch_size=batch, epochs=OVERFIT_EPOCH,
                                validation_data=(x_val, y_val), verbose=0)
        results[batch] = history.history
        print(f"Done for batch size {batch}")

    return results


def plot_(history):
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def plot_results(results):
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan']
    linestyles = ['-', '--', '-.', ':']

    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(top=0.85, bottom=0.15, hspace=0.5, wspace=0.3)

    # Subplot for Accuracy
    plt.subplot(121)
    for i, batch_size in enumerate(results.keys()):
        plt.plot(results[batch_size]['accuracy'], color=colors[i % len(colors)], label=f'Batch {batch_size} - Train')
        plt.plot(results[batch_size]['val_accuracy'], color=colors[i % len(colors)], linestyle='--',
                 label=f'Batch {batch_size} - Val')  # Dashed line for validation

    plt.title('Model Accuracy by Batch Size')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # Subplot for Loss
    plt.subplot(122)
    for i, batch_size in enumerate(results.keys()):
        plt.plot(results[batch_size]['loss'], color=colors[i % len(colors)], label=f'Batch {batch_size} - Train')
        plt.plot(results[batch_size]['val_loss'], color=colors[i % len(colors)], linestyle='--',
                 label=f'Batch {batch_size} - Val')  # Dashed line for validation

    plt.title('Model Loss by Batch Size')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    plt.subplots_adjust(right=0.75)
    plt.show()


def print_confusion_matrix(model):
    preds = model.predict(x_train)
    y_pred = np.argmax(preds, axis=1)
    y_train1 = np.argmax(y_train, axis=1)
    print(classification_report(y_train1, y_pred))
    print(confusion_matrix(y_train1, y_pred))

    preds = model.predict(x_test)
    y_pred = np.argmax(preds, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    print(classification_report(y_test1, y_pred))
    print(confusion_matrix(y_test1, y_pred))


def main():
    # pretrain_overfit_cnn_model()
    results = validate_minibatch_size()
    plot_results(results)
    # model = keras.models.load_model("overfitCNN.model")
    # print_confusion_matrix(model)


if __name__ == '__main__':
    main()
