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
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import SGD

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True

# Model & Data parameters
num_classes = 10
input_shape = (32, 32, 3)
OVERFIT_BATCH_SIZE = 128  # overfitting batch size
OVERFIT_EPOCH = 28  # overfitting epoch number

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

    history = model.fit(x_train, y_train,
                        batch_size=OVERFIT_BATCH_SIZE, epochs=OVERFIT_EPOCH,
                        validation_data=(x_val, y_val))
    model.save("overfitCNN.model")

    plot_(history)


# Investigate the role of mini-batch size on overfitting
def validate_minibatch_size():
    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    batch_sizes = [32, len(x_train)]  # Includes full batch gradient descent
    # batch_sizes = [5, 16, 32, 64, 128, 256, 1024, len(x_train)]  # Includes full batch gradient descent
    results = {}

    for batch in batch_sizes:
        print(f"Validating for batch size {batch}")
        tmp_fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(x_train)):
            print(f"  Fold {fold + 1}/{n_splits}")
            x_train_fold, y_train_fold = x_train[train_idx], y_train[train_idx]
            x_val_fold, y_val_fold = x_train[val_idx], y_train[val_idx]

            tmp_model = keras.models.load_model("overfitCNN.model")
            tmp_model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.01), metrics=["accuracy"])  # use SGD
            history = tmp_model.fit(x_train_fold, y_train_fold,
                                    batch_size=batch, epochs=50,
                                    validation_data=(x_val_fold, y_val_fold), verbose=0)
            tmp_fold_results.append(history.history)

        # print(f'batch: {batch}, history: {tmp_fold_results}')
        results[batch] = tmp_fold_results
        print(f"Done for batch size {batch}")

    results = calculate_average_metrics(results)
    return results


def calculate_average_metrics(results):
    average_results = {}

    for batch_size, histories in results.items():
        sum_loss = np.zeros_like(histories[0])
        sum_accuracy = np.zeros_like(histories[0])
        sum_val_loss = np.zeros_like(histories[0])
        sum_val_accuracy = np.zeros_like(histories[0])

        n_folds = len(histories)

        # Sum up all metrics across all folds
        for history in histories:
            sum_loss = np.add(sum_loss, history['loss'])
            sum_accuracy = np.add(sum_accuracy, history['accuracy'])
            sum_val_loss = np.add(sum_val_loss, history['val_loss'])
            sum_val_accuracy = np.add(sum_val_accuracy, history['val_accuracy'])

        average_results[batch_size] = {
            'loss': sum_loss / n_folds,
            'accuracy': sum_accuracy / n_folds,
            'val_loss': sum_val_loss / n_folds,
            'val_accuracy': sum_val_accuracy / n_folds
        }
        # print(f'batch: {batch_size}, aver ret: {average_results}')
    return average_results


# compute gradient noise by SGD with different batch size
def measure_gradient_noise(model, x_train, y_train, batch_sizes, n_runs=10):
    gradient_variances = {}

    for batch_size in batch_sizes:
        # Initialize a list to store gradients for each variable
        gradients_list = [[] for _ in range(len(model.trainable_variables))]
        for run in range(n_runs):
            indices = np.random.permutation(len(x_train))
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]
            batch_gradients = get_gradients(model, x_train_shuffled[:batch_size], y_train_shuffled[:batch_size])

            # Append gradients for each variable
            for var_index, grad in enumerate(batch_gradients):
                gradients_list[var_index].append(grad)

        # Calculate variance for each variable's gradients
        variances = [np.var(np.stack(grads), axis=0) for grads in gradients_list]
        gradient_variances[batch_size] = variances

    return gradient_variances


def get_gradients(model, x_batch, y_batch):
    with tf.GradientTape() as tape:
        predictions = model(x_batch, training=True)
        loss = keras.losses.categorical_crossentropy(y_batch, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    return [g.numpy() for g in gradients if g is not None]


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


def plot_gradient_variances(gradient_variances):
    for batch_size, variances_list in gradient_variances.items():
        plt.figure(figsize=(10, 6))
        # We'll calculate the mean variance for each layer and plot that
        mean_variances = [np.mean(v) for v in variances_list]

        plt.bar(range(len(mean_variances)), mean_variances)
        plt.title(f'Average Gradient Variances for Batch Size {batch_size}')
        plt.xlabel('Layer Index')
        plt.ylabel('Mean Variance')
        plt.show()


def plot_all_batch_sizes_gradient_variances(gradient_variances_dict):
    plt.figure(figsize=(12, 8))
    for batch_size, variances_list in gradient_variances_dict.items():
        # Calculate the mean variance for each layer
        mean_variances = [np.mean(np.array(v).flatten()) for v in variances_list]
        plt.plot(mean_variances, label=f'Batch Size {batch_size}')

    plt.title('Average Gradient Variances Across Different Batch Sizes')
    plt.xlabel('Layer Index')
    plt.ylabel('Mean Variance')
    plt.legend()
    plt.show()


def main():
    # pretrain_overfit_cnn_model()

    results = validate_minibatch_size()
    plot_results(results)

    # calc noise
    # model = keras.models.load_model("overfitCNN.model")
    # batch_sizes = [16, 32, 64, 128, 256, 1024, len(x_train)]
    # gradient_variances = measure_gradient_noise(model, x_train, y_train, batch_sizes, n_runs=20)
    # plot_all_batch_sizes_gradient_variances(gradient_variances)

    # model = keras.models.load_model("overfitCNN.model")
    # print_confusion_matrix(model)


if __name__ == '__main__':
    main()
