import numpy as np
import scipy.io as sio


# Reads nd-arrays of image data from specific .mat files
#   Params:
#       train_ratio - percent of the data to use for training as a float
#   Returns:
#       x_train     - training x values as an (n * train_ratio) x d numpy array
#       y_train     - training y values as an (n * train_ratio) x 1 numpy array
#       x_validation      - testing x values as an (n * (1 - train_ratio)) x d numpy array
#       y_validation      - testing y values as an (n * (1 - train_ratio)) x 1 numpy array
#       x_test          - testing x values as an (n * (1 - train_ratio)) x d numpy array
#       y_test          - testing y values as an (n * (1 - train_ratio)) x 1 numpy array
def read_data(train_ratio):
    x = np.transpose(sio.loadmat('MNIST_train_image.mat')['trainX'])
    y = sio.loadmat('MNIST_train_label.mat')['trainL']
    x_test = np.transpose(sio.loadmat('MNIST_test_image.mat')['testX'])
    y_test = sio.loadmat('MNIST_test_label.mat')['testL']
    indices = np.random.permutation(x.shape[0])
    cutoff = int(train_ratio * x.shape[0])
    indices_train = indices[0:cutoff]
    indices_validation = indices[cutoff:x.shape[0]]

    return x[indices_train, :], y[indices_train, :], x[indices_validation, :], y[indices_validation, :], x_test, y_test


# Reads nd-arrays of image data from specific .mat files and adds a leading dimension of ones
#   Params:
#       train_ratio - percent of the data to use for training as a float
#   Returns:
#       x_train     - training x values as an (n * train_ratio) x (d + 1) numpy array
#       y_train     - training y values as an (n * train_ratio) x 1 numpy array
#       x_validation      - testing x values as an (n * (1 - train_ratio)) x (d + 1) numpy array
#       y_validation      - testing y values as an (n * (1 - train_ratio)) x 1 numpy array
#       x_test          - testing x values as an (n * (1 - train_ratio)) x (d + 1) numpy array
#       y_test          - testing y values as an (n * (1 - train_ratio)) x 1 numpy array
def read_data_augment(train_ratio):
    x = np.transpose(sio.loadmat('MNIST_train_image.mat')['trainX'])
    y = sio.loadmat('MNIST_train_label.mat')['trainL']
    x_test = augment_bias(np.transpose(sio.loadmat('MNIST_test_image.mat')['testX']))
    y_test = sio.loadmat('MNIST_test_label.mat')['testL']
    indices = np.random.permutation(x.shape[0])
    cutoff = int(train_ratio * x.shape[0])
    indices_train = indices[0:cutoff]
    indices_validation = indices[cutoff:x.shape[0]]
    x_train = augment_bias(x[indices_train, :])
    x_validation = augment_bias(x[indices_validation, :])
    return x_train, y[indices_train, :], x_validation, y[indices_validation, :], x_test, y_test


# Adds a column of ones to the dataset
#   Params:
#       x   - n x d numpy array of data points
#   Returns:
#       x   - n x (d + 1) numpy array of data points
def augment_bias(x):
    return np.concatenate((np.ones((x.shape[0], 1)), x), 1)


if __name__ == '__main__':
    x_train, y_train, x_validation, y_validation, x_test, y_test = read_data(5/6)
    print(x_train.shape)
    print(y_train.shape)
    print(x_validation.shape)
    print(y_validation.shape)
    print(x_test.shape)
    print(y_test.shape)