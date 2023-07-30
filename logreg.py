import numpy as np
import matplotlib.pyplot as plt

from time import time
from joblib import Parallel, delayed, cpu_count
from data import read_data_augment


# Calculates the error of our estimated y value
#   Params:
#       y           - n x 1 numpy array of output points
#       predicted_y - n x 1 numpy array of predicted output points
#   Returns:
#       error       - normalized number of incorrect guesses
def get_error(y, predicted_y):
    error = np.sum(y != predicted_y) / y.shape[0]
    return error


# Calculates the multinomial logistic function for each data point for each class
#   Params:
#       u       - n x c numpy array of linear weighted sums between data and weights
#   Returns:
#       output  - n x c numpy array of the logistic outputs for each data point for each class
def logistic(u):
    u_fixed = u - np.max(u)
    exponential_weighted_sums = np.exp(u_fixed)  # n x c numpy array of e^weighted_sums
    sum_exponential_weighted_sums = np.sum(exponential_weighted_sums, axis=1, keepdims=True)  # n x 1 numpy array sums
    output = exponential_weighted_sums / sum_exponential_weighted_sums  # n x c numpy array
    return output


# Calculates the predicted class at each data point
#   Params:
#       u               - n x c numpy array of linear weighted sums between data and weights
#   Returns:
#       predicted_ys    - n x 1 numpy array of the logistic outputs for each data point
def get_class(u):
    predicted_classes = np.argpartition(-u, 0)[:, [0]]
    return predicted_classes

# Calculates the cross-entropy loss of the multinomial logistic function
#   Params:
#       u           - n x c numpy array of linear weighted sums between data and weights
#       test_y      - n x c numpy array  where each n rows has a one at the index that matches y and 0s elsewhere
#       w           - d x c numpy array of weights where c is the total number of classes
#       regularizer - a scalar to modify the loss function
#   Returns:
#       loss        - a scalar of the cross-entropy loss of the multinomial logistic function
def loss_function(u, test_y, w, regularizer):
    regularization = regularizer * np.sum(np.square(w))  # scalar
    likelihood = logistic(u)  # n x c numpy array representing the likelihood P(k|x) for each k in c
    valid_likelihoods = np.sum(likelihood * test_y, axis=1)  # n x . numpy array representing the likelihood P(y|x)
    loss = -np.sum(np.log(valid_likelihoods + np.finfo(float).eps)) / u.shape[0] + regularization
    return loss


# Calculates the gradient cross-entropy loss of the multinomial logistic function wrt w
#   Params:
#       u           - n x c numpy array of linear weighted sums between data and weights
#       x           - n x d numpy array of data points
#       test_y      - n x c numpy array  where each n rows has a one at the index that matches y and 0s elsewhere
#       w           - d x c numpy array of weights where c is the total number of classes
#       regularizer - a scalar to modify the loss function
#   Returns:
#       w_grad      - d x c numpy array of the gradient cross-entropy loss of the multinomial logistic function wrt w
def loss_gradient_w(u, x, test_y, w, regularizer):
    regularization = 2 * regularizer * w  # scalar
    likelihood = logistic(u)  # n x c numpy array representing the likelihood P(k|x) for each k in c
    w_grad = -np.matmul(np.transpose(x), test_y - likelihood)/u.shape[0] + regularization
    return w_grad


# Calculates the linear weighted sums between data and weights
#   Params:
#       x   - n x d numpy array of data points
#       w   - d x c numpy array of weights where c is the total number of classes
#   Returns:
#       u   - 1 x c numpy array of linear weighted sums between data and weights
def transform_u(x, w):
    return x.dot(w)


# Trains a multinomial logistic regression
#   Params:
#       x           - n x d numpy array of data points
#       y           - n x 1 numpy array of class values from 0 to num_classes - 1
#       num_classes - number of unique values that a single output of y can take on
#       regularizer - a scalar to modify the loss function
#       step_size   - a scalar to modify the rate of gradient descent
#   Returns:
#       w           - d x c numpy array of weights where c is the total number of classes
def train(x, y, regularizer=0.001, step_size=0.2, max_iterations=3000, num_classes=10, get_losses=False):
    w = np.random.normal(0, 1, (x.shape[1], num_classes))

    # n x c numpy array listing the indices of each element k in c
    indices = np.full((x.shape[0], num_classes), np.arange(0, num_classes))
    test_y = indices == y  # n x c numpy array  where each n rows has a one at the index that matches y and 0s elsewhere

    losses = [loss_function(transform_u(x, w), test_y, w, regularizer)] if get_losses else []

    for iterations in range(max_iterations):
        u = transform_u(x, w)  # n x c numpy array of linear weighted sums between data and weights
        w_grad = loss_gradient_w(u, x, test_y, w, regularizer)

        # Take the update step in gradient descent
        w = w - step_size * w_grad

        if get_losses and iterations % 100 == 99:
            losses.append(loss_function(u, test_y, w, regularizer))
    return w, losses


if __name__ == '__main__':
    x_train, y_train, x_validation, y_validation, x_test, y_test = read_data_augment(5/6)

    print_losses = False
    regularizers = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

    train_errors = []
    validation_errors = []
    test_errors = []

    t_0 = time()

    for regularizer in regularizers:
        w, losses = train(x_train, y_train, regularizer, get_losses=print_losses)

        train_errors.append(get_error(y_train, get_class(transform_u(x_train, w))))
        validation_errors.append(get_error(y_validation, get_class(transform_u(x_validation, w))))
        test_errors.append(get_error(y_test, get_class(transform_u(x_test, w))))

    t_1 = time()

    print(f"Done: {t_1 - t_0: .2f}s")

    if print_losses:
        plt.figure(figsize=(16, 9))
        plt.plot(np.arange(0, len(losses)) * 100, losses, label="Losses")
        plt.title("Multinomial Logistic Regression Losses")
        plt.xlabel("Iterations")
        plt.ylabel("Losses")
        plt.legend()
        plt.show()

    plt.figure(figsize=(16, 9))
    plt.plot(np.log10(regularizers), train_errors, label="Training Error")
    plt.plot(np.log10(regularizers), validation_errors, label="Validation Error")
    plt.plot(np.log10(regularizers), test_errors, label="Testing Error")
    plt.title("Multinomial Logistic Regression Error")
    plt.xlabel("Log10 Regularizers")
    plt.ylabel("Error")
    plt.legend()
    plt.show()
