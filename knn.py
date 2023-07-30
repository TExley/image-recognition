import numpy as np
import matplotlib.pyplot as plt

from time import time
from joblib import Parallel, delayed, cpu_count
from data import read_data


# Calculates the error of our estimated y value
#   Params:
#       y           - n x 1 numpy array of output points
#       predicted_y - n x 1 numpy array of predicted output points
#   Returns:
#       error       - normalized number of incorrect guesses
def get_error(y, predicted_y):
    error = np.sum(y != predicted_y) / y.shape[0]
    return error


# Finds and returns the squared L2 norms between data points and the query point
#   Params:
#       x       - n x d numpy array of data points
#       query   - 1 x d numpy array of a query point
#   Returns:
#       norms   - n x 0 list of squared L2 norms between data points and the query point
def get_l2_norms(x, query):
    norms = np.sum(np.square(x - query), axis=1)
    return norms


# Runs a kNN classifier on the query point
#   Params:
#       x           - n x d numpy array of data points
#       y           - n x 1 numpy array of output points
#       query       - 1 x d numpy array of a query point
#       k           - p x 0 numpy array of knn values to use
#   Returns:
#       common_y    - list of p most common values of the nearest k neighbors to the query point
def knn_classify_point(x, y, query, k):
    norms = get_l2_norms(x, query)
    indices = np.argsort(norms)
    common_y = []
    for i in range(k.shape[0]):  # Not the most efficient loop
        values = y[indices[0:k[i]], 0]
        common_y.append(np.bincount(values).argmax())
    return common_y


# Trains a KNN
#   Params:
#       x               - n x d numpy array of data points
#       y               - n x 1 numpy array of output points
#       queries         - h x d numpy array of query points
#       k               - p x 0 numpy array of knn values to use
#   Returns:
#       predicted_ys    - h x p numpy array of predicted y values for the query points for each k
def train(x, y, queries, k):
    predicted_ys = Parallel(n_jobs=cpu_count())(
        delayed(knn_classify_point)(x, y, queries[[i], :], k) for i in range(queries.shape[0]))
    return np.array(predicted_ys)


# Trains a KNN and returns its error values for each k
#   Params:
#       x           - n x d numpy array of data points
#       y           - n x 1 numpy array of output points
#       queries     - h x d numpy array of query points
#       queries_y   - h x 1 numpy array of query output points
#       k           - p x 0 numpy array of knn values to use
#   Returns:
#       errors      - p long list of query errors for each value of k
def train_for_error(x, y, queries, queries_y, k):
    errors = []
    predicted_ys = train(x, y, queries, k)
    for i in range(k.shape[0]):
        errors.append(get_error(queries_y, predicted_ys[:, [i]]))
    return errors


if __name__ == '__main__':
    x_train, y_train, x_validation, y_validation, x_test, y_test = read_data(5/6)

    K = np.array([1, 5, 10, 20, 25, 50, 75, 100, 125, 150, 175, 200, 500])

    t_0 = time()

    train_errors = train_for_error(x_train, y_train, x_train, y_train, K)
    validation_errors = train_for_error(x_train, y_train, x_validation, y_validation, K)
    test_errors = train_for_error(x_train, y_train, x_test, y_test, K)

    t_1 = time()

    print(f"Done: {t_1 - t_0: .2f}s")

    plt.figure(figsize=(16, 9))
    plt.plot(K, train_errors, label="Training Error")
    plt.plot(K, validation_errors, label="Validation Error")
    plt.plot(K, test_errors, label="Testing Error")
    plt.title("KNN Error")
    plt.xlabel("K")
    plt.ylabel("Error")
    plt.legend()
    plt.show()
