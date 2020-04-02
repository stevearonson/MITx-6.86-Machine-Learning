import numpy as np
from sklearn.svm import LinearSVC


### Functions for you to fill in ###

def one_vs_rest_svm(train_x, train_y, test_x, C=0.1):
    """
    Trains a linear SVM for binary classifciation

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (0 or 1) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (0 or 1) for each test data point
    """
    # create a support vector machine (svm)
    # with a linear kernel
    clf = LinearSVC(random_state = 0, C=C)

    # fit model to training data
    clf.fit(train_x, train_y)
     
    # predict using the test data
    pred_test_y = clf.predict(test_x)
     
    return pred_test_y
    # raise NotImplementedError


def multi_class_svm(train_x, train_y, test_x, C=0.1):
    """
    Trains a linear SVM for multiclass classifciation using a one-vs-rest strategy

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (int) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (int) for each test data point
    """

    # create a support vector machine (svm)
    # with a linear kernel
    clf = LinearSVC(multi_class='ovr', random_state = 0, C=C)

    # fit model to training data
    clf.fit(train_x, train_y)
     
    # predict using the test data
    pred_test_y = clf.predict(test_x)
     
    return pred_test_y

    # raise NotImplementedError


def compute_test_error_svm(test_y, pred_test_y):
    return 1 - np.mean(pred_test_y == test_y)
    # raise NotImplementedError
