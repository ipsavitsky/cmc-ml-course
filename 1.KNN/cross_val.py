import numpy as np
from collections import defaultdict


def kfold_split(num_objects, num_folds):
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects (int): number of objects in train set
    num_folds (int): number of folds for cross-validation split

    Returns:
    list((tuple(np.array, np.array))): list of length num_folds, where i-th element of list contains tuple of 2 numpy arrays,
                                       the 1st numpy array contains all indexes without i-th fold while the 2nd one contains
                                       i-th fold
    """
    num_objects_one_fold = num_objects // num_folds
    res = list()
    for i in range(num_folds):
        if i == num_folds - 1:
            i_right = np.array(range(num_objects_one_fold * i, num_objects))
        else:
            i_right = np.array(
                [j + num_objects_one_fold *
                    i for j in range(num_objects_one_fold)]
            )
        res.append(
            (np.array([j for j in range(num_objects) if j not in i_right]), i_right))
    return res


def knn_cv_score(X, y, parameters, score_function, folds, knn_class):
    """Takes train data, counts cross-validation score over grid of parameters (all possible parameters combinations)

    Parameters:
    X (2d np.array): train set
    y (1d np.array): train labels
    parameters (dict): dict with keys from {n_neighbors, metrics, weights, normalizers}, values of type list,
                       parameters['normalizers'] contains tuples (normalizer, normalizer_name), see parameters
                       example in your jupyter notebook
    score_function (callable): function with input (y_predict, y_true) which outputs score metric
    folds (list): output of kfold_split
    knn_class (obj): class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight), value - mean score over all folds
    """
    res = dict()
    for n in parameters["n_neighbors"]:
        for metric in parameters["metrics"]:
            for weight in parameters["weights"]:
                for normalizer in parameters["normalizers"]:
                    if normalizer[0]:
                        normalizer[0].fit(X)
                        X_scaled = normalizer[0].transform(X)
                    else:
                        X_scaled = X

                    score = list()

                    for fold in folds:
                        knn = knn_class(
                            n_neighbors=n, metric=metric, weights=weight)
                        knn.fit(X_scaled[fold[0]], y[fold[0]])
                        y_new = knn.predict(X_scaled[fold[1]])
                        score.append(score_function(y[fold[1]], y_new))
                    key = (normalizer[1], n, metric, weight)
                    value = np.mean(np.array(score))
                    res[key] = value
    return res
