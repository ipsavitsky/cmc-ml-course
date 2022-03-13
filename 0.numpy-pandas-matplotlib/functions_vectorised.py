from typing import Tuple
import numpy as np


def prod_non_zero_diag(X: np.ndarray) -> int:
    """
    Compute product of nonzero elements from matrix diagonal, 
    return -1 if there is no such elements.

    Return type: int / np.integer / np.int32 / np.int64
    """
    n = np.diag(X)
    res = np.prod(np.where(n != 0, n, 1))
    return res if np.sum(n) != 0 else -1


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Return True if both 1-d arrays create equal multisets, False if not.

    Return type: bool / np.bool_
    """
    return np.array_equiv(np.sort(x),np.sort(y))


def max_after_zero(x: np.ndarray) -> int:
    """
    Find max element after zero in 1-d array, 
    return -1 if there is no such elements.

    Return type: int / np.integer / np.int32 / np.int64
    """
    zeros = np.argwhere(x == 0)
    if (len(zeros) == 0 or zeros[0] == len(x) -1):
        return -1
    nums_after_zero = np.array([])
    if (len(x)-1) in zeros:
        nums_after_zero = x[zeros[:-1]+1]
    else:
        nums_after_zero = x[zeros+1]
    return np.max(nums_after_zero)


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Sum up image channels with weights.

    Return type: np.ndarray
    """
    return image @ weights


def run_length_encoding(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make run-length encoding.

    Return type: (np.ndarray, np.ndarray)
    """
    indeces = np.diff(x)
    indeces = np.where(indeces != 0)[0] + 1
    indeces = np.concatenate((np.array([0]), indeces))
    nums = x[indeces]
    indeces = np.concatenate((indeces, np.array([x.shape[0]])))
    counts = np.diff(indeces)

    return nums, counts


def pairwise_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Return pairwise object distance.

    Return type: np.ndarray
    """
    P = np.add.outer(np.sum(X**2, axis=1), np.sum(Y**2, axis=1))
    N = X @ Y.T
    return np.sqrt(P - 2*N)
pass
