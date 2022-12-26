from typing import List, Tuple


def prod_non_zero_diag(X: List[List[int]]) -> int:
    """
    Compute product of nonzero elements from matrix diagonal, 
    return -1 if there is no such elements.

    Return type: int
    """
    prod = 1
    at_least_one = False
    for i in range(min(len(X), len(X[0]))):
        if X[i][i] != 0:
            prod *= X[i][i]
            at_least_one = True
    return prod if at_least_one else -1


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    """
    Return True if both 1-d arrays create equal multisets, False if not.

    Return type: bool
    """
    return True if sorted(x) == sorted(y) else False


def max_after_zero(x: List[int]) -> int:
    """
    Find max element after zero in 1-d array, 
    return -1 if there is no such elements.

    Return type: int
    """
    max = -1
    flag = False
    for i in range(len(x)-1):
        if x[i] == 0:
            if not flag:
                max = x[i+1]
                flag = True
            if x[i+1] > max:
                max = x[i+1]
    return max


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    """
    Sum up image channels with weights.

    Return type: List[List[float]]
    """
    res = [[0]*len(image[0]) for _ in range(len(image))]
    for i in range(len(image)):
        for j in range(len(image[0])):
            for k in range(len(weights)):
                res[i][j] += image[i][j][k]*weights[k]
    return res


def run_length_encoding(x: List[int]) -> Tuple[List[int], List[int]]:
    """
    Make run-length encoding.

    Return type: (List[int], List[int])
    """
    cur_num = x[0]
    cur_sum = 1
    nums = []
    sums = []
    for val in enumerate(x[1:]):
        if val[1] != cur_num:
            nums.append(cur_num)
            sums.append(cur_sum)
            cur_num = val[1]
            cur_sum = 1
        else:
            cur_sum += 1
    nums.append(cur_num)
    sums.append(cur_sum)
    return (nums, sums)


def pairwise_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """
    Return pairwise object distance.

    Return type: List[List[float]]
    """
    res = [[0] * len(X) for _ in range(len(Y))]
    for i in range(len(X)):
        for j in range(len(Y)):
            for z in range(len(X[i])):
                res[i][j] += (X[i][z] - Y[j][z])**2
            res[i][j] = (res[i][j])**(0.5)
    return res
