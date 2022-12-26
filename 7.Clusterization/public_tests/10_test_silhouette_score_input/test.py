import numpy as np
from solution import bcubed_score
import numpy as np
from itertools import product
from sklearn.metrics import silhouette_score as sklearn_silhouette_score
from solution import silhouette_score
import numpy as np
from solution import KMeansClassifier

def test(*args, **kwargs):

    def _check_silhouette_score_corner_test_03():
        data, labels, answer = np.array([[0]]), np.array([-1]), 0
        return np.allclose(silhouette_score(data, labels), answer, atol=1e-10, rtol=0.0)
    
    return _check_silhouette_score_corner_test_03(*args, **kwargs)
