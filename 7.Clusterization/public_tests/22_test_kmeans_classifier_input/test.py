import numpy as np
from solution import bcubed_score
import numpy as np
from itertools import product
from sklearn.metrics import silhouette_score as sklearn_silhouette_score
from solution import silhouette_score
import numpy as np
from solution import KMeansClassifier

def test(*args, **kwargs):

    def _check_kmeans_classifier_corner_test_07():
        n_clusters, cluster_labels, true_labels, mapping, predicted_labels = (
            10,
            np.array([2, 2, 2, 1, 1, 1, 0, 0, 0]),
            np.array([5, 5, 6, 8, 7, 6, 6, 7, 7]),
            np.array([7, 6, 5, 6, 6, 6, 6, 6, 6, 6]),
            np.array([5, 5, 5, 6, 6, 6, 7, 7, 7])
        )
        mapping_checked, predicted_labels_checked = KMeansClassifier(n_clusters)._best_fit_classification(cluster_labels, true_labels)
    
        if not np.allclose(
            mapping_checked,
            mapping,
            atol=1e-10, rtol=0.0
        ) or not np.allclose(
            predicted_labels_checked,
            predicted_labels,
            atol=1e-10, rtol=0.0
        ):
            return False
    
        n_clusters, cluster_labels, true_labels, mapping, predicted_labels = (
            10,
            np.array([ 4,  4,  4, 1, 1, 1, 2, 2, 2, 3, 3, 3,  0, 0,  0, 0,  0]),
            np.array([-1, -1, -1, 5, 5, 6, 8, 7, 6, 6, 7, 7, -1, 8, -1, 3, -1]),
            np.array([3, 5, 6, 7, 6, 6, 6, 6, 6, 6]),
            np.array([6, 6, 6, 5, 5, 5, 6, 6, 6, 7, 7, 7, 3, 3, 3, 3, 3])
        )
        mapping_checked, predicted_labels_checked = KMeansClassifier(n_clusters)._best_fit_classification(cluster_labels, true_labels)
    
        if not np.allclose(
            mapping_checked,
            mapping,
            atol=1e-10, rtol=0.0
        ) or not np.allclose(
            predicted_labels_checked,
            predicted_labels,
            atol=1e-10, rtol=0.0
        ):
            return False
    
        return True
    
    return _check_kmeans_classifier_corner_test_07(*args, **kwargs)
