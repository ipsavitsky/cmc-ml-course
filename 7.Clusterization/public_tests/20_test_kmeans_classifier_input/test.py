import numpy as np
from solution import bcubed_score
import numpy as np
from itertools import product
from sklearn.metrics import silhouette_score as sklearn_silhouette_score
from solution import silhouette_score
import numpy as np
from solution import KMeansClassifier

def test(*args, **kwargs):

    def _check_kmeans_classifier_corner_test_05():
        n_clusters, cluster_labels, true_labels, mapping, predicted_labels = (
            5,
            np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 3]),
            np.array([5, 1, 3, 2, 1, 4, 4, 3, 3, 2]),
            np.array([1, 1, 3, 2, 3]),
            np.array([1, 1, 1, 1, 1, 3, 3, 3, 3, 2])
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
            np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 3]),
            np.array([5, 1, 3, 2, 1, 4, 4, 3, 3, 2]),
            np.array([1, 1, 3, 2, 3, 3, 3, 3, 3, 3]),
            np.array([1, 1, 1, 1, 1, 3, 3, 3, 3, 2])
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
    
    return _check_kmeans_classifier_corner_test_05(*args, **kwargs)
