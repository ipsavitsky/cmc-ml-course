import numpy as np
from solution import bcubed_score
import numpy as np
from itertools import product
from sklearn.metrics import silhouette_score as sklearn_silhouette_score
from solution import silhouette_score
import numpy as np
from solution import KMeansClassifier

def test(*args, **kwargs):

    def _check_kmeans_classifier_corner_test_00():
        n_clusters, cluster_labels, true_labels, mapping, predicted_labels = 1, np.array([0]), np.array([0]), np.array([0]), np.array([0])
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
    
        n_clusters, cluster_labels, true_labels, mapping, predicted_labels = 1, np.array([0]), np.array([1]), np.array([1]), np.array([1])
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
    
    return _check_kmeans_classifier_corner_test_00(*args, **kwargs)
