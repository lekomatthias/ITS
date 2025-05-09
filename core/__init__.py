from .knn_train import knn_train as knn_train
from .knn_slic_metric import SuperpixelClassifier
from .knn_slic_metric_clustering_plus import ClusteringClassifier2 as ClusteringClassifier

__all__ = [
    'knn_train',
    'SuperpixelClassifier',
    'ClusteringClassifier',
]
