from .knn_train import knn_train as knn_train
from .knn_slic import PixelClassifier2 as PixelClassifier
from .knn_slic_metric import SuperpixelClassifier2 as SuperpixelClassifier
from .knn_slic_metric_clustering_plus import ClusteringClassifier2 as ClusteringClassifier

__all__ = [
    'knn_train',
    'PixelClassifier',
    'SuperpixelClassifier',
    'ClusteringClassifier',
]
