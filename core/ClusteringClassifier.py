import numpy as np
import cv2
import os
import skfuzzy as fuzz
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import OPTICS
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch

from core.SuperpixelClassifier import SuperpixelClassifier
from util.timing import timing
from util import *

os.environ["LOKY_MAX_CPU_COUNT"] = "11"

class ClusteringClassifier:
    """
    Classificador de superpixel com classificação de tipos parecidos.
    """

    def __init__(self, num_segments=0, new_model=False, LAB=True):
        self.sp_classifier = SuperpixelClassifier(
            num_segments=num_segments, new_model=new_model, LAB=LAB)
        self.num_segments = num_segments

    def Get_SP_list(self, image, segments):
        sp_list = []
        segments += 1
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        for segment_id in np.unique(segments):
            mask = segments == segment_id
            region = lab_image[mask]
            l, a, b = np.mean(region, axis=0)
            a_std = np.std(region[:, 1], axis=0)
            b_std = np.std(region[:, 2], axis=0)
            sp_list.append([a, b, a_std, b_std])
        sp_list = np.array(sp_list)
        scaler = StandardScaler()
        sp_list = scaler.fit_transform(sp_list)
        return sp_list

    @timing
    def Type_classification(self, image, segments, method='KMeans', show_inertia=False):
        print(f"Classificando em tipos similares com o método {method}...")
        sp_list = self.Get_SP_list(image, segments)
        labels = self._dispatch_clustering(method, sp_list, show_inertia)

        combined_segments = Clusters2segments(segments, labels)
        combined_segments = First2Zero(combined_segments)
        return combined_segments, labels

    @timing
    def Type_visualization(self, image_path=None, mode='KMeans', show_inertia=False):
        image, apply_image_dir, apply_image_name_no_ext = Load_Image(image_path)
        segments_path = os.path.join(apply_image_dir, "segmentos",
                                     f"seg_finais_{apply_image_name_no_ext}_{self.num_segments}.npy")
        if not os.path.exists(segments_path):
            self.sp_classifier.classify(threshold=5.8,
                                        image_path=os.path.join(apply_image_dir,
                                        f"{apply_image_name_no_ext}.jpeg"))
        segments = np.load(segments_path)
        segments_classified, labels = self.Type_classification(image, segments, method=mode, show_inertia=show_inertia)
        print(f"Número de clusters obtidos: {len(np.unique(labels))}") 
        print(f"De um total de: {len(np.unique(segments))} segmentos iniciais")
        output_image = Paint_image(image, segments_classified)
        Save_image(output_image, apply_image_dir, apply_image_name_no_ext, self.num_segments, mode)

        segments_classified_path = os.path.join(apply_image_dir, "segmentos", 
                                                f"seg_{mode}_{apply_image_name_no_ext}_{self.num_segments}.npy")
        np.save(segments_classified_path, segments_classified)

    def _dispatch_clustering(self, method, sp_list, show_inertia):
        clustering_methods = {
            'DBSCAN': lambda: DBSCAN(
                eps=Find_optimal_eps(sp_list, show_plot=show_inertia),
                min_samples=1,
                n_jobs=-1
            ).fit(sp_list).labels_,

            'KMeans': lambda: KMeans(
                n_clusters=Find_optimal_k(sp_list, 'KMeans', show_inertia),
                random_state=0
            ).fit(sp_list).labels_,

            'AgglomerativeClustering': lambda: AgglomerativeClustering(
                n_clusters=Find_optimal_k(sp_list, 'AgglomerativeClustering', show_inertia)
            ).fit_predict(sp_list),

            'OPTICS': lambda: OPTICS(
                min_samples=2, xi=0.05, min_cluster_size=0.05
            ).fit_predict(sp_list),

            'AffinityPropagation': lambda: AffinityPropagation(
                random_state=0
            ).fit_predict(sp_list),

            'MeanShift': lambda: MeanShift(
                bandwidth=estimate_bandwidth(sp_list, quantile=0.2, n_samples=500),
                bin_seeding=True
            ).fit(sp_list).labels_,

            'GaussianMixture': lambda: GaussianMixture(
                n_components=Find_optimal_k(sp_list, 'GaussianMixture', show_inertia),
                random_state=0
            ).fit_predict(sp_list),

            'Birch': lambda: Birch(
                n_clusters=Find_optimal_k(sp_list, 'Birch', show_inertia)
            ).fit_predict(sp_list),

            'FuzzyCMeans': lambda: np.argmax(
                fuzz.cluster.cmeans(
                    np.array(sp_list).T,
                    c=Find_optimal_k(sp_list, 'FuzzyCMeans', show_inertia),
                    m=2.0,
                    error=0.005,
                    maxiter=1000,
                    init=None,
                    seed=0
                )[1],
                axis=0
            ),

            'SOM': lambda: [
                np.ravel_multi_index(
                    MiniSom(5, 5, sp_list.shape[1], sigma=1.0, learning_rate=0.5).train_random(sp_list, 100) or
                    MiniSom(5, 5, sp_list.shape[1], sigma=1.0, learning_rate=0.5).winner(x), (5, 5)
                ) for x in sp_list
            ]
        }

        if method not in clustering_methods:
            raise ValueError(f"Método de classificação desconhecido: {method}")
        
        return clustering_methods[method]()

