import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import OPTICS
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch
from sklearn.exceptions import ConvergenceWarning
import warnings
import skfuzzy as fuzz
from minisom import MiniSom
from tkinter import filedialog

from util.timing import timing
from core.knn_slic_metric_clustering import ClusteringClassifier

class ClusteringClassifier2(ClusteringClassifier):
    """"
    Classificador de superpixels, variando qual o agrupador.
    """

    @timing
    def Find_optimal_k(self, sp_list, algorithm_name="KMeans", k_min=None, k_max=None, show_inertia=False):
        """
        Encontra o valor ótimo de k para K-means, minimizando as diferenças entre superpixels via método do cotovelo.
        """

        print(f"Encontrando o valor ótimo de k do algoritmo {algorithm_name}...")
        if not k_min: k_min = int(len(sp_list)*0.05)
        if not k_max: k_max = int(len(sp_list)*0.95)

        if len(sp_list) < k_min or len(sp_list) < k_max:
            raise ValueError(f"Número de superpixels ({len(sp_list)}) deve ser maior ou igual a k_min ({k_min}) e k_max ({k_max}).")
        
        distortions = []
        k_values = list(range(k_min, k_max + 1))
        if algorithm_name == "KMeans":
            for k in k_values:
                model = KMeans(n_clusters=k, random_state=0)
                model.fit(sp_list)
                distortions.append(model.inertia_)
        elif algorithm_name == "AgglomerativeClustering":
            from sklearn.metrics import pairwise_distances
            for k in k_values:
                model = AgglomerativeClustering(n_clusters=k)
                model.fit(sp_list)
                # Pseudo-distortion: soma das distâncias aos centróides
                centroids = []
                for i in range(k):
                    cluster_points = sp_list[model.labels_ == i]
                    if len(cluster_points) > 0:
                        centroids.append(cluster_points.mean(axis=0))
                    else:
                        centroids.append([0] * sp_list.shape[1])
                distortion = sum(((sp_list[i] - centroids[label])**2).sum() for i, label in enumerate(model.labels_))
                distortions.append(distortion)
        elif algorithm_name == "GaussianMixture":
            for k in k_values:
                model = GaussianMixture(n_components=k, random_state=0)
                model.fit(sp_list)
                bic = model.bic(sp_list)
                distortions.append(bic)
        elif algorithm_name == "Birch":
            for k in k_values:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    model = Birch(n_clusters=k, threshold=0.3)
                    model.fit(sp_list)
                
                # Usar os rótulos do Birch para calcular a distorção, se possível
                if hasattr(model, 'labels_'):
                    labels = model.labels_
                    centroids = []
                    for i in range(k):
                        cluster_points = sp_list[labels == i]
                        if len(cluster_points) > 0:
                            centroids.append(cluster_points.mean(axis=0))
                        else:
                            centroids.append([0] * sp_list.shape[1])
                    distortion = sum(((sp_list[i] - centroids[labels[i]])**2).sum() for i in range(len(sp_list)))
                    distortions.append(distortion)
                else:
                    distortions.append(float('inf'))  # fallback se não gerar labels

        elif algorithm_name == "FuzzyCMeans":
            # sp_list deve estar em shape (n_features, n_samples)
            data = sp_list.T
            for k in k_values:
                cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                    data=data, c=k, m=2, error=0.005, maxiter=1000, init=None)
                # Usar o índice FPC ou jm (função objetivo)
                distortions.append(jm[-1])
        elif algorithm_name == "SOM":
            for k in k_values:
                som_dim = int(k ** 0.5) + 1
                som = MiniSom(som_dim, som_dim, sp_list.shape[1], sigma=1.0, learning_rate=0.5, random_seed=0)
                som.train_random(sp_list, 100)
                # Criar clusters a partir dos neurônios vencedores
                labels = []
                neurons = []
                for x in sp_list:
                    w = som.winner(x)
                    neurons.append(w)
                unique_neurons = list(set(neurons))
                cluster_map = {neuron: idx for idx, neuron in enumerate(unique_neurons)}
                labels = [cluster_map[n] for n in neurons]
                centroids = []
                for i in range(len(unique_neurons)):
                    cluster_points = sp_list[[l == i for l in labels]]
                    centroids.append(cluster_points.mean(axis=0))
                distortion = sum(((sp_list[i] - centroids[labels[i]])**2).sum() for i in range(len(sp_list)))
                distortions.append(distortion)
                
        else:
            raise ValueError(f"Método de classificação desconhecido: {algorithm_name}")
        
        # Suavizar a curva de distorções
        distortions_smooth = gaussian_filter1d(distortions, sigma=5)
        # Encontrando o "cotovelo"
        inertia_curve = np.gradient(np.gradient(distortions_smooth)) # Segunda derivada da inércia
        optimal_k_index = np.argmax(inertia_curve)
        optimal_k = k_values[optimal_k_index]
        if show_inertia: self.Show_inertia(k_values, distortions_smooth, optimal_k)
        
        return optimal_k
    
    @timing
    def Type_classification(self, image, segments, method='KMeans', eps=0.5, show_inertia=False):
        """
        Aplica o método de classificação especificado (DBSCAN ou KMeans) para classificar os superpixels.
        """
        
        print(f"Classificando em tipos similares com o método {method}...")
        sp_list = self.Get_SP_list(image, segments)
        
        if method == 'DBSCAN':
            optimal_eps = self.find_optimal_eps(sp_list, show_plot=show_inertia)
            dbscan = DBSCAN(eps=optimal_eps, min_samples=1, n_jobs=-1).fit(sp_list)
            labels = dbscan.labels_
        elif method == 'KMeans':
            optimal_k = self.Find_optimal_k(sp_list, algorithm_name=method, show_inertia=show_inertia)
            kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(sp_list)
            labels = kmeans.labels_
        elif method == 'AgglomerativeClustering':
            optimal_k = self.Find_optimal_k(sp_list, algorithm_name=method, show_inertia=show_inertia)
            agglomerative = AgglomerativeClustering(n_clusters=optimal_k)
            labels = agglomerative.fit_predict(sp_list)
        elif method == 'OPTICS':
            optics = OPTICS(min_samples=2, xi=0.05, min_cluster_size=0.05)
            labels = optics.fit_predict(sp_list)
        elif method == 'AffinityPropagation':
            affinity_propagation = AffinityPropagation(random_state=0)
            labels = affinity_propagation.fit_predict(sp_list)
        elif method == 'MeanShift':
            bandwidth = estimate_bandwidth(sp_list, quantile=0.2, n_samples=500)
            mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(sp_list)
            labels = mean_shift.labels_
        elif method == 'GaussianMixture':
            optimal_k = self.Find_optimal_k(sp_list, algorithm_name=method, show_inertia=show_inertia)
            gmm = GaussianMixture(n_components=optimal_k, random_state=0)
            labels = gmm.fit_predict(sp_list)
        elif method == 'Birch':
            optimal_k = self.Find_optimal_k(sp_list, algorithm_name=method, show_inertia=show_inertia)
            birch = Birch(n_clusters=optimal_k)
            labels = birch.fit_predict(sp_list)
        elif method == 'FuzzyCMeans':
            data = np.array(sp_list).T
            optimal_k = self.Find_optimal_k(sp_list, algorithm_name=method, show_inertia=show_inertia)
            # Executa o Fuzzy C-Means
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                data,             # Dados de entrada (features x samples)
                c=optimal_k,      # Número de clusters
                m=2.0,            # Exponencial de fuzzificação (2.0 é comum)
                error=0.005,      # Critério de parada
                maxiter=1000,     # Número máximo de iterações
                init=None,        # Inicialização padrão
                seed=0
            )
            # O C-means retorna probabilidade, preciso converter para rótulos
            labels = np.argmax(u, axis=0)  # axis=0 porque u é shape (clusters, samples)
        elif method == 'SOM':
            som_x, som_y = 5, 5
            som = MiniSom(x=som_x, y=som_y, input_len=sp_list.shape[1], sigma=1.0, learning_rate=0.5)
            som.random_weights_init(sp_list)
            som.train_random(sp_list, 100)
            labels = [np.ravel_multi_index(som.winner(x), (som_x, som_y)) for x in sp_list]
        else:
            raise ValueError(f"Método de classificação desconhecido: {method}")

        combined_segments = self.Clusters2segments(segments, labels)
        combined_segments = self.First2Zero(combined_segments)
        return combined_segments, labels
    
def main():
    classifier = ClusteringClassifier2(num_segments=200)
    # classifier.SP_divide()
    # classifier.Train()
    # classifier.classify(threshold=5.8, show_data=False)
    path = filedialog.askopenfilename(title="Selecione a imagem para aplicação",
                                                            filetypes=[("Imagens", "*.jpeg;*.jpg;*.png")])
    alg_list = [
    "DBSCAN", 
     "KMeans", 
     "AgglomerativeClustering", 
     "OPTICS", 
     "AffinityPropagation", 
     "MeanShift", 
     "GaussianMixture", 
     "Birch", 
     "FuzzyCMeans", 
     "SOM"
     ]
    for alg in alg_list: 
        classifier.Type_visualization(image_path=path, method=alg, show_inertia=False)

if __name__ == "__main__":
    
    main()
    # classifier.Type_visualization(image_path=path, method='DBSCAN', show_inertia=False)


    """
    nomes dos métodos implementados:
    DBSCAN
    KMeans
    AgglomerativeClustering
    OPTICS
    AffinityPropagation
    MeanShift
    GaussianMixture
    Birch
    FuzzyCMeans
    SOM
    """

