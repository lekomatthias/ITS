import numpy as np
import matplotlib.pyplot as plt
import warnings
import skfuzzy as fuzz
from minisom import MiniSom
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch
from sklearn.exceptions import ConvergenceWarning

from util.timing import timing

@timing
def Show_inertia(k_values, distortions, optimal_k=None):
    plt.figure(figsize=(8, 5))
    prop = 100
    plt.plot(k_values, distortions / prop, '-', label=f'Inércia/{prop}')
    if optimal_k:
        plt.axvline(optimal_k, color='r', linestyle='--', label=f'valor ótimo = {optimal_k:.2f}')
    inertia_curve = np.gradient(np.gradient(distortions))
    plt.plot(k_values, inertia_curve, 'k-', label='Segunda derivada da inércia')
    plt.xlabel('N° de clusters(k) ou eps')
    plt.ylabel('Inércia')
    plt.title('Método do Cotovelo para Seleção de k ou eps')
    plt.legend()
    plt.show()

@timing
def Find_optimal_eps(sp_list, eps_min=0.1, eps_max=1.0, step=0.01, show_plot=False):
    print("Encontrando o valor ótimo de eps...")
    if len(sp_list) < 2:
        raise ValueError("A lista de superpixels deve conter pelo menos dois elementos.")
    
    distortions = []
    eps_values = np.arange(eps_min, eps_max + step, step)
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=1).fit(sp_list)
        labels = dbscan.labels_
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if num_clusters > 0:
            cluster_centers = np.array([sp_list[labels == i].mean(axis=0) for i in range(num_clusters)])
            inertia = sum(np.linalg.norm(sp_list[labels == i] - cluster_centers[i]) ** 2 for i in range(num_clusters))
            distortions.append(inertia)
        else:
            distortions.append(np.inf)
    
    distortions_smooth = gaussian_filter1d(distortions, sigma=3)
    second_derivative = np.gradient(np.gradient(distortions_smooth))
    optimal_index = np.argmin(second_derivative)
    optimal_eps = eps_values[optimal_index]
    if show_plot: Show_inertia(eps_values, distortions_smooth, optimal_eps)
    return optimal_eps

@timing
def Find_optimal_k(sp_list, algorithm_name="KMeans", k_min=None, k_max=None, show_inertia=False):
    """
    Encontra o valor ótimo de k para K-means, minimizando as diferenças entre superpixels via método do cotovelo.
    """

    print(f"Encontrando o valor ótimo de k do algoritmo {algorithm_name}...")
    if not k_min: k_min = int(len(sp_list)*0.05)
    if not k_max: k_max = int(len(sp_list)*0.5)

    if len(sp_list) < k_min or len(sp_list) < k_max:
        raise ValueError(f"Número de superpixels ({len(sp_list)}) deve ser maior ou igual a k_min ({k_min}) e k_max ({k_max}).")
    
    distortions = []
    k_values = list(range(k_min, k_max + 1))
    if k_values[0] < 2:
        k_values += 1
    if algorithm_name == "KMeans":
        for k in k_values:
            model = KMeans(n_clusters=k, random_state=0)
            model.fit(sp_list)
            distortions.append(model.inertia_)
    elif algorithm_name == "AgglomerativeClustering":
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
            if not hasattr(model, 'labels_'):
                distortions.append(float('inf'))  # fallback se não gerar labels
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
    if show_inertia: Show_inertia(k_values, distortions_smooth, optimal_k)
    
    return optimal_k