import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

from core.knn_slic_metric import SuperpixelClassifier2
from util.AdaptiveMetric import AdaptiveMetric
from util.timing import timing

os.environ["LOKY_MAX_CPU_COUNT"] = "11"

class ClusteringClassifier(SuperpixelClassifier2):
    """
    Classificador de superpixel com classificação de tipos parecidos.
    """

    def __init__(self, num_segments=0, new_model=False, LAB=True):
        """Inicializa o classificador com um modelo treinado."""

        self.new_model = new_model
        self.Similar_SP = AdaptiveMetric()
        self.LAB = LAB
        self.num_segments = num_segments

    @timing
    def Clusters2segments(self, segments, labels):
        """
        Une segmentos transitivamente com base nos rótulos de alguma segmentação.
        """

        n_superpixels = len(labels)
        sp_labels = np.unique(segments)
        new_segments = np.zeros_like(segments)
        for i in range(n_superpixels):
            # para cada superpixel (segments), coloca o rótulo do cluster (labels)
            new_segments[segments == sp_labels[i]] = labels[i]

        return new_segments

    @timing
    def Show_inertia(self, k_values, distortions, optimal_k=None):
        # Exibir gráfico para visualização
        plt.figure(figsize=(8, 5))
        prop = 100
        plt.plot(k_values, distortions/prop, '-', label=f'Inércia/{prop}')
        if optimal_k:
            plt.axvline(optimal_k, color='r', linestyle='--', label=f'valor ótimo = {optimal_k:.2f}')
        # Plotar a segunda derivada da inércia
        inertia_curve = np.gradient(np.gradient(distortions))
        plt.plot(k_values, inertia_curve, 'k-', label='Segunda derivada da inércia')
        plt.xlabel('N° de clusters(k) ou eps')
        plt.ylabel('Inércia')
        plt.title('Método do Cotovelo para Seleção de k ou eps')
        plt.legend()
        plt.show()

    def find_optimal_eps(self, sp_list, eps_min=0.1, eps_max=1.0, step=0.01, show_plot=False):
        """
        Encontra o valor ótimo de eps para o DBSCAN usando o método do cotovelo baseado na segunda derivada da inércia.
        """

        print("Encontrando o valor ótimo de eps...")
        if len(sp_list) < 2:
            raise ValueError("A lista de superpixels deve conter pelo menos dois elementos.")
        
        distortions = []
        eps_values = np.arange(eps_min, eps_max + step, step)
        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=1).fit(sp_list)
            labels = dbscan.labels_
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclui ruído (-1)
            
            # Calculando a inércia como soma das distâncias quadradas dos pontos aos seus centróides
            if num_clusters > 0:
                cluster_centers = np.array([sp_list[labels == i].mean(axis=0) for i in range(num_clusters)])
                inertia = sum(np.linalg.norm(sp_list[labels == i] - cluster_centers[i]) ** 2 for i in range(num_clusters))
                distortions.append(inertia)
            else:
                distortions.append(np.inf)
        
        # Suavizar a curva de distorção
        distortions_smooth = gaussian_filter1d(distortions, sigma=3)
        
        # Encontrar o ponto de maior curvatura (cotovelo) usando a segunda derivada
        second_derivative = np.gradient(np.gradient(distortions_smooth))
        optimal_index = np.argmin(second_derivative)
        optimal_eps = eps_values[optimal_index]
        
        if show_plot: self.Show_inertia(eps_values, distortions_smooth, optimal_eps)
        
        return optimal_eps
    
    @timing
    def Find_optimal_k(self, sp_list, k_min=None, k_max=None, show_inertia=False):
        """
        Encontra o valor ótimo de k para K-means, minimizando as diferenças entre superpixels via método do cotovelo.
        """

        print("Encontrando o valor ótimo de k...")
        if not k_min: k_min = int(len(sp_list)*0.05)
        if not k_max: k_max = int(len(sp_list)*0.95)

        if len(sp_list) < k_min or len(sp_list) < k_max:
            raise ValueError(f"Número de superpixels ({len(sp_list)}) deve ser maior ou igual a k_min ({k_min}) e k_max ({k_max}).")
        
        distortions = []
        k_values = list(range(k_min, k_max + 1))
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(sp_list)
            distortions.append(kmeans.inertia_)  # Inércia: soma das distâncias quadráticas aos centróides
        
        # Suavizar a curva de distorções
        distortions_smooth = gaussian_filter1d(distortions, sigma=5)
        # Encontrando o "cotovelo"
        inertia_curve = np.gradient(np.gradient(distortions_smooth)) # Segunda derivada da inércia
        optimal_k_index = np.argmax(inertia_curve)
        optimal_k = k_values[optimal_k_index]
        if show_inertia: self.Show_inertia(k_values, distortions_smooth, optimal_k)
        
        return optimal_k

    def Return_label_zero(self, segments, new_segments):
        """
        Retorna o rótulo do segmento zero após a junção dos segmentos.
        """
        zero_mask = segments == 0
        new_zero_label = new_segments[zero_mask][0]
        if new_zero_label != 0:
            new_segments[new_segments == 0] = new_zero_label
            new_segments[zero_mask] = 0
        return new_segments

    def Get_SP_list(self, image, segments):
        """
        Extrai características de cor e textura dos superpixels.
        """

        sp_list = []
        # ajuste para começar em 0
        segments += 1
        # Converter a imagem para o espaço de cores LAB
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        for segment_id in np.unique(segments):
            mask = segments == segment_id

            region = lab_image[mask]
            # dados de cores
            l, a, b = np.mean(region, axis=0)
            # desvio padrão de A e B em cada sp
            a_std = np.std(region[:, 1], axis=0)
            b_std = np.std(region[:, 2], axis=0)

            # Me parece que decidir quais dados colocar aqui é o caminho mais certo
            sp_list.append([a, b, a_std, b_std])

        sp_list = np.array(sp_list)
        # Normalizar os dados
        scaler = StandardScaler()
        sp_list = scaler.fit_transform(sp_list)
        return sp_list

    @timing
    def Type_classification(self, image, segments, method='KMeans', show_inertia=False):
        """
        Aplica o método de classificação especificado (DBSCAN ou KMeans) para classificar os superpixels.
        """
        
        print("Classificando em tipos similares...")
        sp_list = self.Get_SP_list(image, segments)
        
        if method == 'DBSCAN':
            optimal_eps = self.find_optimal_eps(sp_list, show_plot=show_inertia)
            dbscan = DBSCAN(eps=optimal_eps, min_samples=1, n_jobs=-1).fit(sp_list)
            labels = dbscan.labels_
        elif method == 'KMeans':
            optimal_k = self.Find_optimal_k(sp_list, show_inertia=show_inertia)
            kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(sp_list)
            labels = kmeans.labels_
        
        else:
            raise ValueError(f"Método de classificação desconhecido: {method}")

        combined_segments = self.Clusters2segments(segments, labels)
        combined_segments = self.First2Zero(combined_segments)
        return combined_segments, labels

    @timing
    def Type_visualization(self, image_path=None, method='KMeans', show_inertia=False):
        """
        Visualiza a segmentação de superpixels com o método especificado (DBSCAN ou KMeans), colorindo uma imagem.
        """
        
        image, apply_image_dir, apply_image_name_no_ext = self.Load_Image(image_path)
        segments_path = os.path.join(apply_image_dir, "segmentos",
        f"seg_finais_{apply_image_name_no_ext}_{self.num_segments}.npy")
        if not os.path.exists(segments_path):
            try:
                self.classify(threshold=5.8,
                            image_path=os.path.join(apply_image_dir,
                            f"{apply_image_name_no_ext}.jpeg"))
            except:
                raise FileNotFoundError(f"Segmentos não encontrados.")
        segments = np.load(segments_path)

        # pega a divisão de superpixels do método especificado
        segments_classified, labels = self.Type_classification(image, segments, method=method, show_inertia=show_inertia)
        print(f"Número de clusters obtidos: {len(np.unique(labels))}") 
        print(f"De um total de: {len(np.unique(segments))} segmentos iniciais")
        output_image = self.Paint_image(image, segments_classified)  
        # Salvar a imagem classificada
        self.Save_image(output_image, apply_image_dir, apply_image_name_no_ext, method)

        # Salvar os segmentos após a junção
        segments_classified_path = os.path.join(apply_image_dir, "segmentos", 
        f"seg_{method}_{apply_image_name_no_ext}_{self.num_segments}.npy")
        np.save(segments_classified_path, segments_classified)

