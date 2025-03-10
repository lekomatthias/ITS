import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter1d

from knn_slic_metric import SuperpixelClassifier2
from timing import timing

os.environ["LOKY_MAX_CPU_COUNT"] = "10"

class SuperpixelClassifier3(SuperpixelClassifier2):
    """
    Classificador de superpixel com classificação de tipos parecidos.
    """

    @timing
    def Clusters2segments(self, segments, labels):
        """
        Une segmentos transitivamente com base nos rótulos do DBSCAN.
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
        plt.plot(k_values, distortions/100, '-', label='Inércia/100')
        if optimal_k:
            plt.axvline(optimal_k, color='r', linestyle='--', label=f'k ótimo = {optimal_k}')
        # Plotar a segunda derivada da inércia
        inertia_curve = np.gradient(np.gradient(distortions))
        plt.plot(k_values, inertia_curve, 'k-', label='Segunda derivada da inércia')
        plt.xlabel('Número de clusters (k)')
        plt.ylabel('Inércia')
        plt.title('Método do Cotovelo para Seleção de k')
        plt.legend()
        plt.show()

    @timing
    def Find_optimal_k(self, sp_list, k_min=None, k_max=None, show_inertia=False):
        """
        Encontra o valor ótimo de k para K-means, minimizando as diferenças entre superpixels via método do cotovelo.
        """

        if not k_min: k_min = int(len(sp_list)*0.05)
        if not k_max: k_max = int(len(sp_list)*0.95)

        if len(sp_list) < k_min:
            raise ValueError(f"Número de superpixels ({len(sp_list)}) deve ser maior ou igual a k_min ({k_min}).")
        
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

    @timing
    def Type_classification(self, image, segments, method='DBSCAN', eps=0.5, show_inertia=False):
        """
        Aplica o método de classificação especificado (DBSCAN ou KMeans) para classificar os superpixels.
        """

        sp_list = []
        # ajuste para começar em 0
        segments += 1
        # Converter a imagem para o espaço de cores LAB
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        for segment_id in np.unique(segments):
            mask = segments == segment_id
            coords = np.argwhere(mask)
            min_row, min_col = coords.min(axis=0)
            max_row, max_col = coords.max(axis=0)

            region = lab_image[mask]
            # dados de cores
            l, a, b = np.mean(region, axis=0)
            # desvio padrão de A e B em cada sp
            a_std = np.std(region[:, 1], axis=0)
            b_std = np.std(region[:, 2], axis=0)

            # Extrair dados do canal de cores A pq ele contêm o verde da imagem
            region = lab_image[min_row:max_row + 1, min_col:max_col + 1]
            # Extrair o canal A da região
            b_channel = region[:, :, 2]  # Canal B (índice 2 no espaço LAB)
            mask_region = mask[min_row:max_row + 1, min_col:max_col + 1]
            # Aplicar a máscara ao canal B (manter apenas os pixels do superpixel)
            b_channel = np.where(mask_region, b_channel, 0)
            glcm = graycomatrix(b_channel, distances=[1], angles=[0], symmetric=True, normed=False)
            # variação de cor entre vizinhos
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            # uniformidade de distribuição de cor
            energy = graycoprops(glcm, 'energy')[0, 0]
            # repetitividade
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            # entropia das cores ignorando log0
            entropy = -np.sum(glcm * np.log2(glcm + 1e-10))

            # Me parece que decidir quais dados colocar aqui é o caminho mais certo
            sp_list.append([a, b, a_std, b_std])

        sp_list = np.array(sp_list)
        # Normalizar os dados
        scaler = StandardScaler()
        sp_list = scaler.fit_transform(sp_list)
        
        if method == 'DBSCAN':
            dbscan = DBSCAN(eps=eps, min_samples=1, n_jobs=-1).fit(sp_list)
            combined_segments = self.Clusters2segments(segments, dbscan.labels_)
            combined_segments = self.Return_label_zero(segments, combined_segments)
            return combined_segments, dbscan.labels_
        elif method == 'KMeans':
            optimal_k = self.Find_optimal_k(sp_list, show_inertia=show_inertia)
            kmeans = KMeans(n_clusters=optimal_k, random_state=0)
            kmeans.fit(sp_list)
            combined_segments = self.Clusters2segments(segments, kmeans.labels_)
            combined_segments = self.Return_label_zero(segments, combined_segments)
            return combined_segments, kmeans.labels_
        else:
            raise ValueError("Método de classificação desconhecido. Use 'DBSCAN' ou 'KMeans'.")

    def Type_visualization(self, method='DBSCAN', eps=0.5, show_inertia=False):
        """
        Visualiza a segmentação de superpixels com o método especificado (DBSCAN ou KMeans), colorindo uma imagem.
        """
        
        image, apply_image_dir, apply_image_name_no_ext = self.Load_Image()
        segments_path = os.path.join(apply_image_dir, "segmentos",
        f"seg_finais_{apply_image_name_no_ext}_{self.num_segments}.npy")
        segments = np.load(segments_path)

        # pega a divisão de superpixels do método especificado
        segments_classified, labels = self.Type_classification(image, segments, method=method, eps=eps, show_inertia=show_inertia)

        print(f"Número de clusters obtidos: {len(np.unique(labels))}") 
        print(f"De um total de: {len(np.unique(segments))} segmentos iniciais")
        output_image = self.Paint_image(image, segments_classified)  
        
        # Salvar a imagem classificada
        self.Save_image(output_image, apply_image_dir, apply_image_name_no_ext, method)

        # Salvar os segmentos após a junção
        segments_classified_path = os.path.join(apply_image_dir, "segmentos", 
        f"seg_{method}_{apply_image_name_no_ext}_{self.num_segments}.npy")
        np.save(segments_classified_path, segments_classified)

if __name__ == "__main__":
    
    classifier = SuperpixelClassifier3(num_segments=200)
    # classifier.SP_divide()
    # classifier.Train()
    # classifier.classify(5.8)
    # classifier.Type_visualization(method='DBSCAN', eps=0.5)
    classifier.Type_visualization(method='KMeans', show_inertia=True)
