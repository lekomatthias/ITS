import numpy as np
import cv2
import os
from skimage.io import imsave
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops

from knn_slic_metric import SuperpixelClassifier2

class SuperpixelClassifier3(SuperpixelClassifier2):
    """
    Classificador de superpixel com classificação de tipos parecidos.
    """

    def clusters2segments(self, segments, labels):
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

    def Superpixel_DBSCAN(self, image, segments, eps=0.5):
        """
        Aplica o DBSCAN para classificar os superpixels.
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
            a_channel = region[:, :, 1]  # Canal A (índice 1 no espaço LAB)
            mask_region = mask[min_row:max_row + 1, min_col:max_col + 1]
            # Aplicar a máscara ao canal A (manter apenas os pixels do superpixel)
            a_channel = np.where(mask_region, a_channel, 0)
            glcm = graycomatrix(a_channel, distances=[1], angles=[0], symmetric=True, normed=False)
            # variação de cor entre vizinhos
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            # uniformidade de distribuição de cor
            energy = graycoprops(glcm, 'energy')[0, 0]
            # repetitividade
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            # entropia das cores ignorando log0
            entropy = -np.sum(glcm * np.log2(glcm + 1e-10))

            # introdução com citação das refs
            # hipóteses
            # resultados atuais com testes/ resultados parciais
            # cronograma
            # referências

            # Me parece que decidir quais dados colocar aqui é o caminho mais certo
            sp_list.append([a, b, a_std, b_std])

        sp_list = np.array(sp_list)
        # Normalizar os dados
        scaler = StandardScaler()
        sp_list = scaler.fit_transform(sp_list)
        
        dbscan = DBSCAN(eps=eps, min_samples=1, n_jobs=-1).fit(sp_list)
        combined_segments = self.clusters2segments(segments, dbscan.labels_)
        
        return combined_segments, dbscan.labels_
    
    def DBSCAN_visualization(self, eps=0.5):
        """
        Visualiza a segmentação de superpixels com DBSCAN, colorindo uma imagem.
        """
        
        image, apply_image_dir, apply_image_name_no_ext = self.Load_Image()
        segments_path = os.path.join(apply_image_dir, "segmentos",
        f"seg_finais_{apply_image_name_no_ext}_{self.num_segments}.npy")
        segments = np.load(segments_path)

        # pega a divisão de superpixels do DBSCAN
        segments_dbscan, labels = self.Superpixel_DBSCAN(image, segments, eps=eps)

        print(f"Número de clusters obtidos: {len(np.unique(labels)) - 1}") 
        print(f"De um total de: {len(np.unique(segments)) - 1}")
        output_image = self.Paint_image(image, segments_dbscan)  
        
        # Salvar a imagem classificada
        self.Save_image(output_image, apply_image_dir, apply_image_name_no_ext, "DBSCAN")

        # Salvar os segmentos após a junção
        segments_dbscan_path = os.path.join(apply_image_dir, "segmentos", 
        f"seg_DBSCAN_{apply_image_name_no_ext}_{self.num_segments}.npy")
        np.save(segments_dbscan_path, segments_dbscan)

if __name__ == "__main__":
    
    classifier = SuperpixelClassifier3(num_segments=200)
    # classifier.SP_divide()
    # classifier.Train()
    # classifier.classify(5.8)
    classifier.DBSCAN_visualization(eps=0.5)
