
import numpy as np
import joblib
from skimage.segmentation import slic
from time import time

from core.knn_apply import PixelClassifier, SuperpixelClassifier
from util.timing import timing

class PixelClassifier2(PixelClassifier):

    def predict(self, pixel):
        """Prediz a classe de um pixel com base nos k vizinhos mais próximos."""
        # Converter o pixel para o formato adequado para o KNN
        pixel = np.array(pixel).reshape(1, -1)
        dist, ind = self.kd_tree.query(pixel, k=self.k)
        neighbors = self.labels[ind[0]]
        predicted_label = np.bincount(neighbors).argmax()
        return not predicted_label

class SuperpixelClassifier2(SuperpixelClassifier):
    """Classificador de superpixels baseado em KNN e cores contrastantes."""
    def __init__(self, model_path):
        """Inicializa o classificador com um modelo treinado."""
        self.classifier = PixelClassifier2(model_path=model_path)
        self.num_classes = self.classifier.get_num_classes()

    @timing
    def classify(self, image, num_segments=100):
        """Aplica a classificação aos superpixels da imagem."""
        # Carregar modelo a partir do caminho especificado
        it = time()
        # Processamento inicial dos pixels
        if len(image.shape) > 2:
            reshaped_image = image.reshape((-1, image.shape[-1]))
        else:
            reshaped_image = image
        results = joblib.Parallel(n_jobs=-1)(joblib.delayed(self.classifier.predict)(pixel) for pixel in reshaped_image)

        # Reconstrói os resultados para o formato original da imagem
        results = np.array(results).reshape(image.shape[:2])
        # Segmentar a imagem usando SLIC
        segments = slic(image, n_segments=num_segments, compactness=15, sigma=1, start_label=0, min_size_factor=20e-2, max_size_factor=1e+1, mask=results)

        # Gerar imagem colorida para visualização
        self.colors = self.generate_contrasting_colors(len(np.unique(segments)))
        # cor preta para classe NIO
        self.colors[0] = np.array([0, 0, 0])
        color_image = np.zeros(image.shape)
        for segment_value, color in zip(np.unique(segments), self.colors):
            color_image[segments == segment_value] = color
        output_image = image * 0.5 + color_image * 0.5

        print(f"tempo para segmentar a imagem: {(time()-it):.1f}s")
        return output_image
