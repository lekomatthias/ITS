import numpy as np
import os
import joblib
from skimage.segmentation import slic
from collections import Counter
from time import time

from util.Image_manager import generate_contrasting_colors

# Define o limite máximo de núcleos lógicos
os.environ["LOKY_MAX_CPU_COUNT"] = "12"

class PixelClassifier:
    """Classificador KNN com base em dados de treinamento pré-carregados."""
    def __init__(self, model_path, k=3):
        """Carrega um modelo KNN pré-treinado de um arquivo .joblib."""
        try:
            self.k = k
            # Carrega o modelo como um dicionário
            self.model_data = joblib.load(model_path)  
            self.kd_tree = self.model_data["kd_tree"]
            self.labels = self.model_data["labels"]
            self.classes_ = self.model_data["classes_"]
        except FileNotFoundError:
            print(f"Erro: O arquivo do modelo '{model_path}' não foi encontrado.")
            exit()
        except KeyError as e:
            print(f"Erro: O modelo salvo está faltando a chave esperada: {e}")
            exit()

    def predict(self, pixel):
        """Prediz a classe de um pixel com base nos k vizinhos mais próximos."""
        # Converter o pixel para o formato adequado para o KNN
        pixel = np.array(pixel).reshape(1, -1)
        dist, ind = self.kd_tree.query(pixel, k=self.k)
        neighbors = self.labels[ind[0]]
        predicted_label = np.bincount(neighbors).argmax()
        return predicted_label

    def get_num_classes(self):
        """Obtém automaticamente o número de classes do modelo KNN."""
        return len(self.classes_)

class SuperpixelClassifier:
    """Classificador de superpixels baseado em KNN e cores contrastantes."""
    def __init__(self, model_path):
        """Inicializa o classificador com um modelo treinado."""
        self.classifier = PixelClassifier(model_path=model_path)
        self.num_classes = self.classifier.get_num_classes()
        self.colors = self.generate_contrasting_colors(self.num_classes)

    # ajuste de código legado
    def generate_contrasting_colors(self, num_colors):
        """Gera cores contrastantes para diferentes classes."""
        return generate_contrasting_colors(num_colors)

    def classify(self, image, num_segments=100):
        """Aplica a classificação aos superpixels da imagem."""
        it = time()
        segments = slic(image, n_segments=num_segments, compactness=10, sigma=1, start_label=1)
        output_image = image.copy()
        class_counts = {i: 0 for i in range(self.num_classes)}

        # Função que processa cada superpixel
        def process_segment(segment_id):
            mask = segments == segment_id
            pixel_classes = []

            # Para cada pixel dentro do superpixel, predizer a classe
            for pixel in image[mask].reshape(-1, 3):  # Reshape para um vetor de pixels RGB
                predicted_label = self.classifier.predict(pixel)
                pixel_classes.append(predicted_label)

            most_common_class = Counter(pixel_classes).most_common(1)[0][0]
            # Atribui a cor do superpixel de acordo com a classe mais comum
            overlay_color = np.array(self.colors[most_common_class % len(self.colors)], dtype=np.uint8)
            return (mask, overlay_color, most_common_class)

        # Paralelizar o processamento de superpixels
        results = joblib.Parallel(n_jobs=-1)(joblib.delayed(process_segment)(segment_id) for segment_id in np.unique(segments))

        # Atualizar a imagem e contar os superpixels de cada classe
        for mask, overlay_color, most_common_class in results:
            output_image[mask] = (output_image[mask] * 0.5 + overlay_color * 0.5).astype(np.uint8)
            class_counts[most_common_class] += 1

        print("Número de superpixels por classe:")
        for class_id, count in class_counts.items():
            print(f"Classe {class_id}: {count} superpixels")

        print(f"tempo para classificar a imagem: {(time()-it):.1f}s")
        return output_image

