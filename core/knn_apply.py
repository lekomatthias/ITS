import numpy as np
import cv2
import os
import joblib
import tkinter as tk
from skimage.segmentation import slic
from skimage.io import imsave
from collections import Counter
from tkinter import filedialog
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

def main():

    root = tk.Tk()
    root.withdraw()

    model_path = filedialog.askopenfilename(title="Selecione o modelo para aplicação", filetypes=[("joblib", "*.joblib")])

    # Carregar o classificador com o modelo treinado
    superpixel_classifier = SuperpixelClassifier(model_path=model_path)

    print("Selecione a imagem para aplicação.")
    apply_image_path = filedialog.askopenfilename(title="Selecione a imagem para aplicação", filetypes=[("Imagens", "*.jpeg;*.jpg;*.png")])
    if not apply_image_path:
        print("Nenhuma imagem selecionada. Encerrando o programa.")
        exit()

    # Carregar a imagem selecionada
    apply_image = cv2.imread(apply_image_path)
    apply_image = cv2.cvtColor(apply_image, cv2.COLOR_BGR2RGB)

    # Classificar a imagem usando o classificador de superpixels
    print("Processando dados...")
    classified_image = superpixel_classifier.classify(apply_image, num_segments=30)

    # Salvar a imagem classificada
    classified_image = np.clip(classified_image, 0, 255)  # Garante valores no intervalo [0, 255]
    classified_image = classified_image.astype(np.uint8)  # Converte para CV_8U
    classified_image = cv2.cvtColor(classified_image, cv2.COLOR_RGB2BGR)
    apply_image_dir, apply_image_name = os.path.split(apply_image_path)
    apply_image_name_no_ext, apply_image_ext = os.path.splitext(apply_image_name)
    classified_image_path = os.path.join(apply_image_dir, f"{apply_image_name_no_ext}(classified){apply_image_ext}")
    imsave(classified_image_path, classified_image)
    print(f"Imagem classificada salva em: {classified_image_path}")

if __name__ == "__main__":
    main()
