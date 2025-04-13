
import numpy as np
import cv2
import os
import tkinter as tk
import joblib
from skimage.segmentation import slic
from skimage.io import imsave
from tkinter import filedialog
from time import time

from knn_apply import PixelClassifier, SuperpixelClassifier
from timing import timing

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

def main():

    root = tk.Tk()
    root.withdraw()

    print("Selecione o modelo pixel para aplicação.")
    model_path = filedialog.askopenfilename(title="Selecione o modelo para aplicação", filetypes=[("joblib", "*.joblib")])

    # Carregar o classificador com o modelo treinado
    superpixel_classifier = SuperpixelClassifier2(model_path=model_path, )

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
    classified_image = superpixel_classifier.classify(apply_image, num_segments=100)

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

