import numpy as np
import cv2
import os
import tkinter as tk
import joblib
from skimage.segmentation import slic
from skimage.io import imsave
from tkinter import filedialog
from time import time

from knn_apply import SuperpixelClassifier
from knn_slic import PixelClassifier2
from agrupador import InteractiveSegmentLabeler
from AdaptiveMetric import AdaptiveMetric

class SuperpixelClassifier2(SuperpixelClassifier):
    """
    Classificador de superpixels baseado em KNN e cores contrastantes.
    """

    def __init__(self, model_path, new_model=False, LAB=False):
        """Inicializa o classificador com um modelo treinado."""
        self.classifier = PixelClassifier2(model_path=model_path)
        self.num_classes = self.classifier.get_num_classes()
        self.new_model = new_model
        self.Similar_SP = AdaptiveMetric()
        self.LAB = LAB

    def Train(self, master, image):
        """
        Faz o treinamento do modelo MLP.
        """

        if not self.new_model: 
            model_path = filedialog.askopenfilename(title="Carregar modelo", filetypes=[("npy", "*.npy")])
            self.Similar_SP.load_metric(model_path)
        segments_path = filedialog.askopenfilename(title="Carregar arquivos de segmentos", filetypes=[("npy", "*.npy")])
        segments = np.load(segments_path)
        if self.LAB:
            img_temp = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            img_temp[:, :, 0] = 0
        else:
            img_temp = image
        labeler = InteractiveSegmentLabeler(master, image, segments)
        segments = labeler.run()
        self.Similar_SP.train(img_temp, segments)
        path = filedialog.asksaveasfilename(title="Salvar o modelo", defaultextension=".npy", filetypes=[("npy", "*.npy")])
        self.Similar_SP.save_metric(path)

        return path

    def Paint_image(self, image, segments):
        """
        Gerar imagem colorida para visualização
        """

        self.colors = self.generate_contrasting_colors(len(np.unique(segments)))
        # cor preta para classe NIO
        self.colors[0] = np.array([0, 0, 0])
        color_image = np.zeros(image.shape)
        for segment_value, color in zip(np.unique(segments), self.colors):
            color_image[segments == segment_value] = color
        output_image = image * 0.5 + color_image * 0.5
        return output_image

    def classify(self, image):
        """
        Aplica a classificação aos superpixels da imagem.
        """

        # Carregar modelo a partir do caminho especificado
        model_path = filedialog.askopenfilename(title="Carregar modelo", filetypes=[("npy", "*.npy")])
        segments_path = filedialog.askopenfilename(title="Carregar arquivos de segmentos", filetypes=[("npy", "*.npy")])
        it = time()
        print("Classificando os superpixels...")
        self.Similar_SP.load_metric(model_path)
        self.Similar_SP.load_metric(model_path)
        segments = np.load(segments_path)
        if self.LAB:
            img_temp = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            img_temp[:, :, 0] = 0
        else:
            img_temp = image
        new_segments = self.Similar_SP.classify_image(img_temp, segments, threshold=0.8)

        output_image = self.Paint_image(image, new_segments)        

        print(f"Tempo para classificar a imagem: {(time()-it):.1f}s")
        return output_image
    
    def SP_divide(self, image, num_segments=100):
        """
        Aplica a segmentação em superpixels para imagem.
        """

        segments_path = filedialog.asksaveasfilename(title="Caminho para salvar os segmentos",
                                                    defaultextension=".npy", filetypes=[("npy", "*.npy")])
        it = time()
        print("Dividindo a imagem em superpixels...")
        # Processamento inicial dos pixels (caso de imagem colorida ou em escala de cinza)
        if len(image.shape) > 2:
            reshaped_image = image.reshape((-1, image.shape[-1]))
        else:
            reshaped_image = image
        
        results = joblib.Parallel(n_jobs=-1)(joblib.delayed(self.classifier.predict)(pixel) for pixel in reshaped_image)

        # Reconstrói os resultados para o formato original da imagem
        results = np.array(results).reshape(image.shape[:2])
        # Segmentar a imagem usando SLIC
        segments = slic(image, n_segments=num_segments, compactness=15, sigma=1,
                        start_label=0, min_size_factor=20e-2, max_size_factor=1e+1, mask=results)
        np.save(segments_path, segments)

        print(f"Tempo para segmentar a imagem: {(time()-it):.1f}s")


def main():

    new_segments = False
    train = False
    new_model = True
    LAB = True
    # quantidade de superpixels
    sp = 200

    print("Selecione o modelo a nível de pixel para aplicação.")
    model_path = filedialog.askopenfilename(title="Selecione o modelo para aplicação", filetypes=[("joblib", "*.joblib")])
    if not model_path:
        print("Nenhum modelo selecionado. Encerrando o programa.")
        return
    print("Selecione a imagem para aplicação.")
    apply_image_path = filedialog.askopenfilename(title="Selecione a imagem para aplicação",
                                                    filetypes=[("Imagens", "*.jpeg;*.jpg;*.png")])
    if not apply_image_path:
        print("Nenhuma imagem selecionada. Encerrando o programa.")
        return

    
    # Carregar o classificador com o modelo treinado
    superpixel_classifier = SuperpixelClassifier2(model_path=model_path, new_model=new_model, LAB=LAB)
    # Carregar a imagem selecionada
    apply_image = cv2.imread(apply_image_path)
    apply_image = cv2.cvtColor(apply_image, cv2.COLOR_BGR2RGB)
    
    # apply_image = cv2.cvtColor(apply_image, cv2.COLOR_BGR2LAB)
    # apply_image[:, :, 0] = 1

    if new_segments:
        superpixel_classifier.SP_divide(apply_image, num_segments=sp)
        return

    if train:
        root = tk.Tk()
        root.withdraw()
        superpixel_classifier.Train(root, apply_image)
    else:
        # Classificar a imagem usando o classificador de superpixels
        classified_image = superpixel_classifier.classify(apply_image)
        # Salvar a imagem classificada
        classified_image = np.clip(classified_image, 0, 255)
        classified_image = classified_image.astype(np.uint8)
        classified_image = cv2.cvtColor(classified_image, cv2.COLOR_RGB2BGR)
        apply_image_dir, apply_image_name = os.path.split(apply_image_path)
        apply_image_name_no_ext, apply_image_ext = os.path.splitext(apply_image_name)
        classified_image_path = os.path.join(apply_image_dir, f"{apply_image_name_no_ext}(classified){apply_image_ext}")
        imsave(classified_image_path, classified_image)
        print(f"Imagem classificada salva em: {classified_image_path}")
        

if __name__ == "__main__":
    main()

