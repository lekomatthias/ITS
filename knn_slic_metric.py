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

    def __init__(self,new_model=False, LAB=False, num_segments=100):
        """Inicializa o classificador com um modelo treinado."""

        self.new_model = new_model
        self.Similar_SP = AdaptiveMetric()
        self.LAB = LAB
        self.num_segments = num_segments

    def Load_Image(self):
        """
        Carrega a imagem para aplicação.
        """

        apply_image_path = filedialog.askopenfilename(title="Selecione a imagem para aplicação",
                                                      filetypes=[("Imagens", "*.jpeg;*.jpg;*.png")])
        if not os.path.exists(apply_image_path):
            print("Nenhuma imagem selecionada. Encerrando o programa.")
            exit()
        apply_image_dir, apply_image_name = os.path.split(apply_image_path)
        apply_image_name_no_ext, _ = os.path.splitext(apply_image_name)
        image = cv2.imread(apply_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, apply_image_dir, apply_image_name_no_ext

    def SP_divide(self):
        """
        Aplica a segmentação em superpixels para imagem.
        """

        image, apply_image_dir, apply_image_name_no_ext = self.Load_Image()
        segments_path = os.path.join(apply_image_dir,
        f"segmentos_{apply_image_name_no_ext}_{self.num_segments}.npy")

        print("Selecione o modelo a nível de pixel para aplicação.")
        model_path = filedialog.askopenfilename(title="Selecione o modelo para aplicação",
                                                filetypes=[("joblib", "*.joblib")])
        if not model_path:
            print("Nenhum modelo selecionado. Encerrando o programa.")
            return
        classifier = PixelClassifier2(model_path=model_path)

        it = time()
        print("Dividindo a imagem em superpixels...")
        # Processamento inicial dos pixels (caso de imagem colorida ou em escala de cinza)
        if len(image.shape) > 2:
            reshaped_image = image.reshape((-1, image.shape[-1]))
        else:
            reshaped_image = image
        
        results = joblib.Parallel(n_jobs=-1)(joblib.delayed(classifier.predict)(pixel) for pixel in reshaped_image)

        # Reconstrói os resultados para o formato original da imagem
        results = np.array(results).reshape(image.shape[:2])
        # Segmentar a imagem usando SLIC
        segments = slic(image, n_segments=self.num_segments, compactness=15, sigma=1, enforce_connectivity=True, # força nada kkkkk
                        start_label=0, min_size_factor=2e-1, max_size_factor=1e+1, mask=results)
        print(f"Tempo para segmentar a imagem: {(time()-it):.1f}s")
        np.save(segments_path, segments)

    def Train(self, master):
        """
        Faz o treinamento do modelo MLP.
        """

        image, apply_image_dir, apply_image_name_no_ext = self.Load_Image()
        
        segments_path = os.path.join(apply_image_dir,
                                    f"segmentos_{apply_image_name_no_ext}_{self.num_segments}.npy")
        if not os.path.exists(segments_path):
            self.SP_divide()

        # seleção de modelo métrica para continuar o treinamento
        metric_name = f"metrica_{'LAB' if self.LAB else 'RGB'}_{self.num_segments}.npy"
        metric_path = os.path.join(apply_image_dir, metric_name)
        if os.path.exists(metric_path) and not self.new_model:
            self.Similar_SP.load_metric(metric_path)
            print(f"Modelo carregado de {metric_path}")
        else:
            print("Nenhum modelo encontrado, treinando um novo modelo.")

        segments = np.load(segments_path)
        if self.LAB:
            img_temp = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            # Zera a luminizidade
            img_temp[:, :, 0] = 0
        else:
            img_temp = image
        labeler = InteractiveSegmentLabeler(master, image, segments)
        segments = labeler.run()
        self.Similar_SP.train(img_temp, segments)
        self.Similar_SP.save_metric(metric_path)
        print(f"Modelo salvo em {metric_path}")

        return metric_path

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

    def classify(self, threshold=1, show_data=False):
        """
        Aplica a classificação aos superpixels da imagem.
        """
        
        image, apply_image_dir, apply_image_name_no_ext = self.Load_Image()

        segments_path = os.path.join(apply_image_dir,
                                    f"segmentos_{apply_image_name_no_ext}_{self.num_segments}.npy")
        if not os.path.exists(segments_path):
            self.SP_divide()

        metric_name = f"metrica_{'LAB' if self.LAB else 'RGB'}_{self.num_segments}.npy"
        print(metric_name)
        metric_path = os.path.join(apply_image_dir, metric_name)
        if not os.path.exists(metric_path):
            model_path = filedialog.askopenfilename(title="Carregar modelo", filetypes=[("npy", "*.npy")])
            if not model_path:
                print("Nenhum modelo selecionado. Encerrando o programa.")
                exit()
            self.Similar_SP.load_metric(model_path)
        else:
            self.Similar_SP.load_metric(metric_path)

        print("Classificando os superpixels...")
        segments = np.load(segments_path)
        if self.LAB:
            img_temp = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            # Zera a luminizidade
            img_temp[:, :, 0] = 0
        else:
            img_temp = image
        new_segments = self.Similar_SP.classify_image(img_temp, segments, threshold=threshold,
                                                      show_data=show_data)

        output_image = self.Paint_image(image, new_segments)        

        # Salvar a imagem classificada
        output_image = np.clip(output_image, 0, 255)
        output_image = output_image.astype(np.uint8)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        output_image_path = os.path.join(apply_image_dir, f"{apply_image_name_no_ext}(classified).jpeg")
        imsave(output_image_path, output_image)
        print(f"Imagem classificada salva em: {output_image_path}")

        return output_image

def main():

    new_segments = False
    train = False
    new_model = False
    LAB = True
    num_segments = 200
    threshold = 4

    superpixel_classifier = SuperpixelClassifier2(new_model=new_model, LAB=LAB, num_segments=num_segments)

    if new_segments:
        # Divide a imagem em superpixels e salva o arquivo de segmentos
        superpixel_classifier.SP_divide()
        return

    if train:
        # Treina a métrica adaptativa e salva o arquivo de modelo
        root = tk.Tk()
        root.withdraw()
        superpixel_classifier.Train(root)
    else:
        # Classifica a imagem e salva a imagem colorida
        superpixel_classifier.classify(threshold, show_data=True)

if __name__ == "__main__":
    main()

