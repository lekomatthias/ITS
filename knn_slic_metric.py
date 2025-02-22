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
from SP_grouper import InteractiveSegmentLabeler
from AdaptiveMetric import AdaptiveMetric
from Enforce_connectivity import Enforce_connectivity

class SuperpixelClassifier2(SuperpixelClassifier):
    """
    Classificador de superpixels baseado em KNN e cores contrastantes.
    """

    def __init__(self, num_segments=200, new_model=False, LAB=True):
        """Inicializa o classificador com um modelo treinado."""

        self.new_model = new_model
        self.Similar_SP = AdaptiveMetric()
        self.LAB = LAB
        self.num_segments = num_segments
        # Garante que os diretórios necessários existam
        self.ensure_directories()

    def ensure_directories(self):
        """
        Garante que os diretórios necessários existam.
        """

        base_dir = os.path.dirname(os.path.abspath(__file__))
        directories = ["segmentos", "classificadas", "metricas", "mascaras"]
        for directory in directories:
            dir_path = os.path.join(base_dir, directory)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

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
        segments_path = os.path.join(apply_image_dir, "segmentos",
                                     f"segmentos_{apply_image_name_no_ext}_{self.num_segments}.npy")
        mask_path = os.path.join(apply_image_dir, "mascaras",
                                 f"mask_{apply_image_name_no_ext}.npy")

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
        segments = slic(image, n_segments=self.num_segments, compactness=15, sigma=1,
                        start_label=0, min_size_factor=2e-1, max_size_factor=1e+1, mask=results)
        print(f"Tempo para segmentar a imagem: {(time()-it):.1f}s")
        it = time()
        segments = Enforce_connectivity(segments)
        print(f"Tempo para garantir conectividade: {(time()-it):.1f}s")
        np.save(mask_path, results)
        self.Create_image_with_segments(mask_path)
        np.save(segments_path, segments)

    def Train(self):
        """
        Faz o treinamento da matriz métrica.
        """

        image, apply_image_dir, apply_image_name_no_ext = self.Load_Image()
        
        segments_path = os.path.join(apply_image_dir, "segmentos",
                                     f"seg_DBSCAN_{apply_image_name_no_ext}_{self.num_segments}.npy")
        if not os.path.exists(segments_path):
            self.SP_divide()

        # seleção de modelo métrica para continuar o treinamento
        metric_name = f"metrica_{'LAB' if self.LAB else 'RGB'}_{self.num_segments}.npy"
        metric_path = os.path.join(apply_image_dir, "metricas", metric_name)
        if os.path.exists(metric_path) and not self.new_model:
            self.Similar_SP.load_metric(metric_path)
            print(f"Modelo carregado de {metric_path}")
        else:
            print("Nenhum modelo encontrado, treinando um novo modelo.")

        segments = np.load(segments_path)
        if self.LAB:
            img_temp = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            # Zera a luminozidade
            img_temp[:, :, 0] = 0
        else:
            img_temp = image
        labeler = InteractiveSegmentLabeler(image, segments)
        try:
            segments = labeler.run()
            self.Similar_SP.train(img_temp, segments)
        except:
            print("Nenhuma segmentação selecionada. Encerrando o programa.")
            exit()
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
        output_image = image * 0.4 + color_image * 0.6
        return output_image
    
    def Save_image(self, output_image, apply_image_dir, apply_image_name_no_ext, type="classified"):
        """
        Salva a imagem.
        """

        output_image = np.clip(output_image, 0, 255)
        output_image = output_image.astype(np.uint8)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        output_image_path = os.path.join(apply_image_dir, "classificadas",
                                         f"({type}){apply_image_name_no_ext}_{self.num_segments}.jpeg")
        imsave(output_image_path, output_image)
        print(f"Imagem classificada salva em: {output_image_path}")

    def classify(self, threshold=1, show_data=False):
        """
        Aplica a classificação aos superpixels da imagem.
        """
        
        image, apply_image_dir, apply_image_name_no_ext = self.Load_Image()

        segments_path = os.path.join(apply_image_dir, "segmentos",
                                     f"segmentos_{apply_image_name_no_ext}_{self.num_segments}.npy")
        if not os.path.exists(segments_path):
            self.SP_divide()

        metric_name = f"metrica_{'LAB' if self.LAB else 'RGB'}_{self.num_segments}.npy"
        print(metric_name)
        metric_path = os.path.join(apply_image_dir, "metricas", metric_name)
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
        self.Save_image(output_image, apply_image_dir, apply_image_name_no_ext, "classified")

        # Salvar os novos segmentos
        final_segments_path = os.path.join(apply_image_dir, "segmentos",
                                           f"seg_finais_{apply_image_name_no_ext}_{self.num_segments}.npy")
        np.save(final_segments_path, new_segments)
        print(f"Novos segmentos salvos em: {final_segments_path}")

        return output_image
    
    def Create_image_with_segments(self, seg_path=None):
        """
        Cria uma imagem com os segmentos coloridos.
        """

        if not seg_path:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            segment_dir = os.path.join(base_dir, "segmentos")
            initial_dir = segment_dir if os.path.exists(segment_dir) else base_dir
            seg_path = filedialog.askopenfilename(initialdir=initial_dir,
                                                title="Selecione os segmentos para criação da imagem",
                                                filetypes=[("npy", "*.npy")])
        
        if not seg_path or not os.path.exists(seg_path):
            print("Nenhum segmento selecionado. Encerrando o programa.")
            exit()
        seg_dir, seg_name = os.path.split(seg_path)

        segments = np.load(seg_path)
        output_image = np.zeros((*segments.shape, 3), dtype=np.uint8)
        if len(np.unique(segments)) == 2: colors = [(0, 0, 0), (255, 255, 255)]
        else: colors = self.generate_contrasting_colors(len(np.unique(segments)))
        segments = segments.astype(int)
        for segment_value, color in zip(np.unique(segments), colors):
            output_image[segments == segment_value] = color
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        imsave(os.path.join(seg_dir, f"{seg_name}_color.jpeg"), output_image)
        print(f"Imagem com segmentos coloridos salva em: {os.path.join(seg_dir, f'{seg_name}_color.jpeg')}")
        return output_image

def main():
    new_segments = False
    train = False
    new_model = True
    LAB = True
    num_segments = 200
    threshold = 5.8

    superpixel_classifier = SuperpixelClassifier2(new_model=new_model, LAB=LAB, num_segments=num_segments)

    # superpixel_classifier.Create_image_with_segments()
    # exit()

    if new_segments:
        # Divide a imagem em superpixels e salva o arquivo de segmentos
        superpixel_classifier.SP_divide()
        return

    if train:
        # Treina a métrica adaptativa e salva o arquivo de modelo
        superpixel_classifier.Train()
    else:
        # Classifica a imagem e salva a imagem colorida
        superpixel_classifier.classify(threshold, show_data=False)

if __name__ == "__main__":
    main()

