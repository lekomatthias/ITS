import numpy as np
import cv2
import os
from skimage.segmentation import slic
from tkinter import filedialog

from core.MaskClassifier import MaskClassifier
from util import *
from util.timing import timing

class SuperpixelClassifier:
    """
    Classificador de superpixels baseado em KNN e cores contrastantes.
    """

    def __init__(self, num_segments=0, new_model=False, LAB=True):
        """Inicializa o classificador com um modelo treinado."""

        self.new_model = new_model
        self.LAB = LAB
        self.num_segments = num_segments

    def Train(self, image_path=None):
        """
        Faz o treinamento da matriz métrica.
        """

        image, apply_image_dir, apply_image_name_no_ext = Load_Image(image_path)
        
        segments_path = os.path.join(apply_image_dir, "segmentos",
                                     f"segmentos_{apply_image_name_no_ext}_{self.num_segments}.npy")
        if not os.path.exists(segments_path):
            self.SP_divide(image_path=os.path.join(apply_image_dir, apply_image_name_no_ext+".jpg"))

        # seleção de modelo métrica para continuar o treinamento
        metric_name = f"metrica_{'LAB' if self.LAB else 'RGB'}_{self.num_segments}.npy"
        metric_path = os.path.join(apply_image_dir, "metricas", metric_name)
        if not os.path.exists(metric_path) and self.new_model:
            print("Nenhum modelo encontrado, treinando um novo modelo.")
        Similar_SP = AdaptiveMetric()
        Similar_SP.load_metric(metric_path)
        print(f"Modelo carregado de {metric_path}")

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
            Similar_SP.train(img_temp, segments)
        except:
            print("Nenhuma segmentação selecionada. Encerrando o programa.")
            exit()
        Similar_SP.save_metric(metric_path)
        print(f"Modelo salvo em {metric_path}")

        return metric_path
    
    @timing
    def SP_divide(self, image_path=None, standard_size=None):
        """
        Aplica a segmentação em superpixels para imagem.
        """

        image, apply_image_dir, apply_image_name_no_ext = Load_Image(image_path)

        # Verifica se a imagem é muito grande e redimensiona
        original_size = image.shape[:2]
        if standard_size == None: standard_size = image.shape[1]//2
        if image.shape[1] > 3000:
            scale_factor = standard_size / image.shape[1]
            new_width = standard_size
            new_height = int(image.shape[0] * scale_factor)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"Imagem redimensionada para {new_width}x{new_height}.")
        
        auto_num_segments = False
        if self.num_segments == 0:
            pix = GetPixelsOfArea(image=image, target_height=600)
            auto_num_segments = True

        segments_path = os.path.join(apply_image_dir, "segmentos",
                                     f"segmentos_{apply_image_name_no_ext}_{self.num_segments}.npy")
        
        mask_path = os.path.join(apply_image_dir, "mascaras",
                                 f"mask_{apply_image_name_no_ext}.npy")
        
        if os.path.exists(mask_path):
            print(f"Máscara já existe em {mask_path}. Carregando...")
            results = np.load(mask_path)
        else:
            maskClassifier = MaskClassifier()
            results = maskClassifier.Classify(image, mask_path)

        if auto_num_segments:
            self.num_segments = int(np.count_nonzero(results) // pix)
            print(f"Quantidade de superpixels: {self.num_segments}")

        # Segmentar a imagem usando SLIC
        segments = slic(image, n_segments=self.num_segments, compactness=15, sigma=1,
                        start_label=0, min_size_factor=2e-1, max_size_factor=1e+1, mask=results)

        # segments = np.load(segments_path)
        # mask = np.load(mask_path)
        # mask = np.where(mask > 0, 1, 0).astype(np.uint8)
        # segments = segments*mask

        # Esta função decide se todos os segmentos devem realmente ser conectados.
        segments = Enforce_connectivity(segments)
        segments = First2Zero(segments)

        # Redimensiona os segmentos e a máscara para o tamanho original, se necessário
        if image.shape[:2] != original_size:
            segments = cv2.resize(segments.astype(np.uint8), (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
            results = cv2.resize(results.astype(np.uint8), (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
            print(f"Segmentos e máscara redimensionados para o tamanho original {original_size[1]}x{original_size[0]}.")

        np.save(segments_path, segments)

    @timing
    def classify(self, threshold=1, image_path=None, show_data=False):
        """
        Aplica a classificação aos superpixels da imagem.
        """
        
        image, apply_image_dir, apply_image_name_no_ext = Load_Image(image_path)

        segments_path = os.path.join(apply_image_dir, "segmentos",
                                     f"segmentos_{apply_image_name_no_ext}_{self.num_segments}.npy")
        if not os.path.exists(segments_path):
            self.SP_divide(image_path=os.path.join(apply_image_dir, apply_image_name_no_ext+".jpg"))

        metric_name = f"metrica_{'LAB' if self.LAB else 'RGB'}_{self.num_segments}.npy"
        print(f"Nome da métrica padrão: {metric_name}")
        metric_path = os.path.join(apply_image_dir, "metricas", metric_name)
        if not os.path.exists(metric_path):
            metric_path = filedialog.askopenfilename(title="Carregar modelo", filetypes=[("npy", "*.npy")])
            if not metric_path:
                print("Nenhum modelo selecionado. Encerrando o programa.")
                exit()
        Similar_SP = AdaptiveMetric()
        Similar_SP.load_metric(metric_path)

        print("Classificando os superpixels...")
        segments = np.load(segments_path)
        if self.LAB:
            img_temp = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            # Zera a luminizidade
            img_temp[:, :, 0] = 0
        else:
            img_temp = image
        new_segments = Similar_SP.classify_image(img_temp, segments, threshold=threshold,
                                                      show_data=show_data)
        
        new_segments = First2Zero(new_segments)
        output_image = Paint_image(image, new_segments)

        # Salvar a imagem classificada
        Save_image(output_image, apply_image_dir, apply_image_name_no_ext, self.num_segments, "classified")

        # Salvar os novos segmentos
        final_segments_path = os.path.join(apply_image_dir, "segmentos",
                                           f"seg_finais_{apply_image_name_no_ext}_{self.num_segments}.npy")
        np.save(final_segments_path, new_segments)
        print(f"Novos segmentos salvos em: {final_segments_path}")

        return output_image
