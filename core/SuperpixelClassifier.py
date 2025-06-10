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
        """
        Inicializa o classificador com um modelo treinado.
        """

        self.new_model = new_model
        self.LAB = LAB
        self.num_segments = num_segments
    
    @timing
    def MakeMask(self, image_path=None, mask_path=None):

        image, dir, n_ext = Load_Image(image_path)
        if mask_path is None:
            mask_path = os.path.join(dir, "mascaras",
                                    f"mask_{n_ext}.npy")
        
        if os.path.exists(mask_path):
            print(f"Máscara já existe em {mask_path}.")
            results = np.load(mask_path)
        else:
            maskClassifier = MaskClassifier()
            results = maskClassifier.Classify(image, mask_path)
            results = results.astype(np.uint8)
            kernel = np.ones((5, 5), np.uint8)
            results = cv2.morphologyEx(results, cv2.MORPH_OPEN, kernel)
            results = cv2.morphologyEx(results, cv2.MORPH_CLOSE, kernel)
        
        return results

    @timing
    def SP_divide(self, image_path=None, mode="slic"):
        """
        Aplica a segmentação em superpixels para imagem.
        """

        image, apply_image_dir, apply_image_name_no_ext = Load_Image(image_path)
        if image_path is None:
            image_path = os.path.join(apply_image_dir, f"{apply_image_name_no_ext}.jpg")
        
        auto_num_segments = False
        if self.num_segments == 0:
            pix = GetPixelsOfArea(image=image, target_height=600)
            auto_num_segments = True

        segments_path = os.path.join(apply_image_dir, "segmentos",
                                     f"{mode}", f"segmentos_{apply_image_name_no_ext}_{self.num_segments}.npy")
        
        mask_path = os.path.join(apply_image_dir, "mascaras",
                                 f"mask_{apply_image_name_no_ext}.npy")
        
        results = self.MakeMask(image_path, mask_path)

        if auto_num_segments:
            self.num_segments = int(np.count_nonzero(results) // pix)
            print(f"Quantidade de superpixels: {self.num_segments}")

        if mode == "slic":
            segments = slic(image, n_segments=self.num_segments, compactness=15, sigma=1,
                            start_label=0, min_size_factor=2e-1, max_size_factor=1e+1, mask=results)
        else:
            try:
                segments = np.load(os.path.join(apply_image_dir, f"{mode}_csv_npy", f"{apply_image_name_no_ext}.npy"))
                mask = np.load(mask_path)
                mask = np.where(mask > 0, 1, 0).astype(np.uint8)
                segments = segments*mask
            except Exception as e:
                raise ValueError(f"algoritmo não encontrado no path base da função SP_divide.\n{e}")

        # Esta função decide se todos os segmentos devem realmente ser conectados.
        segments = Enforce_connectivity(segments)
        segments = First2Zero(segments)

        segments_dir = os.path.join(apply_image_dir, "segmentos")
        create_folders(segments_dir, [f"{mode}"])
        np.save(segments_path, segments)

    @timing
    def classify(self, threshold=5.8, image_path=None, mode="slic", show_data=False):
        """
        Aplica a classificação aos superpixels da imagem.
        """
        
        image, apply_image_dir, apply_image_name_no_ext = Load_Image(image_path)

        segments_path = os.path.join(apply_image_dir, "segmentos",
                                     f"{mode}", f"segmentos_{apply_image_name_no_ext}_{self.num_segments}.npy")
        if not os.path.exists(segments_path):
            self.SP_divide(image_path=os.path.join(apply_image_dir, apply_image_name_no_ext+".jpg"),
                            mode=mode)

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
            img_temp[:, :, 0] = 0 # Zera a luminizidade
        else:
            img_temp = image
        new_segments = Similar_SP.classify_image(img_temp, segments, threshold=threshold,
                                                      show_data=show_data)
        output_image = Paint_image(image, new_segments)
        
        path_save = os.path.join(apply_image_dir, "classificadas", "segmentadas")
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        Save_image(output_image, path_save, apply_image_name_no_ext, self.num_segments, f"{mode}")
        
        final_segments_path = os.path.join(apply_image_dir, "segmentos",
                                           f"seg_finais_{apply_image_name_no_ext}_{self.num_segments}.npy")
        np.save(final_segments_path, new_segments)

        print(f"Novos segmentos salvos em: {final_segments_path}")

        return output_image
