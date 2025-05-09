
from tkinter import filedialog
import numpy as np
import joblib

from core.PixelClassifier import PixelClassifier
from util.timing import timing
from util.Image_manager import Create_image_with_segments

class MaskClassifier:
    @timing
    def Classify(self, image, mask_path):
        """
        Cria uma máscara para a imagem, onde os superpixels são marcados como 1 e o fundo como 0.
        """

        print("Selecione o modelo a nível de pixel para aplicação.")
        model_path = filedialog.askopenfilename(title="Selecione o modelo para aplicação",
                                                filetypes=[("joblib", "*.joblib")])
        if not model_path:
            print("Nenhum modelo selecionado. Encerrando o programa.")
            return
        classifier = PixelClassifier(model_path=model_path)

        print("Criando márcara da imagem...")
        # Processamento inicial dos pixels (caso de imagem colorida ou em escala de cinza)
        if len(image.shape) > 2:
            reshaped_image = image.reshape((-1, image.shape[-1]))
        else:
            reshaped_image = image

        mask = joblib.Parallel(n_jobs=-1)(joblib.delayed(classifier.predict)(pixel) for pixel in reshaped_image)
        # Reconstrói os resultados para o formato original da imagem
        mask = np.array(mask).reshape(image.shape[:2])
        np.save(mask_path, mask)
        Create_image_with_segments(mask_path)

        return mask