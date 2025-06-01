from tkinter import filedialog
import numpy as np

from core.PixelClassifierLUT import PixelClassifierLUT as PixelClassifier
from util.timing import timing
from util.Image_manager import Create_image_with_segments


class MaskClassifier:
    @timing
    def Classify(self, image, mask_path):
        """
        Gera uma máscara da imagem aplicando a classificação de pixels.
        """

        print("Selecione o modelo a nível de pixel para aplicação.")
        model_path = filedialog.askopenfilename(
            title="Selecione o modelo para aplicação",
            filetypes=[("joblib", "*.joblib *.npy")]
        )

        if not model_path:
            print("Nenhum modelo selecionado. Encerrando o programa.")
            return

        classifier = PixelClassifier(model_path=model_path)

        print("Criando máscara da imagem...")

        if len(image.shape) > 2:
            mask = classifier.predict_array(image)
        else:
            # Imagem grayscale → replica nos 3 canais
            rgb_image = np.stack([image] * 3, axis=-1)
            mask = classifier.predict_array(rgb_image)

        np.save(mask_path, mask)
        Create_image_with_segments(mask_path)

        return mask
