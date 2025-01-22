import numpy as np
import os
import shutil
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from PIL import Image

class Segment_cutting:
    def __init__(self):
        self.image = None
        self.segments = None
        self.save_path = None

    def Load_image(self, image_path):
        if image_path.endswith('.npy'):
            self.image = np.load(image_path)
        else:
            self.image = np.array(Image.open(image_path))
        # Cria a pasta para salvar os segmentos separados na mesma pasta da imagem
        base_path = os.path.dirname(image_path)
        segmentos_separados_path = os.path.join(base_path, "segmentos_separados")
        if os.path.exists(segmentos_separados_path):
            shutil.rmtree(segmentos_separados_path)
        os.makedirs(segmentos_separados_path, exist_ok=True)
        self.Set_save_path(segmentos_separados_path)
        
    def Load_segments(self, segments_path):
        self.segments = np.load(segments_path)

    def Set_save_path(self, save_path):
        self.save_path = save_path

    def Save_cut(self, cut, name):
        if self.save_path is None:
            raise ValueError("O caminho para salvar o corte não foi definido.")
        save_file_path = os.path.join(self.save_path, f"sp_{name:03d}.jpg")
        Image.fromarray(cut).save(save_file_path)

    def Separate_segments(self, image, segments):
        """
        Separa cada segmento em uma imagem isolada.
        """

        for i, segment_id in enumerate(np.unique(segments)):
            # Ignora o segmento de fundo
            if segment_id == -1:
                continue
            mask = segments == segment_id
            coords = np.argwhere(mask)
            min_row, min_col = coords.min(axis=0)
            max_row, max_col = coords.max(axis=0)
            cropped_superpixel = image[min_row:max_row+1, min_col:max_col+1].copy()

            isolated_superpixel = np.zeros_like(cropped_superpixel)
            local_mask = mask[min_row:max_row+1, min_col:max_col+1]
            isolated_superpixel[local_mask] = cropped_superpixel[local_mask]
            self.Save_cut(isolated_superpixel, i)

    def Run(self):
        """
        Faz a separação de segmentos para imagens separadas.
        """

        root = Tk()
        root.withdraw()
        image_path = askopenfilename(title="Selecione a imagem", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;")])
        if not image_path:
            print("Nenhuma imagem selecionada.")
            return
        segments_path = askopenfilename(title="Selecione os segmentos (.npy)", filetypes=[("Numpy files", "*.npy")])
        if not segments_path:
            print("Nenhum segmento selecionado.")
            return
        root.destroy()
        self.Load_image(image_path)
        self.Load_segments(segments_path)
        self.Separate_segments(self.image, self.segments)

if __name__ == "__main__":
    segment_cutting = Segment_cutting()
    segment_cutting.Run()