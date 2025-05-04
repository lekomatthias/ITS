
import numpy as np
import os
import cv2
from skimage.io import imsave
from tkinter import filedialog
import matplotlib.pyplot as plt

def generate_contrasting_colors(num_colors):
        """Gera cores contrastantes para diferentes classes."""
        # Garantir que há cores suficientes
        cmap = plt.get_cmap('tab20')
        colors = [np.array(cmap(i % 20)[:3]) * 255 for i in range(num_colors)]
        colors[0] = np.array([0, 0, 0])
        return colors

def Paint_image(image, segments):
    """
    Gerar imagem colorida para visualização
    """

    colors = generate_contrasting_colors(len(np.unique(segments)))
    color_image = np.zeros(image.shape)
    for segment_value, color in zip(np.unique(segments), colors):
        color_image[segments == segment_value] = color
    output_image = image * 0.3 + color_image * 0.7
    return output_image

def Create_image_with_segments(seg_path=None):
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
    seg_name = seg_name.replace(".npy", "")

    segments = np.load(seg_path)
    output_image = np.zeros((*segments.shape, 3), dtype=np.uint8)
    if len(np.unique(segments)) == 2: colors = [(0, 0, 0), (255, 255, 255)]
    else: colors = generate_contrasting_colors(len(np.unique(segments)))
    segments = segments.astype(int)
    for segment_value, color in zip(np.unique(segments), colors):
        output_image[segments == segment_value] = color
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    imsave(os.path.join(seg_dir, f"{seg_name}_color.jpeg"), output_image)
    print(f"Imagem com segmentos coloridos salva em: {os.path.join(seg_dir, f'{seg_name}_color.jpeg')}")
    return output_image