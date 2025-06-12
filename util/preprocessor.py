import numpy as np
import os
from PIL import Image

# load image
def load_image(image_path):
    return np.array(Image.open(image_path))

# load mask em formato numpy array
def load_mask(mask_path):
    return np.load(mask_path)

# multiplica as matrizes imagem e márcara
def multiply_image_and_mask(image, mask):
    return image * mask[:, :, np.newaxis]

# salva a imagem a partir de um numpy arrray
def save_image(image, path):
    image = Image.fromarray(image.astype(np.uint8))
    image.save(path)

# baseado no path da imagem, encontra a márcara e retorna o path dela
# o caminho é caminho da imagem/mascaras/mask_<nome da imagem>.npy
def find_mask_path(image_path):
    base_path = os.path.dirname(image_path)
    mask_folder = os.path.join(base_path, "mascaras")
    file_name = os.path.basename(image_path)
    name, ext = os.path.splitext(file_name)
    mask_name = f"mask_{name}.npy"
    mask_path = os.path.join(mask_folder, mask_name)
    
    return mask_path

def get_maskared_image(image_path):
    image = load_image(image_path)
    mask = load_mask(find_mask_path(image_path))
    return multiply_image_and_mask(image, mask)

def PreProcessor(dir_path, folder_path="pre_processadas"):
    base_path = dir_path
    output_dir = os.path.join(base_path, folder_path)
    os.makedirs(output_dir, exist_ok=True)

    images_path = []
    for file in os.listdir(base_path):
        if file.lower().endswith(('.jpeg', '.jpg', '.png')):
            images_path.append(os.path.join(base_path, file))

    for path in images_path:
        try:
            img = get_maskared_image(path)
        except:
            continue
        new_path = os.path.join(output_dir, os.path.basename(path))
        save_image(img, new_path)
        print(f"imagem salva em: {new_path}")

if __name__ == "__main__":
    from tkinter import filedialog
    import matplotlib.pyplot as plt

    path = filedialog.askdirectory(title="Selecione a pasta para aplicação de entrada")

    PreProcessor(path)
