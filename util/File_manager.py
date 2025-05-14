
import os
import cv2
import numpy as np
from skimage.io import imsave
from tkinter import filedialog

from util.timing import timing

def create_folders(root_path, folder_names=["metricas", "mascaras", "segmentos", "classificadas"]):
    """
    Cria várias pastas dentro de um diretório base.
    """
    root_path = os.path.normpath(root_path)

    for name in folder_names:
        folder_path = os.path.join(root_path, name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

# @timing
def Load_Image(apply_image_path=None):
    """
    Carrega a imagem para aplicação.
    """
    try:
        if apply_image_path is None:
            apply_image_path = filedialog.askopenfilename(
                title="Selecione a imagem para aplicação",
                filetypes=[("Imagens", "*.jpeg;*.jpg;*.png;*.JPEG;*.JPG;*.PNG")]
            )
        if not apply_image_path:
            raise FileNotFoundError("Imagem não selecionada.")

        apply_image_path = os.path.normpath(apply_image_path)
        if not os.path.isfile(apply_image_path):
            # Tenta encontrar o mesmo nome com extensão diferente
            directory, filename = os.path.split(apply_image_path)
            name_no_ext, _ = os.path.splitext(filename)
            possible_exts = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            found = False
            for ext in possible_exts:
                candidate = os.path.join(directory, name_no_ext + ext)
                if os.path.isfile(candidate):
                    apply_image_path = candidate
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"Arquivo não encontrado: {apply_image_path}")

        # Verifica extensão
        valid_extensions = ('.png', '.jpg', '.jpeg')
        if not apply_image_path.lower().endswith(valid_extensions):
            raise ValueError("Extensão inválida para imagem.")

    except (FileNotFoundError, ValueError) as e:
        print(f"{e}\nEncerrando o programa.")
        exit()
    except Exception as e:
        print(f"[ERRO INESPERADO] {e}\nEncerrando o programa.")
        exit()

    apply_image_dir, apply_image_name = os.path.split(apply_image_path)
    create_folders(apply_image_dir)
    apply_image_name_no_ext, _ = os.path.splitext(apply_image_name)
    image = cv2.imread(apply_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, apply_image_dir, apply_image_name_no_ext
            
def Save_image(output_image, apply_image_dir, apply_image_name_no_ext, num_segments, type="classified"):
    """
    Salva a imagem.
    """

    output_image = np.clip(output_image, 0, 255)
    output_image = output_image.astype(np.uint8)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    output_image_path = os.path.join(apply_image_dir, "classificadas",
                                        f"({type}){apply_image_name_no_ext}_{num_segments}.jpeg")
    imsave(output_image_path, output_image)
    print(f"Imagem classificada salva em: {output_image_path}")