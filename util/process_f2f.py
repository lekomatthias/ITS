
import os
import cv2
from tkinter import filedialog

def Process_f2f(process_func, save_func, type_in="csv", type_out="npy"):
    '''
    Função feita para processar imagens de uma pasta para outra.
    '''

    folder_path = filedialog.askdirectory(title="Selecione a pasta para aplicação de entrada")
    base_path = os.path.dirname(folder_path)
    output_dir = os.path.join(base_path, f"{os.path.basename(folder_path)}_{type_in}_{type_out}")
    os.makedirs(output_dir, exist_ok=True)

    images_path = []
    for file in os.listdir(folder_path):
        if file.lower().endswith((f".{type_in}")):
            images_path.append(os.path.join(folder_path, file))

    for path in images_path:
        img = False
        try:
            img = process_func(path)
        except Exception as e:
            print(f"Erro ao processar o arquivo: {path}\n -- {e}")
            continue
        new_path = os.path.join(output_dir, os.path.basename(path).replace(type_in, type_out))

        if type_out == "jpg":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        save_func(new_path, img)
        print(f"arquivo salvo em: {new_path}")


if __name__ == "__main__":
    from skimage.io import imsave
    from numpy import save as np_save
    from CSV2JPG import CSV2JPG
    from CSV2segments import CSV2segments
    from JPG2segments import JPG2segments
    from segments2JPG import segments2JPG

    Process_f2f(CSV2segments, np_save)
    # Process_f2f(CSV2JPG, imsave, type_in="csv", type_out="jpg")
    # Process_f2f(JPG2segments, np_save, type_in="jpg", type_out="npy") # precisa de melhoria
    # Process_f2f(segments2JPG, imsave, type_in="npy", type_out="jpg")
