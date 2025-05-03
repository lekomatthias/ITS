
import os
import cv2
from tkinter import filedialog

def Process_f2f(process_func, save_func, type_in="csv", type_out="npy"):
    '''
    Função feita para processar imagens de uma pasta para outra.
    '''

    folder_path = filedialog.askdirectory(title="Selecione a pasta para aplicação de entrada")
    base_path = os.path.dirname(folder_path)
    output_dir = os.path.join(base_path, f"processadas_{type_in}_{type_out}")
    os.makedirs(output_dir, exist_ok=True)

    images_path = []
    for file in os.listdir(folder_path):
        if (type_in == "jpg" or type_in == "jpeg") and file.lower().endswith(('.jpeg', '.jpg')):
            images_path.append(os.path.join(folder_path, file))
        if type_in == "png" and file.lower().endswith(('.png')):
            images_path.append(os.path.join(folder_path, file))
        if type_in == "csv" and file.lower().endswith(('.csv')):
            images_path.append(os.path.join(folder_path, file))

    for path in images_path:
        img = False
        try:
            img = process_func(path)
        except Exception as e:
            print(f"Erro ao processar o arquivo: {path}, erro : {e}")
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

    # Process_f2f(CSV2segments, np_save)
    Process_f2f(CSV2JPG, imsave, type_in="csv", type_out="jpg")
