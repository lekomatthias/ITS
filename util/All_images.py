import os
from tkinter.filedialog import askdirectory

from util.timing import timing

def All_images(func):
    @timing
    def all_images_of_directory(mode):
        folder = askdirectory(title="Select Image Folder")
        if not folder:
            print("No folder selected.")
            return
        i = 0
        for file in os.listdir(folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                i =+ 1
                print(f"arquivo número: {i}\n=====> Executando para o arquivo: {file}")
                path = os.path.join(folder, file)
                func(image_path=path, mode=mode)
    return all_images_of_directory
