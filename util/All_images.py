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
        
        for file in os.listdir(folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                print(f"\n=====> Executando para o arquivo: {file}")
                path = os.path.join(folder, file)
                func(image_path=path, mode=mode)
    return all_images_of_directory
