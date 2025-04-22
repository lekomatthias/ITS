import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk
import cv2

from util.timing import timing

class Shape_selector:
    def __init__(self, root, image, target_height=600):
        self.root = root
        self.shape = "circle"
        self.size = 20
        self.pixels_of_area = 100

        # Converte imagem BGR (OpenCV) para RGB (PIL) e depois cria Image
        self.image = Image.fromarray(image)
        self.scale_image(target_height)
        
        self.canvas = Canvas(root, width=self.image.width, height=self.image.height)
        self.canvas.pack()
        
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
        self.canvas.bind("<Motion>", self.update_overlay)
        self.overlay = None
        self.create_buttons()
    
    def scale_image(self, target_height):
        w_percent = target_height / float(self.image.height)
        target_width = int(float(self.image.width) * w_percent)
        self.image = self.image.resize((target_width, target_height), Image.LANCZOS)
    
    def create_buttons(self):
        tk.Button(self.root, text="+", command=self.increase_size).pack(side=tk.LEFT)
        tk.Button(self.root, text="-", command=self.decrease_size).pack(side=tk.LEFT)
        tk.Button(self.root, text="Alternar", command=self.toggle_shape).pack(side=tk.LEFT)
        tk.Button(self.root, text="Finalizar", command=self.finish).pack(side=tk.LEFT)
    
    def increase_size(self):
        self.size += 2
    
    def decrease_size(self):
        if self.size > 2:
            self.size -= 2
    
    def toggle_shape(self):
        self.shape = "square" if self.shape == "circle" else "circle"
    
    def update_overlay(self, event):
        self.canvas.delete("overlay")
        x, y = event.x, event.y
        
        if self.shape == "circle":
            self.canvas.create_oval(
                x - self.size//2, y - self.size//2,
                x + self.size//2, y + self.size//2,
                outline="red", width=1, tags="overlay"
            )
        else:
            self.canvas.create_rectangle(
                x - self.size//2, y - self.size//2,
                x + self.size//2, y + self.size//2,
                outline="red", width=1, tags="overlay"
            )
    
    def finish(self):
        area = (self.size ** 2) if self.shape == "square" else (3.1416 * (self.size / 2) ** 2)
        total_pixels = self.image.width * self.image.height
        ratio = total_pixels // area if area > 0 else 0
        self.pixels_of_area = area
        self.root.quit()

    def get_pixels(self):
        return self.pixels_of_area

@timing
def GetPixelsOfArea(image, target_height=600):
    """Seleciona uma área da imagem e retorna o número de pixels."""
    root = tk.Tk()
    root.title("Seletor de tamanho de árvore")
    app = Shape_selector(root, image, target_height)
    root.mainloop()
    app.finish()
    root.destroy()
    return app.get_pixels()

# EXEMPLO DE USO
if __name__ == "__main__":
    from tkinter import filedialog
    import sys

    path = filedialog.askopenfilename(title="Selecione uma imagem")
    if path:
        img = cv2.imread(path)
        if img is None:
            print("Erro ao carregar a imagem!")
            sys.exit(1)

        pixels = GetPixelsOfArea(img)
        print(f"Pixels da área selecionada: {pixels}")

