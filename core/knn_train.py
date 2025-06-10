import tkinter as tk
import numpy as np
import joblib
from tkinter import filedialog
from PIL import Image, ImageTk
from sklearn.neighbors import KDTree

class knn_train:
    def __init__(self):
        self.points = []
        self.regions = []
        self.labels = []
        self.image = None
        self.processed_image = None
        self.R = 3

    def run(self, root=None):
        self.root = root
        if not self.root:
            self.root = tk.Tk()
            self._mainloop = True
        else:
            self._mainloop = False

        self.root.title("Treinamento de Classificador de Pixels")
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill='both', expand=True)
        
        self.canvas = tk.Canvas(main_frame, width=800, height=600, bg='white')
        self.canvas.pack(side='left', padx=10, pady=10)

        button_frame = tk.Frame(main_frame)
        button_frame.pack(side='right', fill='y', padx=10, pady=10)

        tk.Button(button_frame, text="Carregar Imagem", command=self.load_image).pack(pady=10, fill='x')
        tk.Button(button_frame, text="Treinar e Salvar Modelo", command=self.process_and_train).pack(pady=10, fill='x')

        print(
            "Instruções:\n"
            "1. Clique em 'Carregar Imagem' para abrir uma imagem.\n"
            "2. Clique nos pontos para definir regiões.\n"
            "3. Pressione '1' para marcar regiões de interesse (classe 0).\n"
            "4. Pressione '2' para marcar regiões de não interesse (classe 1).\n"
            "5. Pressione '0' para processar a imagem e remover pixels de não interesse.\n"
            "6. Clique em 'Treinar e Salvar Modelo' para salvar."
        )

        self.root.bind('<Button-1>', self.on_click)
        self.root.bind('<Key>', self.on_key)

        if self._mainloop:
            self.root.mainloop()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            img = Image.open(path)
            img = img.resize((800, 600), Image.Resampling.LANCZOS)
            self.image = img
            self.processed_image = np.array(img)
            self.display_image(img)

    def display_image(self, img):
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

    def on_click(self, event):
        self.points.append((event.x, event.y))
        r = self.R
        self.canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, outline='red')

    def on_key(self, event):
        if event.keysym in ['1', '2']:
            label = 0 if event.keysym == '1' else 1
            if self.points:
                self.regions.append(self.points[:])
                self.labels.append(label)
                self.points.clear()
                print(f"Região marcada com label {label}")
        elif event.keysym == '0':
            self.process_image_for_classes()

    def expand_region(self, image_np, region, radius=3):
        mask = np.zeros(image_np.shape[:2], dtype=bool)
        for x, y in region:
            y_range = np.clip(np.arange(y - radius, y + radius + 1), 0, image_np.shape[0] - 1)
            x_range = np.clip(np.arange(x - radius, x + radius + 1), 0, image_np.shape[1] - 1)
            rr, cc = np.meshgrid(y_range, x_range, indexing='ij')
            mask[rr, cc] = True
        return mask

    def extract_samples(self, image, regions, labels, radius=3):
        X, y = [], []
        image_np = np.array(image)
        for region, label in zip(regions, labels):
            mask = self.expand_region(image_np, region, radius)
            pixels = image_np[mask]
            X.extend(pixels)
            y.extend([label] * len(pixels))
        return np.array(X), np.array(y)

    def save_knn_model(self, X, y):
        path = filedialog.asksaveasfilename(defaultextension=".joblib", filetypes=[("Joblib", "*.joblib")])
        if path:
            kd_tree = KDTree(X, leaf_size=30)
            joblib.dump({"kd_tree": kd_tree, "labels": y, "classes_": np.unique(y)}, path)
            print(f"Modelo salvo em: {path}")

    def process_and_train(self):
        if not self.regions:
            print("Nenhuma região foi marcada!")
            return
        X, y = self.extract_samples(self.image, self.regions, self.labels)
        print(f"Treinando modelo com {len(X)} amostras...")
        self.save_knn_model(X, y)
        self.regions.clear()
        self.labels.clear()
        print("Treinamento concluído. Modelo salvo!")

    def process_image_for_classes(self):
        if not self.regions:
            print("Nenhuma região foi marcada para processamento!")
            return
        X, y = self.extract_samples(self.image, self.regions, self.labels)
        kd_tree = KDTree(X, leaf_size=30)
        print("Processando imagem para manter pixels de interesse...")

        img_np = self.processed_image
        h, w = img_np.shape[:2]
        reshaped = img_np.reshape(-1, 3)

        dist, ind = kd_tree.query(reshaped, k=1)
        pixel_classes = y[ind[:, 0]]

        new_img = np.full_like(img_np, 255)
        mask = (pixel_classes == 0).reshape(h, w)
        new_img[mask] = img_np[mask]

        self.processed_image = new_img
        self.display_image(Image.fromarray(new_img))
        print("Processamento concluído: Pixels de não interesse apagados.")


if __name__ == "__main__":
    knn_train().run()
