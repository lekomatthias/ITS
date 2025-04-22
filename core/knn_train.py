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
        self.canvas = None
        self.R = 3

        self.run()

    def load_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if filepath:
            self.image = Image.open(filepath)
            self.image = self.resize_image_to_canvas(self.image)
            self.processed_image = np.array(self.image)
            self.display_image(self.image)

    def resize_image_to_canvas(self, image):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        return image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)

    def display_image(self, image):
        image_tk = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor="nw", image=image_tk)
        self.canvas.image = image_tk

    def on_click(self, event):
        if event.x and event.y:
            x, y = event.x, event.y
            self.points.append((x, y))
            self.canvas.create_oval(x - self.R, y - self.R, x + self.R, y + self.R, outline='red', width=1)

    def on_key(self, event):
        if event.keysym in ['1', '2']:
            label = 0 if event.keysym == '1' else 1
            self.regions.append(self.points[:])
            self.labels.append(label)
            self.points = []
            print(f"Região marcada com label {label}")
        elif event.keysym == '0':
            self.process_image_for_classes()

    def expand_region(self, image, region, radius=3):
        image_np = np.array(image)
        mask = np.zeros(image_np.shape[:2], dtype=bool)
        for x, y in region:
            rr, cc = np.meshgrid(
                np.clip(np.arange(y - radius, y + radius + 1), 0, image_np.shape[0] - 1),
                np.clip(np.arange(x - radius, x + radius + 1), 0, image_np.shape[1] - 1)
            )
            mask[rr, cc] = True
        return mask

    def extract_samples(self, image, regions, labels, radius=3):
        X_train, y_train = [], []
        image_np = np.array(image)
        for region, label in zip(regions, labels):
            mask = self.expand_region(image_np, region, radius)
            pixels = image_np[mask]
            X_train.extend(pixels)
            y_train.extend([label] * len(pixels))
        return np.array(X_train), np.array(y_train)

    def save_knn_model(self, X_train, y_train):
        model_path = filedialog.asksaveasfilename(defaultextension=".joblib", filetypes=[("Joblib files", "*.joblib")])
        if model_path:
            kd_tree = KDTree(X_train, leaf_size=30)
            model_data = {
                "kd_tree": kd_tree,
                "labels": y_train,
                "classes_": np.unique(y_train)
            }
            joblib.dump(model_data, model_path)
            print(f"Modelo salvo em: {model_path}")

    def process_and_train(self):
        if not self.regions or not self.labels:
            print("Nenhuma região foi marcada!")
            return

        X_train, y_train = self.extract_samples(self.image, self.regions, self.labels)

        print(f"Treinando modelo com {len(X_train)} amostras...")
        self.save_knn_model(X_train, y_train)
        self.regions.clear()
        self.labels.clear()
        print("Treinamento concluído. O modelo foi salvo!")

    def process_image_for_classes(self):
        if not self.regions or not self.labels:
            print("Nenhuma região foi marcada para processamento!")
            return

        X_train, y_train = self.extract_samples(self.image, self.regions, self.labels)
        kd_tree = KDTree(X_train, leaf_size=30)
        print("Processando a imagem para manter apenas os pixels de interesse...")

        image_np = np.array(self.processed_image)
        h, w, _ = image_np.shape
        reshaped_image = image_np.reshape(-1, 3)

        dist, ind = kd_tree.query(reshaped_image, k=1)
        pixel_classes = np.array([y_train[idx[0]] for idx in ind])

        new_image = np.zeros_like(image_np)
        interest_mask = (pixel_classes == 0).reshape(h, w)
        non_interest_mask = (pixel_classes == 1).reshape(h, w)

        new_image[interest_mask] = image_np[interest_mask]
        new_image[non_interest_mask] = [255, 255, 255]

        self.processed_image = new_image
        self.display_image(Image.fromarray(self.processed_image))
        print("Processamento concluído: Pixels de não interesse foram apagados.")

    def run(self):
        root = tk.Tk()
        root.title("Treinamento de Classificador de Pixels")

        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()

        btn_load = tk.Button(root, text="Carregar Imagem", command=self.load_image)
        btn_load.pack(pady=10)

        btn_train = tk.Button(root, text="Treinar e Salvar Modelo", command=self.process_and_train)
        btn_train.pack(pady=10)

        lbl_instructions = tk.Label(root, text=(
            "Instruções:\n"
            "1. Clique em 'Carregar Imagem' para abrir uma imagem.\n"
            "2. Clique nos pontos para definir regiões.\n"
            "3. Pressione '1' para marcar regiões de interesse (classe 0).\n"
            "4. Pressione '2' para marcar regiões de não interesse (classe 1).\n"
            "5. Pressione '0' para processar a imagem e remover pixels de não interesse.\n"
            "6. Clique em 'Treinar e Salvar Modelo' para salvar."
        ))
        lbl_instructions.pack(pady=10)

        root.bind('<Button-1>', self.on_click)
        root.bind('<Key>', self.on_key)

        root.mainloop()

if __name__ == "__main__":
    knn_train()
