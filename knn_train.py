import tkinter as tk
import numpy as np
import joblib
from tkinter import filedialog
from PIL import Image, ImageTk
from sklearn.neighbors import KDTree

# Variáveis globais
points = []
regions = []
labels = []
image = None
processed_image = None
canvas = None
R = 3

def load_image():
    global image, processed_image, canvas
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if filepath:
        image = Image.open(filepath)
        image = resize_image_to_canvas(image)
        processed_image = np.array(image)
        display_image(image)

def resize_image_to_canvas(image):
    global canvas
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    return image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)

def display_image(image):
    global canvas
    image_tk = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor="nw", image=image_tk)
    canvas.image = image_tk

def on_click(event):
    global points
    if event.x and event.y:
        x, y = event.x, event.y
        points.append((x, y))
        canvas.create_oval(x-R, y-R, x+R, y+R, outline='red', width=1)

def on_key(event):
    global regions, labels, points
    # essa parte está assim para facilitar pro usuário...
    if event.keysym in ['1', '2']:  # '1' para interesse (0), '2' para não interesse (1)
        label = 0 if event.keysym == '1' else 1  # 1 -> 0, 2 -> 1
        regions.append(points[:])
        labels.append(label)
        points = []
        print(f"Região marcada com label {label}")
    elif event.keysym == '0':
        process_image_for_classes()

def expand_region(image, region, radius=R):
    image_np = np.array(image)
    mask = np.zeros(image_np.shape[:2], dtype=bool)
    for x, y in region:
        rr, cc = np.meshgrid(
            np.clip(np.arange(y - radius, y + radius + 1), 0, image_np.shape[0] - 1),
            np.clip(np.arange(x - radius, x + radius + 1), 0, image_np.shape[1] - 1)
        )
        mask[rr, cc] = True
    return mask

def extract_samples(image, regions, labels, radius=R):
    X_train, y_train = [], []
    image_np = np.array(image)
    for region, label in zip(regions, labels):
        mask = expand_region(image_np, region, radius)
        pixels = image_np[mask]
        X_train.extend(pixels)
        y_train.extend([label] * len(pixels))
    return np.array(X_train), np.array(y_train)

def save_knn_model(X_train, y_train):
    model_path = filedialog.asksaveasfilename(defaultextension=".joblib", filetypes=[("Joblib files", "*.joblib")])
    if model_path:
        # Criação do KDTree
        kd_tree = KDTree(X_train, leaf_size=30)
        model_data = {
            "kd_tree": kd_tree,
            "labels": y_train,
            "classes_": np.unique(y_train)
        }
        joblib.dump(model_data, model_path)
        print(f"Modelo salvo em: {model_path}")

def process_and_train():
    global image, regions, labels
    if not regions or not labels:
        print("Nenhuma região foi marcada!")
        return

    # Extrair dados de treinamento
    X_train, y_train = extract_samples(image, regions, labels)

    print(f"Treinando modelo com {len(X_train)} amostras...")
    save_knn_model(X_train, y_train)
    regions.clear()
    labels.clear()
    print("Treinamento concluído. O modelo foi salvo!")

def process_image_for_classes():
    global processed_image, regions, labels
    if not regions or not labels:
        print("Nenhuma região foi marcada para processamento!")
        return

    # Extrair dados de treinamento
    X_train, y_train = extract_samples(image, regions, labels)

    # Treina o KDTree com base nos rótulos existentes
    kd_tree = KDTree(X_train, leaf_size=30)
    print("Processando a imagem para manter apenas os pixels de interesse...")

    # Para cada pixel da imagem
    image_np = np.array(processed_image)
    h, w, _ = image_np.shape
    reshaped_image = image_np.reshape(-1, 3)

    # Prediz as classes para cada pixel
    dist, ind = kd_tree.query(reshaped_image, k=1)
    pixel_classes = np.array([y_train[idx[0]] for idx in ind])

    # Criar a nova imagem baseada na classificação
    new_image = np.zeros_like(image_np)
    interest_mask = (pixel_classes == 0).reshape(h, w)
    non_interest_mask = (pixel_classes == 1).reshape(h, w)

    # Mantém os pixels de interesse, apaga os demais
    new_image[interest_mask] = image_np[interest_mask]
    # apaga da imagem o que foi classificado como não interesse
    new_image[non_interest_mask] = [255, 255, 255]

    # Atualiza a imagem processada e exibe no canvas
    processed_image = new_image
    display_image(Image.fromarray(processed_image))
    print("Processamento concluído: Pixels de não interesse foram apagados.")

def main():
    global canvas
    root = tk.Tk()
    root.title("Treinamento de Classificador de Pixels")

    canvas = tk.Canvas(root, width=800, height=600)
    canvas.pack()

    btn_load = tk.Button(root, text="Carregar Imagem", command=load_image)
    btn_load.pack(pady=10)

    btn_train = tk.Button(root, text="Treinar e Salvar Modelo", command=process_and_train)
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

    # Eventos de clique e tecla
    root.bind('<Button-1>', on_click)
    root.bind('<Key>', on_key)

    root.mainloop()

if __name__ == "__main__":
    main()
