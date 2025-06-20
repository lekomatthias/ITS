import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog

class SP_grouper:
    def __init__(self, image, root=None):
        """
        Inicializa o rotulador interativo de segmentos.
        """

        if isinstance(image, Image.Image):
            self.image = np.array(image)
        else:
            self.image = image
        
        self.segments = np.load(filedialog.askopenfilename(
                                title="Selecione os segmentos", 
                                filetypes=[("Numpy files", "*.npy")]))
        self.new_labels = np.full_like(self.segments, -1)
        self.processed_image = self._add_white_lines(self.image, self.segments)
        self.selected_segments = set()
        self.current_label = 0

        # Ajusta a escala da imagem para altura fixa
        self.display_image, self.scale_factor = self._resize_image(self.processed_image, target_height=600)

        self.master = root
        if not self.master:
            self.master = tk.Tk()
            self._mainloop = True
        else:
            self._mainloop = False

        self.master.withdraw()
        self.master.deiconify()
        self.master.title("Seleção de Segmentos")

        self.canvas = tk.Canvas(self.master, width=self.display_image.width, height=self.display_image.height)
        self.canvas.pack()

        self.tk_image = ImageTk.PhotoImage(self.display_image)
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        self.button_label = tk.Button(self.master, text="Atribuir Novo Rótulo", command=self.assign_label)
        self.button_label.pack(side=tk.LEFT)

        self.button_finish = tk.Button(self.master, text="Finalizar", command=self.finish)
        self.button_finish.pack(side=tk.RIGHT)

        self.canvas.bind("<Button-1>", self.on_click)

        if self._mainloop:
            self.root.mainloop()

    def _add_white_lines(self, image, segments):
        """
        Adiciona linhas brancas entre os segmentos para melhorar a visualização.
        """
        overlay = image.copy()
        segment_boundaries = np.zeros_like(segments, dtype=bool)

        # Detectar bordas
        segment_boundaries[1:, :] |= (segments[1:, :] != segments[:-1, :])
        segment_boundaries[:, 1:] |= (segments[:, 1:] != segments[:, :-1])

        # Desenhar linhas brancas
        overlay[segment_boundaries] = [255, 255, 255]
        return overlay

    def _resize_image(self, image, target_height):
        """
        Redimensiona a imagem mantendo a proporção, ajustando para altura fixa.
        """
        h, w, _ = image.shape
        scale_factor = target_height / h
        new_width = int(w * scale_factor)
        resized_image = Image.fromarray(image).resize((new_width, target_height), Image.Resampling.LANCZOS)
        return resized_image, scale_factor

    def update_display(self):
        """
        Atualiza a exibição com os segmentos selecionados.
        """
        overlay = self.processed_image.copy()
        for segment_id in self.selected_segments:
            mask = self.segments == segment_id
            overlay[mask] = [255, 0, 0]  # Borda vermelha para segmentos selecionados
        scaled_image = Image.fromarray(overlay).resize(
            (self.display_image.width, self.display_image.height), Image.Resampling.LANCZOS
        )
        self.tk_image = ImageTk.PhotoImage(scaled_image)
        self.canvas.itemconfig(self.image_on_canvas, image=self.tk_image)

    def on_click(self, event):
        """
        Evento de clique para selecionar ou desmarcar segmentos.
        """
        x, y = int(event.x / self.scale_factor), int(event.y / self.scale_factor)
        segment_id = self.segments[y, x]
        if segment_id in self.selected_segments:
            self.selected_segments.remove(segment_id)
        else:
            self.selected_segments.add(segment_id)
        self.update_display()

    def assign_label(self):
        """
        Atribui um novo rótulo aos segmentos selecionados.
        """
        for segment_id in self.selected_segments:
            self.new_labels[self.segments == segment_id] = self.current_label
        self.current_label += 1
        self.selected_segments.clear()
        self.update_display()
        print(f"Novo rótulo atribuído: {self.current_label - 1}")

    def finish(self):
        """
        Finaliza a interação e atribui rótulos únicos aos segmentos restantes.
        """
        unassigned_mask = self.new_labels == -1  # Posições ainda não atribuídas
        unique_segments = np.unique(self.segments[unassigned_mask])

        for i, segment_id in enumerate(unique_segments, start=self.current_label):
            self.new_labels[self.segments == segment_id] = i

        print("Todos os rótulos foram atribuídos.")
        self.master.quit()

    def run(self):
        """
        Inicia o loop principal.
        """
        self.master.mainloop()
        self.master.withdraw()
        return self.new_labels


if __name__ == "__main__":
    
    root = tk.Tk()
    root.withdraw()

    # Carrega a imagem e os segmentos de exemplo
    image_path = filedialog.askopenfilename()
    image = Image.open(image_path)
    segments = np.load(filedialog.askopenfilename())

    # Inicializa o rotulador com a janela existente
    labeler = SP_grouper(image, segments, root=root)

    # Executa o rotulador e captura os segmentos rotulados
    labeled_segments = labeler.run()

    # Exibe os resultados
    print("Segmentos rotulados:", np.unique(labeled_segments))
