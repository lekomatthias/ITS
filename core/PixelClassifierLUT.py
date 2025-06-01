import numpy as np
import joblib

class PixelClassifierLUT:
    def __init__(self, model_path, bins_per_channel=64):
        """
        Inicializa o classificador LUT.

        Se input_path for .joblib → gera a LUT e salva como .npy.
        Se input_path for .npy → carrega a LUT existente.

        bins_per_channel: número de divisões por canal (default 16 → LUT de 16x16x16).
        """
        self.bins = bins_per_channel
        self.step = 256 // bins_per_channel

        if model_path.endswith('.joblib'):
            self.model_path = model_path
            self.lut_path = self.model_path.replace('.joblib', '_LUT.npy')

            # Carrega modelo KNN
            print(f"Carregando modelo KNN de {model_path}...")
            self.model_data = joblib.load(self.model_path)
            self.kd_tree = self.model_data["kd_tree"]
            self.labels = self.model_data["labels"]

            # Gera e salva a LUT
            self.generate_LUT()
            print(f"LUT salva em {self.lut_path}")

        elif model_path.endswith('.npy'):
            self.lut_path = model_path
            print(f"Carregando LUT de {model_path}...")
            self.lut = np.load(self.lut_path)
            print("LUT carregada com sucesso.")

        else:
            raise ValueError("O arquivo deve ser .joblib (modelo) ou .npy (LUT).")


    def generate_LUT(self):
        """
        Gera a LUT quantizada a partir do modelo KNN e salva como .npy.
        """
        print(f"Gerando LUT quantizada ({self.bins}x{self.bins}x{self.bins})...")
        self.lut = np.zeros((self.bins, self.bins, self.bins), dtype=np.uint8)

        for r in range(self.bins):
            for g in range(self.bins):
                for b in range(self.bins):
                    r_val = r * self.step + self.step // 2
                    g_val = g * self.step + self.step // 2
                    b_val = b * self.step + self.step // 2

                    # Faz a predição usando o modelo KNN
                    pixel = np.array([r_val, g_val, b_val]).reshape(1, -1)
                    dist, ind = self.kd_tree.query(pixel, k=3)
                    neighbors = self.labels[ind[0]]
                    predicted_label = np.bincount(neighbors).argmax()

                    self.lut[r, g, b] = predicted_label

        np.save(self.lut_path, self.lut)
        print("LUT gerada e salva com sucesso.")


    def predict(self, pixel):
        """
        Prediz a classe de um pixel usando a LUT.

        pixel: tupla ou lista (R, G, B) com valores de 0 a 255.
        """
        r, g, b = pixel
        r_bin = min(r // self.step, self.bins - 1)
        g_bin = min(g // self.step, self.bins - 1)
        b_bin = min(b // self.step, self.bins - 1)
        return not self.lut[r_bin, g_bin, b_bin]
