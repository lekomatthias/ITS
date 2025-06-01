import numpy as np
import joblib


class PixelClassifierLUT:
    def __init__(self, model_path, bins_per_channel=256):
        self.bins = bins_per_channel
        self.step = 256 // bins_per_channel

        if model_path.endswith('.joblib'):
            self.lut_path = model_path.replace('.joblib', '_LUT.npy')
            print(f"Carregando modelo KNN de {model_path}...")
            model_data = joblib.load(model_path)
            kd_tree = model_data["kd_tree"]
            labels = model_data["labels"]
            self.lut = self._generate_LUT(kd_tree, labels)
            np.save(self.lut_path, self.lut)
            print(f"LUT salva em {self.lut_path}.")
        elif model_path.endswith('.npy'):
            self.lut_path = model_path
            print(f"Carregando LUT de {model_path}...")
            self.lut = np.load(self.lut_path)
            print("LUT carregada com sucesso.")
        else:
            raise ValueError("O arquivo deve ser .joblib (modelo) ou .npy (LUT).")

    def _generate_LUT(self, kd_tree, labels):
        grid = np.arange(self.bins) * self.step + self.step // 2
        rr, gg, bb = np.meshgrid(grid, grid, grid, indexing='ij')
        pixels = np.stack([rr.ravel(), gg.ravel(), bb.ravel()], axis=1)

        _, ind = kd_tree.query(pixels, k=3)
        neighbors = labels[ind]
        predicted = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 1, neighbors)

        lut = predicted.reshape((self.bins, self.bins, self.bins))
        return lut

    def predict(self, pixel):
        r, g, b = pixel
        r_bin = min(r // self.step, self.bins - 1)
        g_bin = min(g // self.step, self.bins - 1)
        b_bin = min(b // self.step, self.bins - 1)
        return not self.lut[r_bin, g_bin, b_bin]

    def predict_array(self, image):
        """
        Classificação vetorizada com paralelização, robusta e correta.
        """
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        h, w = image.shape[:2]
        split_size = 512

        # Divide em blocos
        blocks = []
        for i in range(0, h, split_size):
            for j in range(0, w, split_size):
                i_end = min(i + split_size, h)
                j_end = min(j + split_size, w)
                blocks.append(((i, i_end), (j, j_end)))

        def process_block(block):
            (i_start, i_end), (j_start, j_end) = block
            subimage = image[i_start:i_end, j_start:j_end]

            r = np.clip(subimage[..., 0] // self.step, 0, self.bins - 1)
            g = np.clip(subimage[..., 1] // self.step, 0, self.bins - 1)
            b = np.clip(subimage[..., 2] // self.step, 0, self.bins - 1)

            mask = ~(self.lut[r, g, b].astype(bool))
            return (i_start, i_end, j_start, j_end, mask)

        results = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(process_block)(block) for block in blocks
        )

        # Junta os blocos na máscara final
        mask = np.zeros((h, w), dtype=bool)
        for i_start, i_end, j_start, j_end, submask in results:
            mask[i_start:i_end, j_start:j_end] = submask

        return mask
