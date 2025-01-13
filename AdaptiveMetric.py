import numpy as np
import matplotlib.pyplot as plt

class AdaptiveMetric:
    def __init__(self):
        self.M = None

    def save_metric(self, file_path):
        if self.M is None:
            raise ValueError("A matriz M precisa ser calculada antes de salvar.")
        np.save(file_path, self.M)  # Salva no formato .npy!!!

    def load_metric(self, file_path):
        self.M = np.load(file_path)  # Carrega do formato .npy!!!


    def extract_superpixels(self, image, segments):
        """
        Extrai as características dos superpixels de uma imagem e retorna uma lista de listas [[x, y], [r, g, b]].
        """
        unique_segments = np.unique(segments)
        superpixels = []

        for label in unique_segments:
            mask = segments == label
            coords = np.argwhere(mask)
            if coords.size == 0: continue
            x, y = np.mean(coords, axis=0)
            x /= segments.shape[1]
            y /= segments.shape[0]

            region = image[mask]
            if region.size == 0: continue
            r, g, b = np.mean(region, axis=0) / 255.0

            superpixels.append([[x, y], [r, g, b]])

        return superpixels, unique_segments


    def compute_metric_matrix(self, superpixels, labels):
        """
        Calcula a matriz métrica adaptativa M e armazena em self.M.
        """
        positions = np.array([sp[0] for sp in superpixels])
        colors = np.array([sp[1] for sp in superpixels])
        n, d_pos = positions.shape
        _, d_col = colors.shape

        # Calcular distâncias euclidianas
        pos_distances = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)
        col_distances = np.linalg.norm(colors[:, np.newaxis] - colors, axis=2)

        # Ajuste de labels para que sejam contínuos
        unique_labels, new_labels = np.unique(labels, return_inverse=True)
        K = len(unique_labels)
        
        # Criar matriz de designação Y
        Y = np.zeros((n, K))
        for i, label in enumerate(new_labels):
            Y[i, label] = 1

        # Calcular centros dos clusters
        cluster_centers_pos = np.zeros((K, d_pos))
        cluster_centers_col = np.zeros((K, d_col))
        for k in range(K):
            indices = np.where(labels == k)[0]
            if len(indices) == 0: continue
            cluster_centers_pos[k, :] = np.mean(positions[indices], axis=0)
            cluster_centers_col[k, :] = np.mean(colors[indices], axis=0)

        # Resolver para M
        X_pos_pinv = np.linalg.pinv(pos_distances)
        X_col_pinv = np.linalg.pinv(col_distances)
        Y_pinv = np.linalg.pinv(Y)
        M_pos = X_pos_pinv @ Y @ Y_pinv @ X_pos_pinv.T
        M_col = X_col_pinv @ Y @ Y_pinv @ X_col_pinv.T

        # Ajustar as dimensões das matrizes para combinar corretamente
        M_pos = M_pos[:d_pos, :d_pos]
        M_col = M_col[:d_col, :d_col]

        # Combinar as matrizes de posição e cor
        self.M = np.block([
            [M_pos, np.zeros((d_pos, d_col))],
            [np.zeros((d_col, d_pos)), M_col]
        ])

    def update_metric_matrix(self, new_superpixels, new_labels, alpha=0.8):
        """
        Atualiza a matriz métrica existente com base em novos dados.
        """
        if self.M is None:
            # Se ainda não houver uma métrica, apenas calcule a partir dos novos dados.
            self.compute_metric_matrix(new_superpixels, new_labels)
            return

        M_old = self.M.copy()
        self.compute_metric_matrix(new_superpixels, new_labels)

        # Combinar a métrica antiga com a nova
        self.M = alpha * M_old + (1 - alpha) * self.M


    def mahalanobis_distance(self, sp1, sp2):
        """
        Calcula a distância de Mahalanobis entre dois superpixels e a normaliza.
        """
        if self.M is None:
            raise ValueError("A matriz M precisa ser calculada antes de calcular a distância de Mahalanobis.")

        diff_pos = np.array(sp1[0]) - np.array(sp2[0])
        diff_col = np.array(sp1[1]) - np.array(sp2[1])
        diff = np.concatenate((diff_pos, diff_col))
        dist = np.sqrt(diff.T @ self.M @ diff)
        
        # Normalizar a distância pela raiz quadrada da dimensionalidade dos dados
        d = len(diff)
        normalized_dist = dist / np.sqrt(d)
        
        return normalized_dist

    def cluster_with_metric(self, superpixels, threshold):
        """
        Aplica a métrica para agrupar superpixels.
        """

        n, d = superpixels.shape

        # Garantir que M seja simétrica e positiva semi-definida
        if not np.allclose(self.M, self.M.T):
            raise ValueError("A métrica M deve ser simétrica.")

        # Inicializa os rótulos com -1 (não rotulados)
        labels = np.full(n, -1, dtype=int)
        current_label = 0

        for i in range(n):
            # Se o superpixel ainda não foi rotulado
            if labels[i] == -1:
                labels[i] = current_label
                for j in range(i + 1, n):
                    diff = superpixels[i] - superpixels[j]
                    distance = np.sqrt(diff.T @ self.M @ diff)
                    if distance <= threshold:
                        labels[j] = current_label
                current_label += 1

        return labels

    def data(self, list):
        """
        Recebe uma lista e mostra histograma, média e desvio padrão.
        """

        # Calcular média e desvio padrão
        mean_distance = np.mean(list)
        std_distance = np.std(list)
        print(f"Média: {mean_distance:.2f}")
        print(f"Desvio padrão: {std_distance:.2f}")

        # Criar histograma
        plt.hist(list, bins=30, edgecolor='black')
        plt.title('Histograma dos dados coletados')
        plt.xlabel('Valor')
        plt.ylabel('Frequência')
        plt.axvline(mean_distance, color='r', linestyle='dashed', linewidth=1, label=f'Média: {mean_distance:.2f}')
        plt.axvline(mean_distance + std_distance, color='g', linestyle='dashed', linewidth=1, label=f'+1 Desvio Padrão: {mean_distance + std_distance:.2f}')
        plt.axvline(mean_distance - std_distance, color='g', linestyle='dashed', linewidth=1, label=f'-1 Desvio Padrão: {mean_distance - std_distance:.2f}')
        plt.legend()
        plt.show()

    def merge_similar_segments(self, segments, superpixels, labels, threshold):
        """
        Mescla superpixels semelhantes com base na distância de Mahalanobis.
        """
        
        unique_labels = np.unique(labels)
        label_map = {label: label for label in unique_labels}

        distances = []
        switches = 0
        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i + 1:]:
                sp1_list = [sp for sp, lbl in zip(superpixels, labels) if lbl == label1]
                sp2_list = [sp for sp, lbl in zip(superpixels, labels) if lbl == label2]

                if len(sp1_list) == 0 or len(sp2_list) == 0:
                    continue

                sp1_pos = np.mean([sp[0] for sp in sp1_list], axis=0)
                sp1_col = np.mean([sp[1] for sp in sp1_list], axis=0)
                sp2_pos = np.mean([sp[0] for sp in sp2_list], axis=0)
                sp2_col = np.mean([sp[1] for sp in sp2_list], axis=0)

                dist = self.mahalanobis_distance([sp1_pos, sp1_col], [sp2_pos, sp2_col])
                distances.append(dist)
                if dist < threshold:
                    switches += 1
                    # Unifica transitivamente os labels
                    root1 = label_map[label1]
                    root2 = label_map[label2]
                    new_root = min(int(root1), int(root2))
                    for key, value in label_map.items():
                        if value == root1 or value == root2:
                            label_map[key] = new_root
        
        self.data(distances)

        # Atualizar os segmentos para refletir os novos labels
        updated_segments = segments.copy()

        for index, new_label in label_map.items():
            updated_segments[segments == index] = new_label

        print(f"Antes: {len(np.unique(segments))} segmentos")
        print(f"Depois: {len(np.unique(updated_segments))} segmentos (com {switches} trocas)")

        return updated_segments
    
    def process_image_segments(self, image, segments, threshold=0.1):
        """
        Processa a imagem e os segmentos para agrupar superpixels semelhantes com o mesmo label.
        """
        # Extrair superpixels e labels
        superpixels, labels = self.extract_superpixels(image, segments)
        # Calcular matriz métrica adaptativa
        self.update_metric_matrix(superpixels, labels, alpha=0.8)
        # Mesclar segmentos semelhantes
        updated_segments = self.merge_similar_segments(segments, superpixels, labels, threshold)

        return updated_segments
    
    def train(self, image, segments):
        """
        Treina o modelo de métrica adaptativa com base nos segmentos fornecidos.
        """
        superpixels, labels = self.extract_superpixels(image, segments)
        self.update_metric_matrix(superpixels, labels)

    def classify_image(self, image, segments, threshold=0.1):
        """
        Classifica a imagem com base nos segmentos fornecidos.
        """
        updated_segments = self.process_image_segments(image, segments, threshold)

        return updated_segments
