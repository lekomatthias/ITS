import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from time import time

class AdaptiveMetric:
    def __init__(self):
        self.M = None

    def save_metric(self, file_path):
        if self.M is None:
            raise ValueError("A matriz M precisa ser calculada antes de salvar.")
        np.save(file_path, self.M)  # Salva no formato .npy!

    def load_metric(self, file_path):
        self.M = np.load(file_path)  # Carrega do formato .npy!

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
        Calcula a matriz métrica adaptativa M no espaço reduzido (2D) e armazena em self.M.
        """
        # Reduzir superpixels para 2 dimensões
        distances = []
        for i in range(len(superpixels)):
            for j in range(i + 1, len(superpixels)):
                diff_pos = np.linalg.norm(np.array(superpixels[i][0]) - np.array(superpixels[j][0]))
                diff_col = np.linalg.norm(np.array(superpixels[i][1]) - np.array(superpixels[j][1]))
                distances.append([diff_pos, diff_col])
        
        # Transformar lista em matriz
        X = np.array(distances)
        n, d = X.shape

        # Ajuste de labels para que sejam contínuos
        unique_labels, new_labels = np.unique(labels, return_inverse=True)
        K = len(unique_labels)

        # Criar matriz de designação Y
        Y = np.zeros((n, K))
        for i, label in enumerate(new_labels):
            Y[i, label] = 1

        # Calcular centros dos clusters
        Z = np.zeros((K, d))
        for k in range(K):
            indices = np.where(new_labels == k)[0]
            if len(indices) == 0:
                continue
            Z[k, :] = np.mean(X[indices], axis=0)

        # Resolver para M1
        X_pinv = np.linalg.pinv(X)
        Y_pinv = np.linalg.pinv(Y)
        M1 = X_pinv @ Y @ Y_pinv @ X_pinv.T

        P = np.diag(np.ones(d))
        self.M = (P @ M1 @ P.T)*1e+9 # Ajuste de escala


    def update_metric_matrix(self, new_superpixels, new_labels, alpha=0.8):
        """
        Atualiza a matriz métrica existente com base em novos dados.
        """
        if self.M is not None:
            M_old = self.M.copy()
            self.compute_metric_matrix(new_superpixels, new_labels)
            self.M = alpha * M_old + (1 - alpha) * self.M
        else:
            self.compute_metric_matrix(new_superpixels, new_labels)

    def mahalanobis_distance(self, sp1, sp2):
        """
        Calcula a distância de Mahalanobis entre dois superpixels.
        A entrada sp1 e sp2 contém 5 parâmetros: [x, y] para posição e [r, g, b] para cor.
        """

        if self.M is None:
            raise ValueError("A matriz M precisa ser calculada antes de calcular a distância de Mahalanobis.")

        # Distâncias euclidianas para reduzir a dimensão
        diff_pos = np.linalg.norm(np.array(sp1[0]) - np.array(sp2[0]))  # Diferença de posição
        diff_col = np.linalg.norm(np.array(sp1[1]) - np.array(sp2[1]))  # Diferença de cor

        # Vetor reduzido (2 dimensões)
        diff = np.array([diff_pos, diff_col]).reshape(-1, 1)

        # Calcular distância de Mahalanobis
        dist = np.sqrt(diff.T @ self.M @ diff)
        return float(dist)


    def merge_similar_segments(self, segments, superpixels, labels, threshold, show_data=False):
        """
        Mescla superpixels semelhantes com base na distância de Mahalanobis.
        Compara apenas segmentos vizinhos.
        """
        it = time()
        unique_labels = np.unique(labels)
        label_map = {label: label for label in unique_labels}

        distances = []
        switches = 0

        # Lista de posição dos superpixels
        flattened_superpixels = np.array([sp[0] for sp in superpixels])

        # Encontrar vizinhos usando NearestNeighbors
        nn = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(flattened_superpixels)
        neighbors = nn.kneighbors(flattened_superpixels, return_distance=False)

        for i, label1 in enumerate(unique_labels):
            for j in neighbors[i]:
                label2 = labels[j]
                if label1 == label2:
                    continue

                # Garante que estou verificando todos os segmentos desse label
                # Além de busrcar o sp relacionado ao label
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
                # if i < 10: print(f"Distância entre {label1} e {label2}: {dist:.5f}, threshold: {threshold:.5f}")
                if dist < threshold:
                    switches += 1
                    # Unifica transitivamente os labels
                    root1 = label_map[label1]
                    root2 = label_map[label2]
                    new_root = min(int(root1), int(root2))
                    for key, value in label_map.items():
                        if value == root1 or value == root2:
                            label_map[key] = new_root

        # Atualizar os segmentos para refletir os novos labels
        updated_segments = segments.copy()
        for index, new_label in label_map.items():
            updated_segments[segments == index] = new_label

        print(f"Antes: {len(np.unique(segments))} segmentos")
        print(f"Depois: {len(np.unique(updated_segments))} segmentos (com {switches} trocas)")
        print(f"Tempo para mesclar segmentos: {(time()-it):.1f}s")
        if show_data: self.data(distances)

        return updated_segments

    def data(self, list):
        """
        Recebe uma lista e mostra histograma, média e desvio padrão.
        """

        # Calcular média e desvio padrão
        mean_distance = np.mean(list)
        std_distance = np.std(list)
        print(f"Média: {mean_distance:.3f}", end=", ")
        print(f"Des.p.: {std_distance:.3f}")
        print(f"Min: {np.min(list):.3f}", end=", ")
        print(f"Max: {np.max(list):.3f}")

        # Criar histograma
        plt.hist(list, bins=80, edgecolor='black')
        plt.title('Histograma dos dados coletados')
        plt.xlabel('Valor')
        plt.ylabel('Frequência')
        plt.axvline(mean_distance, color='r', linestyle='dashed', linewidth=1, label=f'Média: {mean_distance:.3f}')
        plt.axvline(mean_distance + std_distance, color='g', linestyle='dashed', linewidth=1, label=f'+1 Desvio Padrão: {mean_distance + std_distance:.3f}')
        plt.axvline(mean_distance - std_distance, color='g', linestyle='dashed', linewidth=1, label=f'-1 Desvio Padrão: {mean_distance - std_distance:.3f}')
        plt.legend()
        plt.show()

    def train(self, image, segments):
        """
        Treina o modelo de métrica adaptativa com base nos segmentos fornecidos.
        """
        # Extrair superpixels e labels
        superpixels, labels = self.extract_superpixels(image, segments)
        # Calcular matriz métrica adaptativa
        self.update_metric_matrix(superpixels, labels)

    def classify_image(self, image, segments, threshold=0.1, show_data=False):
        """
        Processa a imagem e os segmentos para agrupar superpixels semelhantes com o mesmo label.
        """
        if self.M is None:
            print("Erro: A matriz métrica não está disponível. Treine o modelo primeiro.")
            return segments

        # Extrair superpixels e labels
        superpixels, labels = self.extract_superpixels(image, segments)
        # Mesclar segmentos semelhantes usando a métrica atual
        updated_segments = self.merge_similar_segments(segments, superpixels, labels, threshold, show_data=show_data)

        return updated_segments
