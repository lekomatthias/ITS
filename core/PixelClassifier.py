import numpy as np
import os
import joblib

# Define o limite máximo de núcleos lógicos
os.environ["LOKY_MAX_CPU_COUNT"] = "12"

class PixelClassifier:
    """Classificador KNN com base em dados de treinamento pré-carregados."""
    def __init__(self, model_path, k=3):
        """Carrega um modelo KNN pré-treinado de um arquivo .joblib."""
        try:
            self.k = k
            # Carrega o modelo como um dicionário
            self.model_data = joblib.load(model_path)  
            self.kd_tree = self.model_data["kd_tree"]
            self.labels = self.model_data["labels"]
            self.classes_ = self.model_data["classes_"]
        except FileNotFoundError:
            print(f"Erro: O arquivo do modelo '{model_path}' não foi encontrado.")
            exit()
        except KeyError as e:
            print(f"Erro: O modelo salvo está faltando a chave esperada: {e}")
            exit()

    def predict(self, pixel):
        """Prediz a classe de um pixel com base nos k vizinhos mais próximos."""
        # Converter o pixel para o formato adequado para o KNN
        pixel = np.array(pixel).reshape(1, -1)
        dist, ind = self.kd_tree.query(pixel, k=self.k)
        neighbors = self.labels[ind[0]]
        predicted_label = np.bincount(neighbors).argmax()
        return not predicted_label

    def get_num_classes(self):
        """Obtém automaticamente o número de classes do modelo KNN."""
        return len(self.classes_)
