
import cv2
import os

from util import *

class Metric_train:
    """
    Classe para realizar o treinamento da matriz métrica.
    """

    def __init__(self, num_segments=0, new_model=False, LAB=True):
        self.new_model = new_model
        self.LAB = LAB
        self.num_segments = num_segments

    def Train(self, image_path=None, root=None):
        """
        Faz o treinamento da matriz métrica.
        """

        image, apply_image_dir, _ = Load_Image(image_path)
        
        # seleção de modelo métrica para continuar o treinamento
        metric_name = f"metrica_{'LAB' if self.LAB else 'RGB'}_{self.num_segments}.npy"
        metric_path = os.path.join(apply_image_dir, "metricas", metric_name)
        if not os.path.exists(metric_path) and self.new_model:
            print("Nenhum modelo encontrado, treinando um novo modelo.")
        Similar_SP = AdaptiveMetric()
        Similar_SP.load_metric(metric_path)
        print(f"Modelo carregado de {metric_path}")

        if self.LAB:
            img_temp = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            img_temp[:, :, 0] = 0 # Zera a luminizidade
        else:
            img_temp = image
        labeler = SP_grouper(image, root=root)
        try:
            segments = labeler.run()
            Similar_SP.train(img_temp, segments)
        except:
            print("Nenhuma segmentação selecionada. Encerrando o programa.")
            exit()
        Similar_SP.save_metric(metric_path)
        print(f"Modelo salvo em {metric_path}")

        return metric_path