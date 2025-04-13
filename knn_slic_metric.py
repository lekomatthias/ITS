import numpy as np
import cv2
import os
import joblib
from skimage.segmentation import slic
from skimage.io import imsave
from tkinter import filedialog
from time import time

from knn_apply import SuperpixelClassifier
from knn_slic import PixelClassifier2
from SP_grouper import InteractiveSegmentLabeler
from AdaptiveMetric import AdaptiveMetric
from Enforce_connectivity import Enforce_connectivity
from timing import timing
from segment_qntt_select import GetPixelsOfArea

class SuperpixelClassifier2(SuperpixelClassifier):
    """
    Classificador de superpixels baseado em KNN e cores contrastantes.
    """

    def __init__(self, num_segments=0, new_model=False, LAB=True):
        """Inicializa o classificador com um modelo treinado."""

        self.new_model = new_model
        self.Similar_SP = AdaptiveMetric()
        self.LAB = LAB
        self.num_segments = num_segments
        
    @timing
    def Load_Image(self, apply_image_path=None):
        """
        Carrega a imagem para aplicação.
        """

        try:
            if apply_image_path is None:
                apply_image_path = filedialog.askopenfilename(
                    title="Selecione a imagem para aplicação",
                    filetypes=[("Imagens", "*.jpeg;*.jpg;*.png")]
                )
            if not apply_image_path:
                print("Nenhuma imagem selecionada. Encerrando o programa.")
                raise FileNotFoundError("Imagem não selecionada.")
            if not os.path.exists(apply_image_path):
                print("Imagem selecionada não existe. Encerrando o programa.")
                raise FileNotFoundError(f"Arquivo não encontrado: {apply_image_path}")
            # Verifica a extensão da imagem
            valid_extensions = ('.png', '.jpg', '.jpeg')
            if not apply_image_path.lower().endswith(valid_extensions):
                print("Formato de imagem inválido. Apenas PNG e JPG são suportados.")
                raise ValueError("Extensão inválida para imagem.")
        except (FileNotFoundError, ValueError) as e:
            print(f"[ERRO] {e}")
            exit()
        except Exception as e:
            print(f"[ERRO INESPERADO] {e}")
            exit()
        apply_image_dir, apply_image_name = os.path.split(apply_image_path)
        apply_image_name_no_ext, _ = os.path.splitext(apply_image_name)
        image = cv2.imread(apply_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, apply_image_dir, apply_image_name_no_ext
    
    def First2Zero(self, segments):
        init = segments[0, 0]
        if init == 0: return segments
        final = (segments == 0)*init
        initial = (segments == init)*init
        segments = segments - initial + final
        return segments

    @timing
    def SP_divide(self, image_path=None, standard_size=None):
        """
        Aplica a segmentação em superpixels para imagem.
        """

        image, apply_image_dir, apply_image_name_no_ext = self.Load_Image(image_path)

        # Verifica se a imagem é muito grande e redimensiona
        original_size = image.shape[:2]
        if standard_size == None: standard_size = image.shape[1]//2
        if image.shape[1] > 3000:
            scale_factor = standard_size / image.shape[1]
            new_width = standard_size
            new_height = int(image.shape[0] * scale_factor)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"Imagem redimensionada para {new_width}x{new_height}.")
        
        auto_num_segments = False
        if self.num_segments == 0:
            pix = GetPixelsOfArea(image=image, target_height=600)
            auto_num_segments = True

        segments_path = os.path.join(apply_image_dir, "segmentos",
                                     f"segmentos_{apply_image_name_no_ext}_{self.num_segments}.npy")
        mask_path = os.path.join(apply_image_dir, "mascaras",
                                 f"mask_{apply_image_name_no_ext}.npy")

        print("Selecione o modelo a nível de pixel para aplicação.")
        model_path = filedialog.askopenfilename(title="Selecione o modelo para aplicação",
                                                filetypes=[("joblib", "*.joblib")])
        if not model_path:
            print("Nenhum modelo selecionado. Encerrando o programa.")
            return
        classifier = PixelClassifier2(model_path=model_path)

        it = time()
        print("Dividindo a imagem em superpixels...")
        # Processamento inicial dos pixels (caso de imagem colorida ou em escala de cinza)
        if len(image.shape) > 2:
            reshaped_image = image.reshape((-1, image.shape[-1]))
        else:
            reshaped_image = image
        
        results = joblib.Parallel(n_jobs=-1)(joblib.delayed(classifier.predict)(pixel) for pixel in reshaped_image)

        # Reconstrói os resultados para o formato original da imagem
        results = np.array(results).reshape(image.shape[:2])
        np.save(mask_path, results)

        if auto_num_segments:
            self.num_segments = int(np.count_nonzero(results) // pix)
            print(f"Quantidade de superpixels: {self.num_segments}")

        # Segmentar a imagem usando SLIC
        segments = slic(image, n_segments=self.num_segments, compactness=15, sigma=1,
                        start_label=0, min_size_factor=2e-1, max_size_factor=1e+1, mask=results)
        print(f"Tempo para segmentar a imagem: {(time()-it):.1f}s")
        it = time()
        # Esta função decide se todos os segmentos devem realmente ser conectados.
        segments = Enforce_connectivity(segments)
        print(f"Tempo para garantir conectividade: {(time()-it):.1f}s")

        # Redimensiona os segmentos e a máscara para o tamanho original, se necessário
        if image.shape[:2] != original_size:
            segments = cv2.resize(segments.astype(np.uint8), (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
            results = cv2.resize(results.astype(np.uint8), (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
            print(f"Segmentos e máscara redimensionados para o tamanho original {original_size[1]}x{original_size[0]}.")

        self.Create_image_with_segments(mask_path)
        np.save(segments_path, segments)

    def Train(self, image_path=None):
        """
        Faz o treinamento da matriz métrica.
        """

        image, apply_image_dir, apply_image_name_no_ext = self.Load_Image(image_path)
        
        segments_path = os.path.join(apply_image_dir, "segmentos",
                                     f"seg_DBSCAN_{apply_image_name_no_ext}_{self.num_segments}.npy")
        if not os.path.exists(segments_path):
            self.SP_divide(image_path=image_path)

        # seleção de modelo métrica para continuar o treinamento
        metric_name = f"metrica_{'LAB' if self.LAB else 'RGB'}_{self.num_segments}.npy"
        metric_path = os.path.join(apply_image_dir, "metricas", metric_name)
        if os.path.exists(metric_path) and not self.new_model:
            self.Similar_SP.load_metric(metric_path)
            print(f"Modelo carregado de {metric_path}")
        else:
            print("Nenhum modelo encontrado, treinando um novo modelo.")

        segments = np.load(segments_path)
        if self.LAB:
            img_temp = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            # Zera a luminozidade
            img_temp[:, :, 0] = 0
        else:
            img_temp = image
        labeler = InteractiveSegmentLabeler(image, segments)
        try:
            segments = labeler.run()
            self.Similar_SP.train(img_temp, segments)
        except:
            print("Nenhuma segmentação selecionada. Encerrando o programa.")
            exit()
        self.Similar_SP.save_metric(metric_path)
        print(f"Modelo salvo em {metric_path}")

        return metric_path

    @timing
    def Paint_image(self, image, segments):
        """
        Gerar imagem colorida para visualização
        """

        self.colors = self.generate_contrasting_colors(len(np.unique(segments)))
        # cor preta para classe NIO
        self.colors[0] = np.array([0, 0, 0])
        color_image = np.zeros(image.shape)
        for segment_value, color in zip(np.unique(segments), self.colors):
            color_image[segments == segment_value] = color
        output_image = image * 0.3 + color_image * 0.7
        return output_image
    
    def Save_image(self, output_image, apply_image_dir, apply_image_name_no_ext, type="classified"):
        """
        Salva a imagem.
        """

        output_image = np.clip(output_image, 0, 255)
        output_image = output_image.astype(np.uint8)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        output_image_path = os.path.join(apply_image_dir, "classificadas",
                                         f"({type}){apply_image_name_no_ext}_{self.num_segments}.jpeg")
        imsave(output_image_path, output_image)
        print(f"Imagem classificada salva em: {output_image_path}")

    @timing
    def classify(self, threshold=1, image_path=None, show_data=False):
        """
        Aplica a classificação aos superpixels da imagem.
        """
        
        image, apply_image_dir, apply_image_name_no_ext = self.Load_Image(image_path)

        segments_path = os.path.join(apply_image_dir, "segmentos",
                                     f"segmentos_{apply_image_name_no_ext}_{self.num_segments}.npy")
        if not os.path.exists(segments_path):
            self.SP_divide(image_path=image_path)

        metric_name = f"metrica_{'LAB' if self.LAB else 'RGB'}_{self.num_segments}.npy"
        print(metric_name)
        metric_path = os.path.join(apply_image_dir, "metricas", metric_name)
        if not os.path.exists(metric_path):
            model_path = filedialog.askopenfilename(title="Carregar modelo", filetypes=[("npy", "*.npy")])
            if not model_path:
                print("Nenhum modelo selecionado. Encerrando o programa.")
                exit()
            self.Similar_SP.load_metric(model_path)
        else:
            self.Similar_SP.load_metric(metric_path)

        print("Classificando os superpixels...")
        segments = np.load(segments_path)
        if self.LAB:
            img_temp = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            # Zera a luminizidade
            img_temp[:, :, 0] = 0
        else:
            img_temp = image
        new_segments = self.Similar_SP.classify_image(img_temp, segments, threshold=threshold,
                                                      show_data=show_data)
        new_segments = self.First2Zero(new_segments)

        output_image = self.Paint_image(image, new_segments)        

        # Salvar a imagem classificada
        self.Save_image(output_image, apply_image_dir, apply_image_name_no_ext, "classified")

        # Salvar os novos segmentos
        final_segments_path = os.path.join(apply_image_dir, "segmentos",
                                           f"seg_finais_{apply_image_name_no_ext}_{self.num_segments}.npy")
        np.save(final_segments_path, new_segments)
        print(f"Novos segmentos salvos em: {final_segments_path}")

        return output_image
    
    def Create_image_with_segments(self, seg_path=None):
        """
        Cria uma imagem com os segmentos coloridos.
        """

        if not seg_path:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            segment_dir = os.path.join(base_dir, "segmentos")
            initial_dir = segment_dir if os.path.exists(segment_dir) else base_dir
            seg_path = filedialog.askopenfilename(initialdir=initial_dir,
                                                title="Selecione os segmentos para criação da imagem",
                                                filetypes=[("npy", "*.npy")])
        
        if not seg_path or not os.path.exists(seg_path):
            print("Nenhum segmento selecionado. Encerrando o programa.")
            exit()
        seg_dir, seg_name = os.path.split(seg_path)
        seg_name = seg_name.replace(".npy", "")

        segments = np.load(seg_path)
        output_image = np.zeros((*segments.shape, 3), dtype=np.uint8)
        if len(np.unique(segments)) == 2: colors = [(0, 0, 0), (255, 255, 255)]
        else: colors = self.generate_contrasting_colors(len(np.unique(segments)))
        segments = segments.astype(int)
        for segment_value, color in zip(np.unique(segments), colors):
            output_image[segments == segment_value] = color
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        imsave(os.path.join(seg_dir, f"{seg_name}_color.jpeg"), output_image)
        print(f"Imagem com segmentos coloridos salva em: {os.path.join(seg_dir, f'{seg_name}_color.jpeg')}")
        return output_image

def main():
    new_segments = False
    train = False
    new_model = True
    LAB = True
    num_segments = 200
    threshold = 5.8

    superpixel_classifier = SuperpixelClassifier2(new_model=new_model, LAB=LAB, num_segments=num_segments)

    # superpixel_classifier.Create_image_with_segments()
    # exit()

    if new_segments:
        # Divide a imagem em superpixels e salva o arquivo de segmentos
        superpixel_classifier.SP_divide()
        return

    if train:
        # Treina a métrica adaptativa e salva o arquivo de modelo
        superpixel_classifier.Train()
    else:
        # Classifica a imagem e salva a imagem colorida
        superpixel_classifier.classify(threshold, show_data=False)

if __name__ == "__main__":
    main()

