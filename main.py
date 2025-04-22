
from tkinter import filedialog
from core import *

if __name__ == "__main__":

    classifier = ClusteringClassifier2(num_segments=200)
    # classifier.SP_divide()
    # classifier.Train()
    # classifier.classify(threshold=5.8, show_data=False)
    path = filedialog.askopenfilename(title="Selecione a imagem para aplicação",
                                                            filetypes=[("Imagens", "*.jpeg;*.jpg;*.png")])
    alg_list = [
    "DBSCAN", 
     "KMeans", 
     "AgglomerativeClustering", 
     "OPTICS", 
     "AffinityPropagation", 
     "MeanShift", 
     "GaussianMixture", 
     "Birch", 
     "FuzzyCMeans", 
     "SOM"
     ]
    for alg in alg_list: 
        classifier.Type_visualization(image_path=path, method=alg, show_inertia=False)
