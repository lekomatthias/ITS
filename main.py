
from core import *
from tkinter import filedialog

if __name__ == "__main__":

    path = filedialog.askopenfilename(title="Selecione a imagem para aplicação",
                                                            filetypes=[("Imagens", "*.jpeg;*.jpg;*.png")])
    
    # knn_train()
    classifier = SuperpixelClassifier(num_segments=200)
    # classifier.SP_divide(image_path=path, algorithm="slic")
    # classifier.Train()
    classifier.classify(image_path=path, threshold=5.8, show_data=False)

    alg_list = [
    "KMeans", 
    "DBSCAN", 
    "AgglomerativeClustering", 
    "OPTICS", 
    "AffinityPropagation", 
    "MeanShift", 
    "GaussianMixture", 
    "Birch", 
    "FuzzyCMeans", 
    "SOM"
     ]
    cluster = ClusteringClassifier(num_segments=200)
    # for alg in alg_list: 
    #     cluster.Type_visualization(image_path=path, method=alg, show_inertia=False)
