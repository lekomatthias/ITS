
from core import *
from interface import Menu
from tkinter import filedialog

if __name__ == "__main__":

    # path = filedialog.askopenfilename(title="Selecione a imagem para aplicação",
    #                                                         filetypes=[("Imagens", "*.jpeg;*.jpg;*.png")])
    
    # knn_train()

    SP_list = [
    "slic", 
    "etps", 
    "seeds", 
    "ers", 
    "crs", 
    ]

    classifier = SuperpixelClassifier(num_segments=200)
    # classifier.Train()
    for sp in SP_list:
    #     classifier.MakeMask(image_path=path)
        # classifier.SP_divide(image_path=path, algorithm=sp)
        # classifier.classify(image_path=path, algorithm=sp, threshold=5.8)
        pass

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
    "SOM",
     ]
    
    cluster = ClusteringClassifier(num_segments=200)
    # for alg in alg_list: 
    #     cluster.Type_visualization(image_path=path, method=alg, show_inertia=False)

    functions = [
        {'name': 'Contador', 'function': classifier.classify, 'mode': 'select', 'list': SP_list},
        {'name': 'Classificador', 'function': cluster.Type_visualization, 'mode': 'select', 'list': alg_list},
    ]

    menu = Menu(functions=functions, 
                title="Operações")

    menu.build_interface()
