
from core import *
from interface import Menu

if __name__ == "__main__":

    knn = knn_train()
    classifier = SuperpixelClassifier(num_segments=200)
    clusterer = ClusteringClassifier(num_segments=200)

    SP_list = [
        "slic", 
        "etps", 
        "seeds", 
        "ers", 
        "crs", 
    ]

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
    
    functions = [
        {'name': 'Treinar K-nn', 'function': knn.run, 'mode': 'open'}, 
        {'name': 'Treinar métrica', 'function': classifier.Train, 'mode': 'open'}, 
        {'name': 'Segmentador', 'function': classifier.SP_divide, 'mode': 'select', 'list': SP_list}, 
        {'name': 'Contador', 'function': classifier.classify, 'mode': 'select', 'list': SP_list}, 
        {'name': 'Classificador', 'function': clusterer.Type_visualization, 'mode': 'select', 'list': alg_list}, 
    ]
    
    menu = Menu(functions=functions, 
                title="Operações")

    menu.build_interface()
