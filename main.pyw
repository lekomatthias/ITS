
from core import *
from interface import Menu
from util import CSV2segments_process

if __name__ == "__main__":

    n_seg = 200
    knn = knn_train()
    metric = Metric_train(num_segments=n_seg)
    classifier = SuperpixelClassifier(num_segments=n_seg)
    clusterer = ClusteringClassifier(num_segments=n_seg)

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
        {'name': 'Segmentador', 'function': classifier.SP_divide, 'mode': 'select', 'list': SP_list}, 
        {'name': 'Treinar métrica', 'function': metric.Train, 'mode': 'open'}, 
        {'name': 'Contador', 'function': classifier.classify, 'mode': 'select', 'list': SP_list}, 
        {'name': 'Classificador', 'function': clusterer.Type_visualization, 'mode': 'select', 'list': alg_list}, 
        {'name': 'Classificador completo', 'function': clusterer.Type_visualization_list, 'mode': 'batch', 'list': alg_list}, 
        {'name': 'CSV para segmentos', 'function': CSV2segments_process, 'mode': 'button'}, 

    ]
    
    menu = Menu(functions=functions, 
                title="Operações")

    menu.build_interface()
