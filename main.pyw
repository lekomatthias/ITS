from time import time
it = time()

from core import *
from interface import Menu
from util import CSV2segments_process, All_images

print(f"tempo até imports: {time()-it:.2f}s")

if __name__ == "__main__":

    n_seg = 200
    knn = knn_train()
    metric = Metric_train(num_segments=n_seg)
    classifier = SuperpixelClassifier(num_segments=n_seg)
    all_sp_divide = All_images(classifier.SP_divide)
    all_count = All_images(classifier.classify)
    clusterer = ClusteringClassifier(num_segments=n_seg)
    all_classify = All_images(clusterer.Type_visualization_list)

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
        {'name': 'CSV para segmentos', 'function': CSV2segments_process, 'mode': 'button'}, 
        {'name': 'Treinar K-nn', 'function': knn.run, 'mode': 'open'}, 
        {'name': 'Segmentador', 'function': classifier.SP_divide, 'mode': 'select', 'list': SP_list}, 
        {'name': 'Treinar métrica', 'function': metric.Train, 'mode': 'open'}, 
        {'name': 'Contador', 'function': classifier.classify, 'mode': 'select', 'list': SP_list}, 
        {'name': 'Classificador', 'function': clusterer.Type_visualization, 'mode': 'select', 'list': alg_list}, 
        {'name': 'Classificador completo', 'function': clusterer.Type_visualization_list, 'mode': 'batch', 'list': alg_list}, 
        {'name': 'Segmentador em pasta', 'function': all_sp_divide, 'mode': 'select', 'list': SP_list}, 
        {'name': 'Contador em pasta', 'function': all_count, 'mode': 'select', 'list': SP_list},  
        {'name': 'Classificar em pasta', 'function': all_classify, 'mode': 'batch', 'list': alg_list}, 

    ]
    
    menu = Menu(functions=functions, 
                title="Operações")
    
    print(f"tempo até criação de Menu: {time()-it:.2f}s")

    menu.build_interface()
