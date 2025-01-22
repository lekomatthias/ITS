import numpy as np
from scipy.ndimage import label

def Enforce_connectivity(segments):
    """
    Garante que todos os pixels em cada segmento estão conectados em uma das 4 direções.
    Se houver mais de um componente desconectado, os menores recebem o rótulo do primeiro pixel à direita.
    """
    unique_segments = np.unique(segments)
    for segment_id in unique_segments:
        # Ignora o segmento de fundo
        if segment_id == -1:
            continue
        
        mask = segments == segment_id
        labeled_array, num_features = label(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        
        if num_features > 1:
            # Verifica o maior
            component_sizes = [np.sum(labeled_array == feature_id) for feature_id in range(1, num_features + 1)]
            largest_component_id = np.argmax(component_sizes) + 1
            
            # Renomeia os componentes menores
            for feature_id in range(1, num_features + 1):
                if feature_id != largest_component_id:
                    feature_mask = labeled_array == feature_id
                    rightmost_pixel = np.argwhere(feature_mask)[-1]
                    new_label = segment_id
                    
                    # Tenta colocar o novo rótulo à direita
                    rightmost_row, rightmost_col = rightmost_pixel
                    if rightmost_col + 1 < segments.shape[1]:
                        new_label = segments[rightmost_row, rightmost_col + 1]
                    # Se não, tenta colocar à esquerda
                    else:
                        leftmost_pixel = np.argwhere(feature_mask)[0]
                        leftmost_row, leftmost_col = leftmost_pixel
                        if leftmost_col - 1 >= 0:
                            new_label = segments[leftmost_row, leftmost_col - 1]
                    
                    segments[feature_mask] = new_label

    return segments
