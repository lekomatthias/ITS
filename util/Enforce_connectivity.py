import numpy as np
from scipy.ndimage import label

def Enforce_connectivity(segments):
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=int)

    h, w = segments.shape
    segments = segments.copy()

    for segment_id in np.unique(segments):
        if segment_id == -1: continue
        mask = segments == segment_id
        labeled_array, num_features = label(mask, structure=structure)
        if num_features <= 1: continue

        # Contagem dos tamanhos dos componentes (ignora 0)
        component_sizes = np.bincount(labeled_array.ravel())[1:]
        largest_component_id = np.argmax(component_sizes) + 1

        # Máscara dos componentes menores
        for feature_id in range(1, num_features + 1):
            if feature_id == largest_component_id: continue
            feature_mask = labeled_array == feature_id
            indices = np.flatnonzero(feature_mask.ravel())
            if indices.size == 0: continue
            # Localiza o último pixel (direita) do componente
            flat_index = indices[-1]
            row, col = divmod(flat_index, w)
            if col + 1 < w:
                new_label = segments[row, col + 1]
            elif col - 1 >= 0:
                new_label = segments[row, col - 1]
            else:
                new_label = segment_id
            segments[feature_mask] = new_label

    return segments
