import numpy as np
from scipy.ndimage import label, find_objects, binary_dilation
from util.timing import timing

@timing
def Enforce_connectivity(segment_map):
    h, w = segment_map.shape
    segment_map = segment_map.copy()
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=int)

    unique_labels = np.unique(segment_map)
    total = len(unique_labels)
    bar_length = 50
    step = max(total // bar_length, 1)

    print(f"Garantindo conectividade ({len(unique_labels)} labels):")
    print("." + "_" * bar_length + ".")
    print("[", end="", flush=True)

    for i, label_id in enumerate(unique_labels):
        if i % step == 0 and i // step < bar_length:
            print("=", end="", flush=True)

        if label_id <= 0:
            continue

        mask = segment_map == label_id
        if np.count_nonzero(mask) == 0:
            continue

        components, num_components = label(mask, structure=structure)
        if num_components <= 1:
            continue

        component_sizes = np.bincount(components.ravel())[1:]
        largest_component_id = np.argmax(component_sizes) + 1
        slices = find_objects(components)

        for comp_id in range(1, num_components + 1):
            if comp_id == largest_component_id:
                continue

            sl = slices[comp_id - 1]
            if sl is None:
                continue

            local_mask = components[sl] == comp_id
            if not np.any(local_mask):
                continue

            global_mask = np.zeros_like(segment_map, dtype=bool)
            global_mask[sl] = local_mask

            border = binary_dilation(global_mask, structure=structure) & (~global_mask)
            neighbors = segment_map[border]

            # Corrigido: remove negativos e o próprio rótulo
            neighbors = neighbors[(neighbors != label_id) & (neighbors > 0)]

            if neighbors.size > 0:
                new_label = np.bincount(neighbors).argmax()
            else:
                new_label = label_id

            segment_map[sl][local_mask] = new_label

    print("]")
    return segment_map
