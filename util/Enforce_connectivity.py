import numpy as np
import os
from scipy.ndimage import label, find_objects, binary_dilation
from concurrent.futures import ThreadPoolExecutor

from util.timing import timing

def process_label(segment_map, label_id, structure):
    if label_id <= 0:
        return None

    mask = segment_map == label_id
    if not np.any(mask):
        return None

    components, num_components = label(mask, structure=structure)
    if num_components <= 1:
        return None

    component_sizes = np.bincount(components.ravel())[1:]
    largest_component_id = np.argmax(component_sizes) + 1
    slices = find_objects(components)

    updates = []

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

        neighbors = neighbors[(neighbors != label_id) & (neighbors > 0)]

        if neighbors.size > 0:
            labels, counts = np.unique(neighbors, return_counts=True)
            new_label = labels[np.argmax(counts)]
        else:
            new_label = label_id

        updates.append((sl, local_mask, new_label))

    return updates

@timing
def Enforce_connectivity(segment_map):

    print("Garantindo conectividade...")
    segment_map = segment_map.copy()
    structure = np.array([[0,1,0],
                          [1,1,1],
                          [0,1,0]], dtype=int)

    unique_labels = np.unique(segment_map)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_label, segment_map, label_id, structure)
                   for label_id in unique_labels]

        results = [f.result() for f in futures]

    for updates in results:
        if updates is None:
            continue
        for sl, local_mask, new_label in updates:
            segment_map[sl][local_mask] = new_label

    return segment_map
