import numpy as np
from scipy.ndimage import label, find_objects

from util.timing import timing

def get_right_or_left_label(row, col, w, segments, segment_id):
    if col + 1 < w:
        new_label = segments[row, col + 1]
        if new_label != segment_id:
            return new_label
    if col - 1 >= 0:
        new_label = segments[row, col - 1]
        if new_label != segment_id:
            return new_label
    return segment_id


@timing
def Enforce_connectivity(segments):

    print("Garantindo conectividade...")
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=int)

    h, w = segments.shape
    segments = segments.copy()

    for segment_id in np.unique(segments):
        if segment_id <= 0:
            continue

        mask = segments == segment_id
        if np.count_nonzero(mask) == 0:
            continue

        labeled_array, num_features = label(mask, structure=structure)
        if num_features <= 1:
            continue

        component_sizes = np.bincount(labeled_array.ravel())[1:]
        largest_component_id = np.argmax(component_sizes) + 1

        objects = find_objects(labeled_array)

        for feature_id in range(1, num_features + 1):
            if feature_id == largest_component_id:
                continue

            sl = objects[feature_id - 1]
            if sl is None:
                continue

            feature_mask = (labeled_array[sl] == feature_id)

            rows, cols = np.where(feature_mask)
            if rows.size == 0:
                continue

            rows_global = rows + sl[0].start
            cols_global = cols + sl[1].start

            row = rows_global[-1]
            col = cols_global[-1]

            new_label = get_right_or_left_label(row, col, w, segments, segment_id)

            segments[sl][feature_mask] = new_label

    return segments
