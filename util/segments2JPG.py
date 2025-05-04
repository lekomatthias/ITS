
import random
import numpy as np

def Color_gen():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

import numpy as np

def segments2JPG(path):

    segments = np.load(path)
    h, w = segments.shape
    unique_labels = np.unique(segments)
    color_map = {label: Color_gen() for label in unique_labels}
    output = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            label = segments[y, x]
            if label in color_map:
                output[y, x] = color_map[label]
            else:
                output[y, x] = (0, 0, 0)
    
    return output


if __name__ == "__main__":
    
    pass
