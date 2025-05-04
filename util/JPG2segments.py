
import numpy as np
from PIL import Image

def JPG2segments(path):
    image = np.array(Image.open(path))
    # Condensa os canais RGB em um único inteiro por pixel
    packed = (image[:, :, 0].astype(np.uint32) << 16) | \
             (image[:, :, 1].astype(np.uint32) << 8)  | \
              image[:, :, 2].astype(np.uint32)

    _, inverse = np.unique(packed, return_inverse=True)
    segments = inverse.reshape(image.shape[:2]) + 1

    return segments


if __name__ == "__main__":

    pass
