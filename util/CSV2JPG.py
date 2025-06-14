
import csv
import random
import numpy as np
from collections import defaultdict
from skimage.io import imsave

from util.process_f2f import Process_f2f

def Color_gen():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def CSV2JPG(path):
    colors = defaultdict(Color_gen)
    pixels = []

    with open(path, mode='r') as file:
        reader = csv.reader(file)
        for linha in reader:
            row = []
            for valor in linha:
                label = int(valor)
                row.append(colors[label])
            pixels.append(row)

    return np.array(pixels, dtype=np.uint8)


def CSV2JPG_process():
    Process_f2f(CSV2JPG, imsave, type_in="csv", type_out="jpg")

if __name__ == "__main__":

    pass
