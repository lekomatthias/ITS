
import csv
import random
import numpy as np
from PIL import Image

def Color_gen():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def CSV2JPG(path):

    with open(path, mode='r') as arquivo_csv:
        reader = csv.reader(arquivo_csv)
        data = list(reader)

    unique_labels = set()
    for linha in data:
        for valor in linha:
            unique_labels.add(int(valor))

    # Criar um dicionário de cores associadas a cada número único
    colors = {numero: Color_gen() for numero in unique_labels}

    h = len(data)
    w = len(data[0])
    image = Image.new('RGB', (w, h))
    for y, line in enumerate(data):
        for x, value in enumerate(line):
            label = int(value)
            color = colors[label]
            image.putpixel((x, y), color)
    image = np.array(image, np.uint8)

    return image

if __name__ == "__main__":

    pass
