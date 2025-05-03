
import csv
import numpy as np

def CSV2segments(path):
    
    with open(path, mode='r') as arquivo_csv:
        reader = csv.reader(arquivo_csv)
        data = [[int(valor) for valor in linha] for linha in reader]
    array = np.array(data, dtype=np.int32)
    return array

if __name__ == "__main__":

    pass
