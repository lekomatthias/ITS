
import csv
import numpy as np

from util.process_f2f import Process_f2f

def CSV2segments(path):
    
    with open(path, mode='r') as arquivo_csv:
        reader = csv.reader(arquivo_csv)
        data = [[int(valor) for valor in linha] for linha in reader]
    array = np.array(data, dtype=np.int32)
    return array

def CSV2segments_process():
    Process_f2f(CSV2segments, np.save)

if __name__ == "__main__":
    
    pass
