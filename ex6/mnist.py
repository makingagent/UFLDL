import struct
from array import array
import numpy as np

def loadMNISTImages(path):

    with open(path, "rb") as f:
        magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
        image_data = array("B", f.read())
        patches = np.zeros((rows*cols, size))
        for i in range(size):
            patches[:,i] = np.array(image_data[i * rows * cols:(i + 1) * rows * cols]).reshape(rows,cols).T.flatten()

    patches = patches / 255

    return patches

def loadMNISTLabels(path):

    with open(path, "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        label_data = array("B", f.read())
        labels = np.zeros(size, dtype=int)
        for i in range(size):
            labels[i] = label_data[i]

    return labels

