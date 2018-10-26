import struct
from array import array
import numpy as np
import scipy.io as sio
from random import randint
from displayData import displayData

def sampleIMAGES():
    # sampleIMAGES
    # Returns 10000 patches for training

    numpatches = 10000

    with open("train-images-idx3-ubyte", "rb") as f:
        magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
        image_data = array("B", f.read())
        patches = np.zeros((rows*cols, numpatches))
        for i in range(numpatches):
            patches[:,i] = np.array(image_data[i * rows * cols:(i + 1) * rows * cols]).reshape(rows,cols).T.flatten()

    patches = patches / 255

    return patches