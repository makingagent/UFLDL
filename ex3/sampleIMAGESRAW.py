import numpy as np
import scipy.io as sio
from random import randint

#DISPLAYDATA Display 2D data in a nice grid
#   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
#   stored in X in a nice grid. It returns the figure handle h and the 
#   displayed array if requested.
def sampleIMAGESRAW():

    data = sio.loadmat('IMAGES_RAW.mat')
    IMAGES = data['IMAGESr']

    patchSize = 12
    numPatches = 10000

    # Initialize patches with zeros.  Your code will fill in this matrix--one
    # column per patch, 10000 columns. 
    patches = np.zeros((patchSize*patchSize,numPatches))

    p = 0
    for im in range(IMAGES.shape[2]):

        # Sample Patches
        numsamples = (int)(numPatches / IMAGES.shape[2])
        for s in range(numsamples):
            y = randint(0,IMAGES.shape[0]-patchSize)
            x = randint(0,IMAGES.shape[1]-patchSize)
            sample = IMAGES[y:y+patchSize, x:x+patchSize,im]
            patches[:,p] = sample.flatten()
            p = p + 1

    return patches