import matplotlib.pyplot as plt
import numpy as np

def displayColorNetwork(A):

    if np.min(A) >= 0:
        A = A - np.mean(A)

    cols = (int)(np.round(np.sqrt(A.shape[1])))

    channel_size = (int)(A.shape[0] / 3)
    dim = (int)(np.sqrt(channel_size))
    dimp = dim + 1
    rows = (int)(np.ceil(A.shape[1]/cols))
    B = A[0:channel_size,:]
    C = A[channel_size:channel_size*2,:]
    D = A[2*channel_size:channel_size*3,:]
    B = B / np.amax(np.abs(B), axis=0)
    C = C / np.amax(np.abs(C), axis=0)
    D = D / np.amax(np.abs(D), axis=0)

    # Initialization of the image
    I = np.ones((dim*rows+rows-1,dim*cols+cols-1,3))

    #Transfer features to this image matrix
    for i in range(rows):
        for j in range(cols):

            if i*cols+j > B.shape[1]:
                break

            # This sets the patch
            I[i*dimp:i*dimp+dim,j*dimp:j*dimp+dim,0] = B[:,i*cols+j].reshape(dim, dim)
            I[i*dimp:i*dimp+dim,j*dimp:j*dimp+dim,1] = C[:,i*cols+j].reshape(dim, dim)
            I[i*dimp:i*dimp+dim,j*dimp:j*dimp+dim,2] = D[:,i*cols+j].reshape(dim, dim)

    I = I + 1
    I = I / 2

    plt.imshow(I)
    plt.axis('off')