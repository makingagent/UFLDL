import numpy as np

#cnnPool Pools the given convolved features
#
# Parameters:
#  poolDim - dimension of pooling region
#  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
#                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)
#
# Returns:
#  pooledFeatures - matrix of pooled features in the form
#                   pooledFeatures(featureNum, imageNum, poolRow, poolCol)
#

def cnnPool(poolDim, convolvedFeatures):

    numImages = convolvedFeatures.shape[1]
    numFeatures = convolvedFeatures.shape[0]
    convolvedDim = convolvedFeatures.shape[2]
    resDim = (int)(np.floor(convolvedDim / poolDim))

    pooledFeatures = np.zeros((numFeatures, numImages, resDim, resDim))

    # Instructions:
    #   Now pool the convolved features in regions of poolDim x poolDim,
    #   to obtain the
    #   numFeatures x numImages x (convolvedDim/poolDim) x (convolvedDim/poolDim)
    #   matrix pooledFeatures, such that
    #   pooledFeatures(featureNum, imageNum, poolRow, poolCol) is the
    #   value of the featureNum feature for the imageNum image pooled over the
    #   corresponding (poolRow, poolCol) pooling region
    #   (see http://ufldl/wiki/index.php/Pooling )
    #
    #   Use mean pooling here.

    for imageNum in range(numImages):
        for featureNum in range(numFeatures):
            for poolRow in range(resDim):
                for poolCol in range(resDim):

                    patch = convolvedFeatures[
                        featureNum,
                        imageNum,
                        poolRow*poolDim:poolRow*poolDim+poolDim,
                        poolCol*poolDim:poolCol*poolDim+poolDim
                    ]

                    pooledFeatures[featureNum,imageNum,poolRow,poolCol] = np.mean(patch)

    return pooledFeatures
