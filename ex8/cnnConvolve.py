import numpy as np
import scipy.signal

from sigmoid import sigmoid

#cnnConvolve Returns the convolution of the features given by W and b with
#the given images
#
# Parameters:
#  patchDim - patch (feature) dimension
#  numFeatures - number of features
#  images - large images to convolve with, matrix in the form
#           images(r, c, channel, image number)
#  W, b - W, b for features from the sparse autoencoder
#  ZCAWhite, meanPatch - ZCAWhitening and meanPatch matrices used for
#                        preprocessing
#
# Returns:
#  convolvedFeatures - matrix of convolved features in the form
#                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)

def cnnConvolve(patchDim, numFeatures, images, W, b, ZCAWhite, meanPatch):

    numImages = images.shape[3]
    imageDim = images.shape[0]
    imageChannels = images.shape[2]

    convolvedFeatures = np.zeros((numFeatures, numImages, imageDim - patchDim + 1, imageDim - patchDim + 1))

    # Instructions:
    #   Convolve every feature with every large image here to produce the
    #   numFeatures x numImages x (imageDim - patchDim + 1) x (imageDim - patchDim + 1)
    #   matrix convolvedFeatures, such that
    #   convolvedFeatures(featureNum, imageNum, imageRow, imageCol) is the
    #   value of the convolved featureNum feature for the imageNum image over
    #   the region (imageRow, imageCol) to (imageRow + patchDim - 1, imageCol + patchDim - 1)
    #
    # Expected running times:
    #   Convolving with 100 images should take less than 3 minutes
    #   Convolving with 5000 images should take around an hour
    #   (So to save time when testing, you should convolve with less images, as
    #   described earlier)

    wt = W.dot(ZCAWhite)
    bt = b.reshape(-1,1) - wt.dot(meanPatch)

    for imageNum in range(numImages):
        for featureNum in range(numFeatures):
            convolvedImage = np.zeros((imageDim - patchDim + 1, imageDim - patchDim + 1))

            for channel in range(imageChannels):

                # Obtain the feature (patchDim x patchDim) needed during the convolution
                feature = wt[featureNum, patchDim*patchDim*channel:patchDim*patchDim*(channel+1)].reshape(patchDim, patchDim)
                feature = np.flipud(np.fliplr(feature))

                im = images[:,:,channel,imageNum]

                convolvedImage = convolvedImage + scipy.signal.convolve(im, feature, 'valid')

            convolvedImage = sigmoid(convolvedImage + bt[featureNum])
            convolvedFeatures[featureNum,imageNum,:,:] = convolvedImage

    return convolvedFeatures
