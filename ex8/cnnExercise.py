## CS294A/CS294W Convolutional Neural Networks Exercise

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random

from displayColorNetwork import displayColorNetwork
from cnnConvolve import cnnConvolve
from feedForwardAutoencoder import feedForwardAutoencoder
from cnnPool import cnnPool
from softmaxCost import softmaxCost
from scipy.optimize import minimize

plt.ion()

##======================================================================
## STEP 0: Initialization
#  Here we initialize some parameters used for the exercise.

imageDim = 64         # image dimension
imageChannels = 3     # number of channels (rgb, so 3)

patchDim = 8          # patch dimension
numPatches = 50000    # number of patches

visibleSize = patchDim * patchDim * imageChannels  # number of input units
outputSize = visibleSize   # number of output units
hiddenSize = 400           # number of hidden units

epsilon = 0.1	       # epsilon for ZCA whitening

poolDim = 19          # dimension of pooling region

data = sio.loadmat('features.mat')
optTheta = data['optTheta']
ZCAWhite = data['ZCAWhite']
meanPatch = data['meanPatch']

W = optTheta.reshape(-1,1)[0:hiddenSize * (visibleSize + 1)].reshape(hiddenSize, visibleSize + 1)
b = W[:,0]
W = W[:,1:]

displayColorNetwork( W.dot(ZCAWhite).T )

input('\nProgram paused. Press enter to continue.\n')


##======================================================================
## STEP 2: Implement and test convolution and pooling
#  In this step, you will implement convolution and pooling, and test them
#  on a small part of the data set to ensure that you have implemented
#  these two functions correctly. In the next step, you will actually
#  convolve and pool the features with the STL10 images.

## STEP 2a: Implement convolution
#  Implement convolution in the function cnnConvolve in cnnConvolve.m

# Note that we have to preprocess the images in the exact same way
# we preprocessed the patches before we can obtain the feature activations.

data = sio.loadmat('stlTrainSubset.mat')

## Use only the first 8 images for testing
convImages = data['trainImages'][:,:,:,0:8]

# NOTE: Implement cnnConvolve in cnnConvolve.m first!
convolvedFeatures = cnnConvolve(patchDim, hiddenSize, convImages, W, b, ZCAWhite, meanPatch)

## STEP 2b: Checking your convolution
#  To ensure that you have convolved the features correctly, we have
#  provided some code to compare the results of your convolution with
#  activations from the sparse autoencoder

# For 1000 random points
for i in range(1000):
    featureNum = random.randint(0, hiddenSize-1)
    imageNum = random.randint(0, 7)
    imageRow = random.randint(0, imageDim - patchDim)
    imageCol = random.randint(0, imageDim - patchDim)

    patch = convImages[imageRow:imageRow+patchDim,imageCol:imageCol+patchDim,:,imageNum]
    patch = np.concatenate((patch[:,:,0].flatten(), patch[:,:,1].flatten(), patch[:,:,2].flatten())).reshape(-1,1)
    patch = patch - meanPatch
    patch = ZCAWhite.dot(patch)

    features = feedForwardAutoencoder(optTheta.reshape(-1,1), hiddenSize, visibleSize, patch)

    if abs(features[featureNum,0] - convolvedFeatures[featureNum, imageNum, imageRow, imageCol] > 1e-9):
        print('Convolved feature does not match activation from autoencoder')
        print('Feature Number    : %d'%featureNum)
        print('Image Number      : %d'%imageNum)
        print('Image Row         : %d'%imageRow)
        print('Image Column      : %d'%imageCol)
        print('Convolved feature : %0.5f'%convolvedFeatures[featureNum, imageNum, imageRow, imageCol])
        print('Sparse AE feature : %0.5f'%features[featureNum,0])
        exit(0)

print('Congratulations! Your convolution code passed the test.')
input('\nProgram paused. Press enter to continue.\n')

## STEP 2c: Implement pooling
#  Implement pooling in the function cnnPool in cnnPool.m

# NOTE: Implement cnnPool in cnnPool.m first!
pooledFeatures = cnnPool(poolDim, convolvedFeatures)

## STEP 2d: Checking your pooling
#  To ensure that you have implemented pooling, we will use your pooling
#  function to pool over a test matrix and check the results.

testMatrix = np.arange(64).reshape(8,8)
expectedMatrix = np.array([[np.mean(testMatrix[0:4, 0:4]), np.mean(testMatrix[0:4, 4:8])],
                          [np.mean(testMatrix[4:8, 0:4]), np.mean(testMatrix[4:8, 4:8])]])

testMatrix = testMatrix.reshape((1,1,8,8))
pooledFeatures = cnnPool(4, testMatrix)

if not (pooledFeatures == expectedMatrix).all():
    print('Pooling incorrect')
    print('Expected matrix')
    print(expectedMatrix)
    print('Got')
    print(pooledFeatures)
    exit(0)

print('Congratulations! Your pooling code passed the test.')
input('\nProgram paused. Press enter to continue.\n')

##======================================================================
## STEP 3: Convolve and pool with the dataset
#  In this step, you will convolve each of the features you learned with
#  the full large images to obtain the convolved features. You will then
#  pool the convolved features to obtain the pooled features for
#  classification.
#
#  Because the convolved features matrix is very large, we will do the
#  convolution and pooling 50 features at a time to avoid running out of
#  memory. Reduce this number if necessary

stepSize = 10

data = sio.loadmat('stlTrainSubset.mat')
numTrainImages = (int)(data['numTrainImages'])
trainImages = data['trainImages']
trainLabels = data['trainLabels']

data = sio.loadmat('stlTestSubset.mat')
numTestImages = (int)(data['numTestImages'])
testImages = data['testImages']
testLabels = data['testLabels']


pooledFeaturesDim = (int)(np.floor( (imageDim-patchDim+1)/poolDim ))

pooledFeaturesTrain = np.zeros((hiddenSize,numTrainImages,pooledFeaturesDim,pooledFeaturesDim))
pooledFeaturesTest = np.zeros((hiddenSize,numTestImages,pooledFeaturesDim,pooledFeaturesDim))

for convPart in range(hiddenSize//stepSize):

    featureStart = convPart*stepSize
    featureEnd = (convPart+1)*stepSize

    print('Step %d: features %d to %d'%(convPart, featureStart, featureEnd))
    wt = W[featureStart:featureEnd,:]
    bt = b[featureStart:featureEnd]

    print('Convolving and pooling train images')
    convolvedFeaturesThis = cnnConvolve(patchDim, stepSize, trainImages, wt, bt, ZCAWhite, meanPatch)
    pooledFeaturesThis = cnnPool(poolDim, convolvedFeaturesThis)
    pooledFeaturesTrain[featureStart:featureEnd,:,:,:] = pooledFeaturesThis

    print('Convolving and pooling test images')
    convolvedFeaturesThis = cnnConvolve(patchDim, stepSize, testImages, wt, bt, ZCAWhite, meanPatch)
    pooledFeaturesThis = cnnPool(poolDim, convolvedFeaturesThis)
    pooledFeaturesTest[featureStart:featureEnd,:,:,:] = pooledFeaturesThis


# You might want to save the pooled features since convolution and pooling takes a long time
sio.savemat('cnnPooledFeatures.mat', {
    'pooledFeaturesTrain': pooledFeaturesTrain,
    'pooledFeaturesTest': pooledFeaturesTest
})


##======================================================================
## STEP 4: Use pooled features for classification
#  Now, you will use your pooled features to train a softmax classifier,
#  using softmaxTrain from the softmax exercise.
#  Training the softmax classifer for 1000 iterations should take less than
#  10 minutes.

# Add the path to your softmax solution, if necessary
# addpath /path/to/solution/

# data = sio.loadmat('cnnPooledFeatures.mat')
#
# pooledFeaturesTrain = data['pooledFeaturesTrain']
# pooledFeaturesTest = data['pooledFeaturesTest']

# Setup parameters for softmax
softmaxLambda = 1e-4
numClasses = 4
# Reshape the pooledFeatures to form an input vector for softmax
softmaxX = np.transpose(pooledFeaturesTrain, axes=[0,2,3,1])
softmaxX = softmaxX.reshape((softmaxX.size//numTrainImages, numTrainImages))
softmaxY = trainLabels.flatten()-1
print(softmaxY)
input('\nProgram paused. Press enter to continue.\n')

theta = 0.005 * np.random.randn(numClasses * softmaxX.size//numTrainImages)

costFunc = lambda p: softmaxCost(p, numClasses, softmaxX.size//numTrainImages, softmaxLambda, softmaxX, softmaxY)
res = minimize(costFunc, theta, method='L-BFGS-B', jac=True, bounds=None, options={'maxiter': 1000, 'disp':True})
theta = res.x

##======================================================================
## STEP 5: Test classifer
#  Now you will test your trained classifer against the test images

softmaxX = np.transpose(pooledFeaturesTest, axes=[0,2,3,1])
softmaxX = softmaxX.reshape((softmaxX.size//numTestImages, numTestImages))
softmaxY = testLabels.flatten()-1

theta = theta.reshape((numClasses, softmaxX.size//numTestImages))

a = np.dot(theta, softmaxX)
a = a - np.amax(a, axis=0)
e = np.exp(a)
a = e / np.sum(e, axis=0)

acc = np.mean(np.double(np.argmax(a, axis=0) == softmaxY))
print('Accuracy: %0.3f%%\n'%(acc * 100))

# You should expect to get an accuracy of around 80% on the test images.