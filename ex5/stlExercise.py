## CS294A/CS294W Self-taught Learning Exercise

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mnist import loadMNISTImages, loadMNISTLabels
from initializeParameters import initializeParameters
from sparseAutoencoderCost import sparseAutoencoderCost
from displayData import displayData
from feedForwardAutoencoder import feedForwardAutoencoder
from softmaxCost import softmaxCost

plt.ion()

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  self-taught learning. You will need to complete code in feedForwardAutoencoder.m
#  You will also need to have implemented sparseAutoencoderCost.m and
#  softmaxCost.m from previous exercises.
#
## ======================================================================
#  STEP 0: Here we provide the relevant parameters values that will
#  allow your sparse autoencoder to get good filters; you do not need to
#  change the parameters below.

inputSize = 28 * 28
numLabels = 5
hiddenSize = 200
sparsityParam = 0.1  # desired average activation of the hidden units.
                     # (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		             #  in the lecture notes).
_lambda = 3e-3       # weight decay parameter
beta = 3            # weight of sparsity penalty term
maxIter = 600

## ======================================================================
#  STEP 1: Load data from the MNIST database
#
#  This loads our training and test data from the MNIST database files.
#  We have sorted the data for you in this so that you will not have to
#  change it.

# Load MNIST database files
mnistData = loadMNISTImages("train-images-idx3-ubyte")
mnistLabels = loadMNISTLabels("train-labels-idx1-ubyte")

# Set Unlabeled Set (All Images)

# Simulate a Labeled and Unlabeled set
labeledSet = np.where((mnistLabels >= 0) & (mnistLabels <=4))
unlabeledSet = np.where(mnistLabels >= 5)

labeledLabels = mnistLabels[labeledSet]
numTrain = labeledLabels.shape[0] // 2

unlabeledData = mnistData[:,unlabeledSet[0]]

trainData = mnistData[:, labeledSet[0][0:numTrain]]
trainLabels = mnistLabels[labeledSet[0][0:numTrain]]

testData = mnistData[:, labeledSet[0][numTrain:]]
testLabels = mnistLabels[labeledSet[0][numTrain:]]


# Output Some Statistics
print('# examples in unlabeled set: %d', unlabeledData.shape[1])
print('# examples in supervised training set: %d', trainData.shape[1])
print('# examples in supervised testing set: %d', testData.shape[1])


## ======================================================================
#  STEP 2: Train the sparse autoencoder
#  This trains the sparse autoencoder on the unlabeled training
#  images.

theta = initializeParameters(hiddenSize, inputSize)
opttheta = theta

unlabeledData = unlabeledData[:,np.random.randint(unlabeledData.shape[1], size=10000)]
costFunc = lambda p: sparseAutoencoderCost(p, inputSize, hiddenSize, _lambda, sparsityParam, beta, unlabeledData.T)
res = minimize(costFunc, opttheta, method='L-BFGS-B', jac=True, bounds=None, options={'maxiter': maxIter, 'disp':True})
opttheta = res.x

Theta1 = opttheta[0:hiddenSize * (inputSize + 1)].reshape(hiddenSize, inputSize + 1)

displayData(Theta1[:,1:])

input('Program paused. Press enter to continue.')


##======================================================================
## STEP 3: Extract Features from the Supervised Dataset
#
#  You need to complete the code in feedForwardAutoencoder.m so that the
#  following command will extract features from the data.

trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, trainData)
testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, testData)

##======================================================================
## STEP 4: Train the softmax classifier

inputSize = 200
numClasses = 10

_lambda = 1e-4

theta = 0.005 * np.random.randn(numClasses * inputSize)

costFunc = lambda p: softmaxCost(p, numClasses, inputSize, _lambda, trainFeatures, trainLabels)
res = minimize(costFunc, theta, method='L-BFGS-B', jac=True, bounds=None, options={'maxiter': 100, 'disp':True})
theta = res.x

input('Program paused. Press enter to continue.')

##======================================================================
## STEP 5: Testing

## ----------------- YOUR CODE HERE ----------------------
# Compute Predictions on the test set (testFeatures) using softmaxPredict
# and softmaxModel

theta = theta.reshape((numClasses, inputSize))

a = np.dot(theta, testFeatures)
a = a - np.amax(a, axis=0)
e = np.exp(a)
a = e / np.sum(e, axis=0)

acc = np.mean(np.double(np.argmax(a, axis=0) == testLabels))
print('Accuracy: %0.3f%%\n'%(acc * 100))

input('\nProgram paused. Press enter to continue.\n')