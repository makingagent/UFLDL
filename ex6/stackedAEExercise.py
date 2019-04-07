## CS294A/CS294W Stacked Autoencoder Exercise

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mnist import loadMNISTImages, loadMNISTLabels
from initializeParameters import initializeParameters
from sparseAutoencoderCost import sparseAutoencoderCost
from displayData import displayData
from feedForwardAutoencoder import feedForwardAutoencoder
from softmaxCost import softmaxCost
from stackedAECost import stackedAECost
from stackedAEPredict import stackedAEPredict
from checkStackedAECost import checkStackedAECost

plt.ion()

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  sstacked autoencoder exercise. You will need to complete code in
#  stackedAECost.m
#  You will also need to have implemented sparseAutoencoderCost.m and
#  softmaxCost.m from previous exercises. You will need the initializeParameters.m
#  loadMNISTImages.m, and loadMNISTLabels.m files from previous exercises.
#
#  For the purpose of completing the assignment, you do not need to
#  change the code in this file.
#
##======================================================================
## STEP 0: Here we provide the relevant parameters values that will
#  allow your sparse autoencoder to get good filters; you do not need to
#  change the parameters below.

inputSize = 28 * 28
numClasses = 10
hiddenSizeL1 = 200      # Layer 1 Hidden Size
hiddenSizeL2 = 200      # Layer 2 Hidden Size
sparsityParam = 0.1     # desired average activation of the hidden units.
                        # (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		                #  in the lecture notes).
_lambda = 3e-3          # weight decay parameter
beta = 3                # weight of sparsity penalty term

maxIter = 400


##======================================================================
## STEP 1: Load data from the MNIST database
#
#  This loads our training data from the MNIST database files.

# Load MNIST database files
trainData = loadMNISTImages('train-images-idx3-ubyte')
trainLabels = loadMNISTLabels('train-labels-idx1-ubyte')

##======================================================================
## STEP 2: Train the first sparse autoencoder
#  This trains the first sparse autoencoder on the unlabelled STL training
#  images.
#  If you've correctly implemented sparseAutoencoderCost.m, you don't need
#  to change anything here.


#  Randomly initialize the parameters
sae1Theta = initializeParameters(hiddenSizeL1, inputSize)

## ---------------------- YOUR CODE HERE  ---------------------------------
#  Instructions: Train the first layer sparse autoencoder, this layer has
#                an hidden size of "hiddenSizeL1"
#                You should store the optimal parameters in sae1OptTheta

costFunc = lambda p: sparseAutoencoderCost(p, inputSize, hiddenSizeL1, _lambda, sparsityParam, beta, trainData.T)
res = minimize(costFunc, sae1Theta, method='L-BFGS-B', jac=True, bounds=None, options={'maxiter': maxIter, 'disp':True})
sae1OptTheta = res.x
sae1OptTheta = sae1OptTheta[0:hiddenSizeL1 * (inputSize + 1)]

Theta1 = sae1OptTheta[0:hiddenSizeL1 * (inputSize + 1)].reshape(hiddenSizeL1, inputSize + 1)

displayData(Theta1[:,1:])

input('\nProgram paused. Press enter to continue.\n')

# -------------------------------------------------------------------------

##======================================================================
## STEP 2: Train the second sparse autoencoder
#  This trains the second sparse autoencoder on the first autoencoder
#  featurse.
#  If you've correctly implemented sparseAutoencoderCost.m, you don't need
#  to change anything here.

sae1Features = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, inputSize, trainData)

#  Randomly initialize the parameters
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1)

## ---------------------- YOUR CODE HERE  ---------------------------------
#  Instructions: Train the second layer sparse autoencoder, this layer has
#                an hidden size of "hiddenSizeL2" and an inputsize of
#                "hiddenSizeL1"
#
#                You should store the optimal parameters in sae2OptTheta

costFunc = lambda p: sparseAutoencoderCost(p, hiddenSizeL1, hiddenSizeL2, _lambda, sparsityParam, beta, sae1Features.T)
res = minimize(costFunc, sae2Theta, method='L-BFGS-B', jac=True, bounds=None, options={'maxiter': maxIter, 'disp':True})
sae2OptTheta = res.x
sae2OptTheta = sae2OptTheta[0:hiddenSizeL2 * (hiddenSizeL1 + 1)]

input('\nProgram paused. Press enter to continue.\n')

# -------------------------------------------------------------------------

##======================================================================
## STEP 3: Train the softmax classifier
#  This trains the sparse autoencoder on the second autoencoder features.
#  If you've correctly implemented softmaxCost.m, you don't need
#  to change anything here.

sae2Features = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, hiddenSizeL1, sae1Features)

#  Randomly initialize the parameters
saeSoftmaxTheta = 0.005 * np.random.randn(numClasses * hiddenSizeL2)


## ---------------------- YOUR CODE HERE  ---------------------------------
#  Instructions: Train the softmax classifier, the classifier takes in
#                input of dimension "hiddenSizeL2" corresponding to the
#                hidden layer size of the 2nd layer.
#
#                You should store the optimal parameters in saeSoftmaxOptTheta
#
#  NOTE: If you used softmaxTrain to complete this part of the exercise,
#        set saeSoftmaxOptTheta = softmaxModel.optTheta(:);

costFunc = lambda p: softmaxCost(p, numClasses, hiddenSizeL2, _lambda, sae2Features, trainLabels)
res = minimize(costFunc, saeSoftmaxTheta, method='L-BFGS-B', jac=True, bounds=None, options={'maxiter': maxIter, 'disp':True})
saeSoftmaxOptTheta = res.x

# -------------------------------------------------------------------------

##======================================================================
## STEP 5: Finetune softmax model

# Implement the stackedAECost to give the combined cost of the whole model
# then run this cell.

# Initialize the stack using the parameters learned

netconfig = []
netconfig.append([inputSize, hiddenSizeL1, hiddenSizeL1*(inputSize+1)])
netconfig.append([hiddenSizeL1, hiddenSizeL2, hiddenSizeL2*(hiddenSizeL1+1)])
stackedAETheta = np.append(sae1OptTheta, sae2OptTheta)
stackedAETheta = np.append(stackedAETheta, saeSoftmaxOptTheta)

## ---------------------- YOUR CODE HERE  ---------------------------------
#  Instructions: Train the deep network, hidden size here refers to the '
#                dimension of the input to the classifier, which corresponds
#                to "hiddenSizeL2".
#
#

checkStackedAECost()

costFunc = lambda p: stackedAECost(p, inputSize, hiddenSizeL2, numClasses, netconfig, _lambda, trainData.T, trainLabels)
res = minimize(costFunc, stackedAETheta, method='L-BFGS-B', jac=True, bounds=None, options={'maxiter': maxIter, 'disp':True})
stackedAEOptTheta = res.x

# -------------------------------------------------------------------------


##======================================================================
## STEP 6: Test
#  Instructions: You will need to complete the code in stackedAEPredict.m
#                before running this part of the code
#

# Get labelled test images
# Note that we apply the same kind of preprocessing as the training set
testData = loadMNISTImages('t10k-images.idx3-ubyte')
testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte')

pred = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, numClasses, netconfig, testData.T)

acc = np.mean(np.double(np.argmax(pred, axis=0) == testLabels))
print('Accuracy: %0.3f%%\n'%(acc * 100))

pred = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, numClasses, netconfig, testData.T)

acc = np.mean(np.double(np.argmax(pred, axis=0) == testLabels))
print('Accuracy: %0.3f%%\n'%(acc * 100))

input('\nProgram paused. Press enter to continue.\n')

# Accuracy is the proportion of correctly classified images
# The results for our implementation were:
#
# Before Finetuning Test Accuracy: 87.7%
# After Finetuning Test Accuracy:  97.6%
#
# If your values are too low (accuracy less than 95%), you should check
# your code for errors, and make sure you are training on the
# entire data set of 60000 28x28 training images
# (unless you modified the loading code, this should be the case)