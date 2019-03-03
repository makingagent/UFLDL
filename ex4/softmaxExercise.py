## CS294A/CS294W Softmax Exercise

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mnist import loadMNISTImages, loadMNISTLabels
from softmaxCost import softmaxCost
from computeNumericalGradient import computeNumericalGradient

plt.ion()

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  softmax exercise. You will need to write the softmax cost function
#  in softmaxCost.m and the softmax prediction function in softmaxPred.m.
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#  (However, you may be required to do so in later exercises)

##======================================================================
## STEP 0: Initialise constants and parameters
#
#  Here we define and initialise some constants which allow your code
#  to be used more generally on any arbitrary input.
#  We also initialise some parameters used for tuning the model.

inputSize = 28 * 28 # Size of input vector (MNIST images are 28x28)
numClasses = 10     # Number of classes (MNIST images fall into 10 classes)

_lambda = 1e-4 # Weight decay parameter

##======================================================================
## STEP 1: Load data
#
#  In this section, we load the input and output data.
#  For softmax regression on MNIST pixels,
#  the input data is the images, and
#  the output data is the labels.
#

# Change the filenames if you've saved the files under different names
# On some platforms, the files might be saved as
# train-images.idx3-ubyte / train-labels.idx1-ubyte

images = loadMNISTImages("train-images-idx3-ubyte")
labels = loadMNISTLabels("train-labels-idx1-ubyte")
#labels[np.where(labels==0)] = 10 # Remap 0 to 10

inputData = images

# For debugging purposes, you may wish to reduce the size of the input data
# in order to speed up gradient checking.
# Here, we create synthetic dataset using random data for testing

DEBUG = False # Set DEBUG to true when debugging.
if DEBUG:
    inputSize = 8
    inputData = np.random.randn(8, 100)
    labels = np.random.randint(10, size=100)

# Randomly initialise theta
theta = 0.005 * np.random.randn(numClasses * inputSize)

##======================================================================
## STEP 2: Implement softmaxCost
#
#  Implement softmaxCost in softmaxCost.m.

cost, grad = softmaxCost(theta, numClasses, inputSize, _lambda, inputData, labels)

##======================================================================
## STEP 3: Gradient checking
#
#  As with any learning algorithm, you should always check that your
#  gradients are correct before learning the parameters.
#

if DEBUG:
    costFunc = lambda p: softmaxCost(p, numClasses, inputSize, _lambda, inputData, labels)

    numgrad = computeNumericalGradient(costFunc, theta)
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print(diff)
    # The difference should be small.
    # In our implementation, these values are usually less than 1e-7.

    # When your gradients are correct, congratulations!

##======================================================================
## STEP 4: Learning parameters
#
#  Once you have verified that your gradients are correct,
#  you can start training your softmax regression code using softmaxTrain
#  (which uses minFunc).

costFunc = lambda p: softmaxCost(p, numClasses, inputSize, _lambda, inputData, labels)
res = minimize(costFunc, theta, method='L-BFGS-B', jac=True, bounds=None, options={'maxiter': 100, 'disp':True})
theta = res.x

# Although we only use 100 iterations here to train a classifier for the
# MNIST data set, in practice, training for more iterations is usually
# beneficial.

##======================================================================
## STEP 5: Testing
#
#  You should now test your model against the test images.
#  To do this, you will first need to write softmaxPredict
#  (in softmaxPredict.m), which should return predictions
#  given a softmax model and the input data.

images = loadMNISTImages('t10k-images.idx3-ubyte')
labels = loadMNISTLabels('t10k-labels.idx1-ubyte')
#labels(labels==0) = 10 # Remap 0 to 10

inputData = images

theta = theta.reshape((numClasses, inputSize))

a = np.dot(theta, inputData)
a = a - np.amax(a, axis=0)
e = np.exp(a)
a = e / np.sum(e, axis=0)

acc = np.mean(np.double(np.argmax(a, axis=0) == labels))
print('Accuracy: %0.3f%%\n'%(acc * 100))

# Accuracy is the proportion of correctly classified images
# After 100 iterations, the results for our implementation were:
#
# Accuracy: 92.200%
#
# If your values are too low (accuracy less than 0.91), you should check
# your code for errors, and make sure you are training on the
# entire data set of 60000 28x28 training images
# (unless you modified the loading code, this should be the case)

input('\nProgram paused. Press enter to continue.\n')
