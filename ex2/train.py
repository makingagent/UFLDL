## CS294A/CS294W Programming Assignment Starter Code

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sampleIMAGES import sampleIMAGES
from displayData import displayData
from initializeParameters import initializeParameters
from sparseAutoencoderCost import sparseAutoencoderCost
from sigmoid import sigmoid

plt.ion()

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  programming assignment. You will need to complete the code in sampleIMAGES.m,
#  sparseAutoencoderCost.m and computeNumericalGradient.m. 
#  For the purpose of completing the assignment, you do not need to
#  change the code in this file. 
#
##======================================================================
## STEP 0: Here we provide the relevant parameters values that will
#  allow your sparse autoencoder to get good filters; you do not need to 
#  change the parameters below.

visibleSize = 28 * 28   # number of input units 
hiddenSize = 196        # number of hidden units 
sparsityParam = 0.1     # desired average activation of the hidden units.
_lambda = 3e-3          # weight decay parameter
beta = 3                # weight of sparsity penalty term

##======================================================================
## STEP 1: Implement sampleIMAGES
#
#  After implementing sampleIMAGES, the display_network command should
#  display a random sample of 200 patches from the dataset

patches = sampleIMAGES()
displayData(patches[:,np.random.randint(10000, size=100)].T)

input('\nProgram paused. Press enter to continue.\n')

##======================================================================
## STEP 2: After verifying that your implementation of
#  sparseAutoencoderCost is correct, You can start training your sparse
#  autoencoder with minFunc (L-BFGS).

#  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, visibleSize)

costFunc = lambda p: sparseAutoencoderCost(p, visibleSize, hiddenSize, _lambda, sparsityParam, beta, patches.T)
res = minimize(costFunc, theta, method='L-BFGS-B', jac=True, bounds=None, options={'maxiter': 600, 'disp':True})
theta = res.x

Theta1 = theta[0:hiddenSize * (visibleSize + 1)].reshape(hiddenSize, visibleSize + 1)

displayData(Theta1[:,1:])

input('\nProgram paused. Press enter to continue.\n')