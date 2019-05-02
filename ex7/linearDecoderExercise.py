## CS294A/CS294W Linear Decoder Exercise

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sparseAutoencoderLinearCost import sparseAutoencoderLinearCost
from initializeParameters import  initializeParameters
from computeNumericalGradient import computeNumericalGradient
from displayColorNetwork import displayColorNetwork
import scipy.io as sio

plt.ion()

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear decoder exericse. For this exercise, you will only need to modify
#  the code in sparseAutoencoderLinearCost.m. You will not need to modify
#  any code in this file.

##======================================================================
## STEP 0: Initialization
#  Here we initialize some parameters used for the exercise.

imageChannels = 3       # number of channels (rgb, so 3)

patchDim   = 8          # patch dimension
numPatches = 100000     # number of patches

visibleSize = patchDim * patchDim * imageChannels  # number of input units 
outputSize  = visibleSize   # number of output units
hiddenSize  = 400           # number of hidden units

sparsityParam = 0.035   # desired average activation of the hidden units.
_lambda = 3e-3           # weight decay parameter       
beta = 5                # weight of sparsity penalty term       

epsilon = 0.1	        # epsilon for ZCA whitening

##======================================================================
## STEP 1: Create and modify sparseAutoencoderLinearCost.m to use a linear decoder,
#          and check gradients
#  You should copy sparseAutoencoderCost.m from your earlier exercise 
#  and rename it to sparseAutoencoderLinearCost.m. 
#  Then you need to rename the function from sparseAutoencoderCost to
#  sparseAutoencoderLinearCost, and modify it so that the sparse autoencoder
#  uses a linear decoder instead. Once that is done, you should check 
# your gradients to verify that they are correct.

# NOTE: Modify sparseAutoencoderCost first!

# To speed up gradient checking, we will use a reduced network and some
# dummy patches

debugHiddenSize = 5
debugVisibleSize = 8
patches = np.random.rand(8,10)
theta = initializeParameters(debugHiddenSize, debugVisibleSize);

# Short hand for cost function
costFunc = lambda p: sparseAutoencoderLinearCost(p, debugVisibleSize, debugHiddenSize, _lambda, sparsityParam, beta, patches.T)

cost, grad = costFunc(theta)
numgrad = computeNumericalGradient(costFunc, theta)

# Evaluate the norm of the difference between two solutions.  
# If you have a correct implementation, and assuming you used EPSILON = 0.0001 
# in computeNumericalGradient.m, then diff below should be less than 1e-9
diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)

print('If your backpropagation implementation is correct, then \n \
        the relative difference will be small (less than 1e-9). \n \
        \nRelative Difference: %g\n'%diff)

input('\nProgram paused. Press enter to continue.\n')

##======================================================================
## STEP 2: Learn features on small patches
#  In this step, you will use your sparse autoencoder (which now uses a
#  linear decoder) to learn features on small patches sampled from related
#  images.

## STEP 2a: Load patches
#  In this step, we load 100k patches sampled from the STL10 dataset and
#  visualize them. Note that these patches have been scaled to [0,1]

data = sio.loadmat('stlSampledPatches.mat')
patches = data['patches']

displayColorNetwork(patches[:,0:100])

input('\nProgram paused. Press enter to continue.\n')

## STEP 2b: Apply preprocessing
#  In this sub-step, we preprocess the sampled patches, in particular,
#  ZCA whitening them.
#
#  In a later exercise on convolution and pooling, you will need to replicate
#  exactly the preprocessing steps you apply to these patches before
#  using the autoencoder to learn features on them. Hence, we will save the
#  ZCA whitening and mean image matrices together with the learned features
#  later on.

# Subtract mean patch (hence zeroing the mean of the patches)
meanPatch = np.mean(patches, axis=1).reshape(-1,1)
patches = patches - meanPatch

# Apply ZCA whitening
sigma = patches.dot(patches.T) / numPatches
U, S, v = np.linalg.svd(sigma)
ZCAWhite = U.dot(np.diag(1/np.sqrt(S+epsilon))).dot(U.T)
patches = ZCAWhite.dot(patches)

displayColorNetwork(patches[:,0:100])

input('\nProgram paused. Press enter to continue.\n')

## STEP 2c: Learn features
#  You will now use your sparse autoencoder (with linear decoder) to learn
#  features on the preprocessed patches. This should take around 45 minutes.

theta = initializeParameters(hiddenSize, visibleSize)

# Use minFunc to minimize the function

costFunc = lambda p: sparseAutoencoderLinearCost(p, visibleSize, hiddenSize, _lambda, sparsityParam, beta, patches.T)
res = minimize(costFunc, theta, method='L-BFGS-B', jac=True, bounds=None, options={'maxiter': 400, 'disp':True})

W = res.x[0:hiddenSize * (visibleSize + 1)].reshape(hiddenSize, visibleSize + 1)[:,1:]
displayColorNetwork(np.dot(W,ZCAWhite).T)

input('\nProgram paused. Press enter to continue.\n')