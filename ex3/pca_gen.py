import numpy as np
import matplotlib.pyplot as plt
from sampleIMAGESRAW import sampleIMAGESRAW
from displayData import displayData

plt.ion()

##================================================================
## Step 0a: Load data
#  Here we provide the code to load natural image data into x.
#  x will be a 144 * 10000 matrix, where the kth column x(:, k) corresponds to
#  the raw image data from the kth 12x12 image patch sampled.
#  You do not need to change the code below.

x = sampleIMAGESRAW()
randsel = np.random.randint(10000, size=36)
displayData(x[:,randsel].T)

input('Program paused. Press enter to continue.')

##================================================================
## Step 0b: Zero-mean the data (by row)
#  You can make use of the mean and repmat/bsxfun functions.

x = x - np.mean(x, axis=1).reshape(-1,1)

##================================================================
## Step 1a: Implement PCA to obtain xRot
#  Implement PCA to obtain xRot, the matrix in which the data is expressed
#  with respect to the eigenbasis of sigma, which is the matrix U.

xRot = np.zeros(x.shape) # You need to compute this

sigma = np.dot(x, x.T) / x.shape[1]
U, S, v = np.linalg.svd(sigma, full_matrices=True)

xRot = np.dot(U.T, x)

##================================================================
## Step 1b: Check your implementation of PCA
#  The covariance matrix for the data expressed with respect to the basis U
#  should be a diagonal matrix with non-zero entries only along the main
#  diagonal. We will verify this here.
#  Write code to compute the covariance matrix, covar. 
#  When visualised as an image, you should see a straight line across the
#  diagonal (non-zero entries) against a blue background (zero entries).

covar = np.dot(xRot, xRot.T) / xRot.shape[1]

# Visualise the covariance matrix. You should see a line across the
# diagonal against a blue background.
plt.figure()
plt.imshow(covar, cmap='RdYlBu_r')

input('Program paused. Press enter to continue.')

##================================================================
## Step 2: Find k, the number of components to retain
#  Write code to determine k, the number of components to retain in order
#  to retain at least 99# of the variance.

# -------------------- YOUR CODE HERE -------------------- 
k = 0 # Set k accordingly

sub = 0
sum = np.sum(S)
for i in range(S.shape[0]):
    k = k + 1
    sub += S[i]
    if sub / sum >= 0.90:
        break

##================================================================
## Step 3: Implement PCA with dimension reduction
#  Now that you have found k, you can reduce the dimension of the data by
#  discarding the remaining dimensions. In this way, you can represent the
#  data in k dimensions instead of the original 144, which will save you
#  computational time when running learning algorithms on the reduced
#  representation.
# 
#  Following the dimension reduction, invert the PCA transformation to produce 
#  the matrix xHat, the dimension-reduced data with respect to the original basis.
#  Visualise the data and compare it to the raw data. You will observe that
#  there is little loss due to throwing away the principal components that
#  correspond to dimensions with low variation.

xHat = np.dot(U.T[0:k,:].T,np.dot(U.T[0:k,:], x))
plt.figure()
displayData(x[:,randsel].T)
plt.figure()
displayData(xHat[:,randsel].T)

input('Program paused. Press enter to continue.')

##================================================================
## Step 4a: Implement PCA with whitening and regularisation
#  Implement PCA with whitening and regularisation to produce the matrix
#  xPCAWhite. 

epsilon = 0.1
xPCAWhite = np.dot(np.diag(1/np.sqrt(S+epsilon)), np.dot(U.T, x))

##================================================================
## Step 4b: Check your implementation of PCA whitening 
#  Check your implementation of PCA whitening with and without regularisation. 
#  PCA whitening without regularisation results a covariance matrix 
#  that is equal to the identity matrix. PCA whitening with regularisation
#  results in a covariance matrix with diagonal entries starting close to 
#  1 and gradually becoming smaller. We will verify these properties here.
#  Write code to compute the covariance matrix, covar. 
#
#  Without regularisation (set epsilon to 0 or close to 0), 
#  when visualised as an image, you should see a red line across the
#  diagonal (one entries) against a blue background (zero entries).
#  With regularisation, you should see a red line that slowly turns
#  blue across the diagonal, corresponding to the one entries slowly
#  becoming smaller.

covar = np.dot(xPCAWhite, xPCAWhite.T) / xPCAWhite.shape[1]

# Visualise the covariance matrix. You should see a red line across the
# diagonal against a blue background.
plt.figure()
plt.imshow(covar, cmap='RdYlBu_r')

input('Program paused. Press enter to continue.')

##================================================================
## Step 5: Implement ZCA whitening
#  Now implement ZCA whitening to produce the matrix xZCAWhite. 
#  Visualise the data and compare it to the raw data. You should observe
#  that whitening results in, among other things, enhanced edges.

xZCAwhite = np.dot(U, xPCAWhite)
plt.figure()
displayData(xZCAwhite[:,randsel].T)

input('Program paused. Press enter to continue.')