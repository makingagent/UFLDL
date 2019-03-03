import numpy as np

def softmaxCost(theta, numClasses, inputSize, _lambda, data, labels):

    # numClasses - the number of classes
    # inputSize - the size N of the input vector
    # lambda - weight decay parameter
    # data - the N x M input matrix, where each column data(:, i) corresponds to
    #        a single test set
    # labels - an M x 1 matrix containing the labels corresponding for the input data
    #

    # Unroll the parameters from theta
    theta = theta.reshape((numClasses, inputSize))
    m = data.shape[1]

    a = np.dot(theta, data)
    a = a - np.amax(a, axis=0)
    e = np.exp(a)
    a = e / np.sum(e, axis=0)

    gtm = np.zeros((numClasses, m), dtype=int)
    gtm[labels, np.arange(m)] = 1

    J = -np.sum(gtm*np.log(a)) / m + _lambda * np.sum(theta*theta) / 2
    grad = -np.dot(gtm-a,data.T) / m + _lambda * theta

    return J, grad.flatten()

