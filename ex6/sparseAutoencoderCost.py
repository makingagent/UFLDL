import numpy as np
from sigmoid import sigmoid
from displayData import displayData
import scipy.stats

def sparseAutoencoderCost(theta, visibleSize, hiddenSize, _lambda, sparsityParam, beta, data):
    
    Theta1 = theta[0:hiddenSize * (visibleSize + 1)].reshape(hiddenSize, visibleSize + 1)
    Theta2 = theta[hiddenSize * (visibleSize + 1):].reshape(visibleSize, hiddenSize + 1)

    m = data.shape[0] # number of training examples

    a1 = np.insert(data, 0, 1, axis=1)
    a2 = sigmoid(np.dot(a1, Theta1.T))

    rho = np.sum(a2, axis=0) / m
    KL = np.sum(sparsityParam*np.log(sparsityParam/rho) + (1.0-sparsityParam)*np.log((1.0-sparsityParam)/(1.0-rho)))

    a2 = np.insert(a2, 0, 1, axis=1)
    a3 = sigmoid(np.dot(a2, Theta2.T))

    regTheta1 = Theta1[:,1:]
    regTheta2 = Theta2[:,1:]

    J = np.sum( (a3-data)*(a3-data) ) / m/2 + \
        _lambda * np.sum(regTheta1*regTheta1) / 2 + \
        _lambda * np.sum(regTheta2*regTheta2) / 2 + \
        beta * KL

    sparsity_delta = -sparsityParam/rho+(1.0-sparsityParam)/(1.0-rho)

    d3 = (a3 - data) * a3 * (1.0 - a3)
    d2 = (np.dot(d3, Theta2)[:,1:] + beta * sparsity_delta) * a2[:,1:] * (1.0 - a2[:,1:])
    delta1 = np.dot(d2.T, a1)
    delta2 = np.dot(d3.T, a2)

    regTheta1 = np.insert(regTheta1, 0, 0, axis=1)
    regTheta2 = np.insert(regTheta2, 0, 0, axis=1)
    Theta1_grad = delta1 / m + _lambda * regTheta1
    Theta2_grad = delta2 / m + _lambda * regTheta2

    grad = np.append(Theta1_grad.flatten(), Theta2_grad.flatten())
    #print('cost value: %lf'%J)
    
    return J, grad
    