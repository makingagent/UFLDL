import numpy as np
from sigmoid import sigmoid

def stackedAECost(theta, inputSize, hiddenSize, numClasses, netconfig, _lambda, data, labels):

    Theta1 = theta[0:netconfig[0][2]].reshape(netconfig[0][1], netconfig[0][0] + 1)
    Theta2 = theta[netconfig[0][2]:netconfig[0][2]+netconfig[1][2]].reshape(netconfig[1][1], netconfig[1][0] + 1)

    m = data.shape[0] # number of training examples

    a1 = np.insert(data, 0, 1, axis=1)
    a2 = sigmoid(np.dot(a1, Theta1.T))

    a2 = np.insert(a2, 0, 1, axis=1)
    a3 = sigmoid(np.dot(a2, Theta2.T))

    Theta3 = theta[netconfig[0][2]+netconfig[1][2]:].reshape(numClasses, hiddenSize)

    a = np.dot(Theta3, a3.T)
    a = a - np.amax(a, axis=0)
    e = np.exp(a)
    a = e / np.sum(e, axis=0)

    gtm = np.zeros((numClasses, m), dtype=int)
    gtm[labels, np.arange(m)] = 1

    J = -np.sum(gtm*np.log(a)) / m \
        + _lambda * np.sum(Theta3*Theta3) / 2

    Theta3_grad = -np.dot(gtm-a,a3) / m + _lambda * Theta3

    d3 = -np.dot(Theta3.T, gtm-a).T * a3 * (1.0 - a3)
    d2 = np.dot(d3, Theta2)[:,1:] * a2[:,1:] * (1.0 - a2[:,1:])
    delta1 = np.dot(d2.T, a1)
    delta2 = np.dot(d3.T, a2)

    Theta1_grad = delta1 / m
    Theta2_grad = delta2 / m

    grad = np.append(Theta1_grad.flatten(), Theta2_grad.flatten())
    grad = np.append(grad, Theta3_grad.flatten())

    return J, grad.flatten()