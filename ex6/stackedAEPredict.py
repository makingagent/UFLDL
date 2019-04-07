import numpy as np
from sigmoid import sigmoid


def stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data):

    Theta1 = theta[0:netconfig[0][2]].reshape(netconfig[0][1], netconfig[0][0] + 1)
    Theta2 = theta[netconfig[0][2]:netconfig[0][2]+netconfig[1][2]].reshape(netconfig[1][1], netconfig[1][0] + 1)

    a1 = np.insert(data, 0, 1, axis=1)
    a2 = sigmoid(np.dot(a1, Theta1.T))

    a2 = np.insert(a2, 0, 1, axis=1)
    a3 = sigmoid(np.dot(a2, Theta2.T))

    Theta3 = theta[netconfig[0][2]+netconfig[1][2]:].reshape(numClasses, hiddenSize)

    a = np.dot(Theta3, a3.T)
    a = a - np.amax(a, axis=0)
    e = np.exp(a)
    a = e / np.sum(e, axis=0)

    return a