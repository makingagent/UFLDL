import numpy as np
from sigmoid import sigmoid


def feedForwardAutoencoder(theta, hiddenSize, visibleSize, data):

    Theta1 = theta[0:hiddenSize * (visibleSize + 1)].reshape(hiddenSize, visibleSize + 1)

    a1 = np.insert(data, 0, 1, axis=0)
    a2 = sigmoid(np.dot(Theta1, a1))

    return a2