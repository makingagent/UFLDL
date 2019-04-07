import numpy as np

def initializeParameters(hiddenSize, visibleSize):
    
    # Initialize parameters randomly based on layer sizes.
    epsilon = np.sqrt(6) / np.sqrt(hiddenSize+visibleSize+1)
    W1 = np.random.rand(hiddenSize,visibleSize) * 2 * epsilon - epsilon
    W2 = np.random.rand(visibleSize,hiddenSize) * 2 * epsilon - epsilon
    W1 = np.insert(W1, 0, 0, axis=1)
    W2 = np.insert(W2, 0, 0, axis=1)

    return np.append(W1.flatten(), W2.flatten())