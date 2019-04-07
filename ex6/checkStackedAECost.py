import numpy as np
from stackedAECost import stackedAECost
from computeNumericalGradient import computeNumericalGradient

def checkStackedAECost():

    inputSize = 4
    hiddenSize = 5
    _lambda = 0.01
    data = np.random.randn(inputSize, 5)
    labels = np.array([0, 1, 0, 1, 0])
    numClasses = 2

    netconfig = []
    netconfig.append([inputSize, hiddenSize, hiddenSize * (inputSize + 1)])
    netconfig.append([hiddenSize, hiddenSize, hiddenSize * (hiddenSize + 1)])

    sae1OptTheta = np.random.randn(hiddenSize, inputSize+1)
    sae2OptTheta = np.random.randn(hiddenSize, hiddenSize+1)
    saeSoftmaxOptTheta = np.random.randn(numClasses, hiddenSize)

    stackedAETheta = np.append(sae1OptTheta, sae2OptTheta)
    stackedAETheta = np.append(stackedAETheta, saeSoftmaxOptTheta)

    costFunc = lambda p: stackedAECost(p, inputSize, hiddenSize, numClasses, netconfig, _lambda, data.T, labels)

    cost, grad = costFunc(stackedAETheta)
    numgrad = computeNumericalGradient(costFunc, stackedAETheta)

    # Evaluate the norm of the difference between two solutions.
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001
    # in computeNumericalGradient.m, then diff below should be less than 1e-9
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)

    print(diff)