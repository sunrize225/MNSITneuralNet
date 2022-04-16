import numpy as np
# functions for neural net
# derivatives are defined only for one value because they are taken with respect to only one value


# inputs are numpy arrays
def meanSquaredError(yArray, yHatArray):
    return np.mean((yHatArray - yArray)**2) 

# pC/pa
# defined for a single value
def meanSquaredError_deriv(y, yHat):
    return 2 * (yHat - y)

# y and yHat are a numpy array
# defined for an array of values
def binaryCrossEntropy(yArray, yHatArray):
    termOne = yArray * np.log(yHatArray)
    termTwo = (1-yHatArray) * np.log( 1 - yHatArray)
    return -1 * np.mean(termOne + termTwo)

# pC/pa
# defined for a single value
def binaryCrossEntropy_deriv(y, yHat):
    if y == 1:
        if yHat != 0:
            return -1 / yHat
        else:
            return 1000
    else: 
        if yHat != 1:
            return 1 / (1 - yHat)
        else:
            return 1000


# defined for an array of values
def ReLu_leaky(zArray):
    return np.where(zArray > 0, zArray, zArray * 0.01)

# pa/pz
# defined for a single value
def ReLu_leaky_deriv(z):
    return np.where(z > 0, 1, 0.01)

# defined for an array of values
def softMax(z):
    # subtract max value to prevent overflow -- does not affect final output values
    output = np.exp(z - np.amax(z))
    output /= np.sum(output)
    return output

# defined for a single value
# the entire z array for the output layer is necessary because its values comprise a constant in the derivative
def softMax__deriv(zarray, z):
    sum = np.exp(zarray - np.amax(zarray))
    sum = np.sum(zarray)
    return np.exp(z - np.amax(zarray)) / sum