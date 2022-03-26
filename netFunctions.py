import numpy as np
# functions for neural net
# derivatives are defined only for one value because they are taken with respect to only one value


# y and yHat are a numpy array
# defined for an array of values
def binaryCrossEntropy(yArray, yHatArray):
    sum = 0.0
    for y, yHat in zip(yArray, yHatArray):
        if y == 1:
            if yHat != 0:
                sum += -1 * np.log(yHat)
            else:
                sum += 100
        else: 
            if yHat != 1:
                sum += 1 * np.log(1 - yHat)
            else:
                sum += 100
    sum /= yArray.size
    return sum

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
    a = []
    for z in zArray:
        if z > 0:
            a.append(z)
        else:
            a.append(0.01 * z)
    return np.array(a)

# pa/pz
# defined for a single value
def ReLu_leaky_deriv(z):
        if z > 0:
            return 1
        else:
            return 0.01

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
    