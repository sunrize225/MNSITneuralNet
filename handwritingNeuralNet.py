import numpy as np
import matplotlib.pyplot as plt
import netFunctions as nf

class Images:
    def __init__(self):
        self.images = []
        self.labels = []

    def __loadImages(self,x):
        with open("train-images.idx3-ubyte","rb") as f:
            f.read(16)
            return np.frombuffer(f.read(x*784), dtype="uint8").reshape((x,28,28))

    def __loadLabels(self,x):
        with open("train-labels.idx1-ubyte","rb") as f:
            f.read(8)
            return np.frombuffer(f.read(x), dtype="uint8")
    
    def loadData(self,a=60000):
        # Loads in dataset of 28 x 28 images
        # of handwritten numbers 0-9
        # each pixel has a grayscale value of 
        # 0-255, but is scaled down to 0.0-1.0
        images = self.__loadImages(a) / 255
        labels = self.__loadLabels(a)

        self.labels = np.zeros((a,10))
        for x in range(0,a):
            self.labels[x][labels[x]] = 1

        self.input = []
        for x in range(0,a):
            self.images.append(images[x].reshape(784))

class Network:
    def __init__(self, layers):
        self.layers = layers
        self.input = []
        self.labels = []
        # Weights and biases loaded in as random numbers with a normal distribution
        self.biases  = [np.random.randn(x) for x in layers[1:]]
        self.weights = [np.random.randn(x,y) for x,y in zip(layers[1:],layers[:-1])]
        self.output = [np.zeros(x) for x in layers[1:]]

    def loadTrainingData(self, input, labels): 
        self.input = input
        self.labels = labels

    def saveModel(self):
        lines = []
        layerString = [str(x) for x in self.layers]
        layerString = " ".join(layerString)+("\n")
        lines.append(layerString)
        weights = [str(weight) for layer in self.weights for neuron in layer for weight in neuron]
        weights = " ".join(weights)+("\n")
        lines.append(weights)
        biases = [str(biase) for layer in self.biases for biase in layer]
        biases = " ".join(biases)+("\n")
        lines.append(biases)
        with open("model.txt", "w") as f:
            f.writelines(lines)
    
    def loadModel(self):
        with open("model.txt", "r") as f:
            model = f.read().split("\n")
        self.layers = [int(i) for i in model[0].split(" ")]
        weights = model[1].split(" ")
        biases = model[2].split(" ")
        for n,k in enumerate(self.layers[:-1]):
            for i in range(self.layers[n+1]):
                for j in range(k):
                    # calculate index to start on
                    q = 0
                    if n !=0:
                        for l in range(n):
                            q += self.layers[l] * self.layers[l+1]
                    self.weights[n][i][j] = float(weights[q+i*k+j])
                q = 0
                for l in range(n):
                    q+= self.layers[l+1]
                self.biases[n][i] = float(biases[q+i])

    # using leaky relu
    def ReLu(self, x, deriv=False):
        if not deriv:
            rList = []
            for y in x:
                if y>0:
                    rList.append(y)
                else:
                    rList.append(0.01 * y)
            return np.array(rList)
        else:
            if x>0:
                return 1.0
            return 0.01
    
    # entire z output array as input, as the sum of output is needed for function 
    # No derivative as d/dx e^x = e^x
    def softMax(self, z):
        # subtract max value to prevent overflow -- does not affect final output values
        output = np.exp(z - np.amax(z))
        output /= np.sum(output)
        return output

    # calculates error/cost from ob (observed) and pr (predicted)
    # Also includes option for derivative with respect to observed
    # This is the cost for the entire output layer, not each individual output neuron
    # As such, the ob and pr parameters should be an array
    def cost(self, ob, pr, deriv=False):
        # binary cross entropy cost function
        if not deriv:
            return nf.binaryCrossEntropy(pr, ob)
        else:
            return nf.binaryCrossEntropy_deriv(pr, ob)


    # runs input through neural net and returns an array for output
    # Hidden layers use Relu, output layer uses softmax
    def feedFoward(self, input, prop=False):
        if not prop:
            for w,b in zip(self.weights[:-1], self.biases[:-1]):
                input = nf.ReLu_leaky(np.dot(input, w.T) + b)
                if np.isnan(input[0]): # DELETE LATER ===================
                    raise Exception("Input value is nan")
            # intermediate variable i for reducing z value to prevent overflow error
            i = np.dot(input, self.weights[-1].T) + self.biases[-1]
            self.output = self.softMax(i)
            if np.isnan(self.output[0]): # DELETE LATER ==================
                raise Exception("Output values are nan")
        else:
            # z is output before activation function
            # a is output after activation function
            # first layer of a and z is input
            z = []
            a = []
            a.append(input)
            z.append(input)
            for w,b in zip(self.weights[:-1], self.biases[:-1]):
                z.append(np.dot(a[-1], w.T) + b)
                a.append(self.ReLu(z[-1]))
            z.append(np.dot(a[-1], self.weights[-1].T) + self.biases[-1])
            # i is simply an intermediate variable used for calculation
            # We need to subtract the max value of z from all others to prevent
            # an overflow error
            a.append(self.softMax(z[-1])) 
            self.output = a[-1]
            return z, a
      
    def backPropagation(self, input):
        dWeights = [np.zeros(784),np.zeros((16,784)),np.zeros((16,16)),np.zeros((10,16))]
        dBiases = [np.zeros(x) for x in self.layers]
        z, a = self.feedFoward(self.input[input], True)

        # Now we will find the derivative of the cost function with respect to the last layer of weights and biases
        # The derivative of the cost function with respect to a last layer weight is: (Where d is a partial derivative)
        # Notice for da/dz derivative is same as function
        # dC/dw = dC/da * da/dz * dz/dw | Where dC/da = dCost(a) | da/dz= e^z / sum = output layer | dz/dw = previous neuron
        # The derivative of the cost function with respect to bias:
        # dC/db = dC/da * da/dz * dz/db | Where dC/da = dCost(a) | da/dz= e^z / sum = output layer | dz/db = 1

        # for last layer
        for i in range(self.layers[-1]):
            pCpa = nf.binaryCrossEntropy_deriv(self.labels[input][i], a[-1][i])
            papz = nf.softMax__deriv(z[-1], z[-1][i])
            for j in range(self.layers[-2]):
                dWeights[-1][i][j] = pCpa * papz * a[-2][j]
            dBiases[-1][i] = pCpa * papz
        
        # for second to last layer
        for i in range(self.layers[-2]):
            # pCpa is depedent on all the output neurons as any given neuron
            # in the second to last layer directly influences every neuron in the output layer
            pCpa = 0
            for j in range(self.layers[-1]):
                n = nf.binaryCrossEntropy_deriv(self.labels[input][j], a[-1][j]) * nf.softMax__deriv(z[-1], z[-1][j]) * self.weights[-2][j][i]
                pCpa += n
            pCpa /= self.layers[-1] # may not be necessary
            papz = nf.ReLu_leaky_deriv(z[-2][i])
            for j in range(self.layers[-3]):
                dWeights[-2][i][j] = pCpa * papz * a[-3][j]
            dBiases[-2][i] = pCpa * papz
        
        # for third to last layer
        for i in range(self.layers[-3]):
            pCpa = 0
            for j in range(self.layers[-2]):
                for k in range(self.layers[-1]):
                    n = nf.binaryCrossEntropy_deriv(self.labels[input][k], a[-1][k]) * nf.softMax__deriv(z[-1], z[-1][k]) * self.weights[-2][k][j] * nf.ReLu_leaky_deriv(z[-2][j]) * self.weights[-3][j][i]
                    pCpa += n
            pCpa /= self.layers[-2] * self.layers[-1] # may not be necessary
            papz = nf.ReLu_leaky_deriv(z[-2][i])
            for j in range(self.layers[-4]):
                dWeights[-3][i][j] = pCpa * papz * a[-4][j]
            dBiases[-3][i] = pCpa * papz

        return dWeights, dBiases

    def train(self, seed=0, batchSize=64, numBatches=100, learningRate=0.5, showResults=False):
        results = []
        for y in range(numBatches):
            print(f"Batch #{y+1}")
            avW = [np.zeros(784),np.zeros((16,784)),np.zeros((16,16)),np.zeros((10,16))]
            avB = [np.zeros(x) for x in self.layers]
            # Returns derivative of cost function with respect to weights and biases based on given input
            for x in range((y*batchSize),(y*batchSize)+batchSize-1):
                dW, dB = self.backPropagation(seed+x)
                for i,z in enumerate(dW):
                    avW[i] += (z / batchSize) * learningRate 
                    avB[i] += (dB[i] / batchSize) * learningRate
            # adds average results from batch to weights and biases
            for x in range(1,len(self.layers)):
                self.biases[-x] -= avB[-x]
                self.weights[-x] -= avW[-x]
            if showResults:
                results.append(self.test(100,0,"plot"))
        if showResults:
            plt.plot(range(1,numBatches+1),results)
            plt.show()

    def test(self, numTests=100, seed=0, type="print"):
        avg = 0
        for x in range(seed, numTests+seed):
            self.feedFoward(self.input[x])
            avg += self.cost(self.output,self.labels[x])
        avg /= numTests
        if type=="print":
            print(f"Average cost over {numTests} trials: {avg}")
        if type=="plot":
            return avg
    
    def test2(self, num):
        samples = np.random.randint(0, len(self.input), num)
        avg = 0
        for n in range(num):
            self.feedFoward(self.input[samples[n]])
            answer = np.argmax(self.labels[samples[n]])
            result = np.argmax(self.output)
            print(f"Input: {answer} Output: {result} ")
            if answer == result:
                avg += 1
        print(f"Trials attempted {n}. Trials correct: {avg} Percent correct: {(avg/num)*100}%")
# 784 is the number of inputs; one for each pixel (28x28 = 784)
# There are two hidden layers and 10 outputs for each number 0-9
layers = [784, 16, 16, 10]
data = Images()
data.loadData()

NN = Network(layers)
NN.loadTrainingData(data.images, data.labels)
NN.loadModel()
NN.train(49000,64,10,10000000000, True)
NN.test(1000)
NN.saveModel()
