import numpy as np
import matplotlib.pyplot as plt

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
    
    def ReLu(self, x, deriv=False):
        if not deriv:
            return np.maximum(0,x)
        else:
            if x>0:
                return 1.0
            return 0
    
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
        i = ob[pr.tolist().index(1)]
        if not deriv:
            if i != 0: 
                return -1 * np.log(i)
            return 1.0
        if i != 0:
            return  -1 * 1/i
        return 1.0

    # runs input through neural net and returns an array for output
    # Hidden layers use Relu, output layer uses softmax
    def feedFoward(self, input, prop=False):
        if not prop:
            for w,b in zip(self.weights[:-1], self.biases[:-1]):
                input = self.ReLu(np.dot(input, w.T) + b)
            # intermediate variable i for reducing z value to prevent overflow error
            i = np.dot(input, self.weights[-1].T) + self.biases[-1]
            self.output = self.softMax(i)
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
            i = np.exp(z[-1] - np.amax(z[-1]))
            i /= np.sum(i)
            a.append(self.softMax(z[-1])) 
            self.output = a[-1]
            return z, a
        
    def backPropagation(self, input):
        # empty array for changes in weights and biases
        dWeights = [np.zeros(784),np.zeros((16,784)),np.zeros((16,16)),np.zeros((10,16))]
        dBiases = [np.zeros(x) for x in self.layers]
        z, a = self.feedFoward(self.input[input], True)

        # Now we will find the derivative of the cost function with respect to the last layer of weights and biases
        # The derivative of the cost function with respect to a last layer weight is: (Where d is a partial derivative)
        # Notice that da/dz is equal to a() because the derivative e^x is e^x
        # dC/dw = dC/da * da/dz * dz/dw | Where dC/da = dCost(a) | da/dz= e^z / sum = output layer | dz/dw = previous neuron
        # The derivative of the cost function with respect to bias:
        # dC/db = dC/da * da/dz * dz/db | Where dC/da = dCost(a) | da/dz= e^z / sum = output layer | dz/db = 1

        # for the last layer
        # We only need to change the output neuron that is supposed to be 1 because all other derivatives are 0
        correctLabelIndex = self.labels[input].tolist().index(1)
        for j in range(self.layers[-2]):
            dWeights[-1][correctLabelIndex][j] = self.cost(self.output, self.labels[input], True) \
        * a[-1][correctLabelIndex] * a[-2][j]
        dBiases[-1][correctLabelIndex] = self.cost(self.output, self.labels[input], True) * a[-1][correctLabelIndex]

        # one variable for same value to reduce function calls
        dCda = self.cost(self.output, self.labels[input], True) 
        # Layer -2 ... dC/da has changed as each neuron in the hidden layers influences 
        # each neuron on the output layer. Therefore we take the sum of each dC/dw and dC/db
        for i in range(self.layers[-2]):
            # for each weight attached to neuron
            dCdz = dCda * self.ReLu(z[-2][i], True)
            for j in range(self.layers[-3]):
                dWeights[-2][i][j] += dCdz * a[-3][j]
            # biase derivatives
            dBiases[-2][i] += dCdz
        
        dCda = self.cost(self.output, self.labels[input], True) 
        # Layer -3 ... dC/da has changed again as each neuron influences each neuron in the next layer which influences 
        # each output. Therefore, dC/da =
        for i in range(self.layers[-3]):
            # for each weight attached to neuron
            dCdz = dCda * self.ReLu(z[-3][i], True)
            for j in range(self.layers[-4]):
                dWeights[-3][i][j] +=  dCdz * a[-4][j]
            # biase derivatives
            dBiases[-3][i] += dCdz
        
        return dWeights, dBiases

    def train(self, seed=0, batchSize=64, numBatches=100, learningRate=0.5, showResults=False):
        results = []
        for y in range(numBatches):
            print(f"Batch #{y+1}")
            avW = [np.zeros(784),np.zeros((16,784)),np.zeros((16,16)),np.zeros((10,16))]
            avB = [np.zeros(x) for x in self.layers]
            # Returns derivative of cost function with respect to weights and biases based on given input
            for x in range((y*batchSize),(y*batchSize)+batchSize-1):
                dW, dB = self.backPropagation(x)
                for i,z in enumerate(dW):
                    avW[i] += z / batchSize * learningRate 
                    avB[i] += dB[i] / batchSize * learningRate
            # adds average results from batch to weights and biases
            for x in range(1,len(self.layers)):
                self.biases[-x] -= avB[-x]
                self.weights[-x] -= avW[-x]
            if showResults:
                results.append(self.test(1000,0,"plot"))
        if showResults:
            plt.plot(range(1,numBatches+1),results)
            plt.show()

    def test(self, numTests=100, seed=0, type="print"):
        avg = 0
        for x in range(seed, numTests+seed):
            self.feedFoward(self.input[x])
            avg+= np.sum(self.cost(self.output,self.labels[x]))
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
#NN.loadModel()
NN.train(0,128,10,0.025, True)
NN.test(1000)
NN.saveModel()
