from handwritingNeuralNet import * 
# 784 is the number of inputs; one for each pixel (28x28 = 784)
# There are two hidden layers and 10 outputs for each number 0-9
def main():
    layers = [784, 16, 16, 10]
    data = Images()
    data.loadData()

    NN = Network(layers)
    NN.loadTrainingData(data.images, data.labels)
    NN.loadModel()
    NN.train(0,32,10,0.025, True)
    NN.test(1000)
    NN.saveModel()

if __name__ == "__main__":
    main()