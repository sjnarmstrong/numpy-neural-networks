import numpy as np
from NeuralNetworkClass import NeuralNetwork
from RPROP import RProp_minus

##Binarry counter test
inputs=[[[0],[0],[0]],[[0],[0],[1]],[[0],[1],[0]],[[0],[1],[1]],[[1],[0],[0]],[[1],[0],[1]],[[1],[1],[0]],[[1],[1],[1]]]
outputs=inputs[1:]+[inputs[0]]

inputs=np.array(inputs)
outputs=np.array(outputs)
           
a=NeuralNetwork([3,5,5,3])
b=RProp_minus(a)

for loop in range(100000):
    for i,o in zip(inputs,outputs):
        b.updateWeights(i,o)
        print(NeuralNetwork.getError(b.neuralNetwork.forwardPropagate(i),o))