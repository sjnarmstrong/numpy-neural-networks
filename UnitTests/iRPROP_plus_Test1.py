import numpy as np
from NeuralNetworkClass import NeuralNetwork
from RPROP import iRProp_plus_batch

##Binarry counter test
inputs=[[[0],[0],[0]],[[0],[0],[1]],[[0],[1],[0]],[[0],[1],[1]],[[1],[0],[0]],[[1],[0],[1]],[[1],[1],[0]],[[1],[1],[1]]]
outputs=inputs[1:]+[inputs[0]]

inputs=np.array(inputs)
outputs=np.array(outputs)
          
a=NeuralNetwork([3,100,3])
b=iRProp_plus_batch(a)

for loop in range(1000):
    for i,o in zip(inputs,outputs):
        b.updateDE_Dw(i,o)
        print(NeuralNetwork.getError(b.neuralNetwork.forwardPropagate(i),o))
    b.updateWeights()
    