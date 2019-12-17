from NeuralNetworkClass import NeuralNetwork 
import numpy as np

##Binarry counter test
inputs=[[[0],[0],[0]],[[0],[0],[1]],[[0],[1],[0]],[[0],[1],[1]],[[1],[0],[0]],[[1],[0],[1]],[[1],[1],[0]],[[1],[1],[1]]]
outputs=inputs[1:]+[inputs[0]]

inputs=np.array(inputs)
outputs=np.array(outputs)
           
a=NeuralNetwork([3,5,5,3])
#a=NeuralNetwork([3,8,3])

for loop in range(100000):
    learningRate=np.exp(-loop/50000-2)
    for i,o in zip(inputs,outputs):
        print(a.backwardPropagate(i,o,learningRate))