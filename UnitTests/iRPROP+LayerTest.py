import sys
sys.path.append("..")

from Classes.NeuralNetworkClass import NeuralNetwork
from Classes.RPROPLayer import iRProp_plus_Layer
import numpy as np

##Binarry counter test
inputs=[[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
outputs=inputs[1:]+[inputs[0]]

inputs=np.array(inputs)
outputs=np.array(outputs)

layerInfo=[3,10,3]
     
a=NeuralNetwork(layerInfo, layerTypes=[iRProp_plus_Layer]*(len(layerInfo)-1))
#a=NeuralNetwork([3,8,3])

for loop in range(1000):
    learningRate=1.2
    print(a.backwardPropagate(inputs,outputs,learningRate))
        
