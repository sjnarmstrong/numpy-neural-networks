import sys
sys.path.append("..")

from Classes.NeuralNetworkClass import NeuralNetwork 
from Classes.MomentumLayer import NetworkLayerWithMomentum, NetworkLayerWithAdaptiveWeights
import numpy as np

##Binarry counter test
inputs=[[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
outputs=inputs[1:]+[inputs[0]]

inputs=np.array(inputs)
outputs=np.array(outputs)

layerInfo=[3,10,3]
     
a=NeuralNetwork(layerInfo, layerTypes=[NetworkLayerWithMomentum]*(len(layerInfo)-1), Momentum =0.3)
#a=NeuralNetwork([3,8,3])

for loop in range(100):
    learningRate=3
    print(a.backwardPropagate(inputs,outputs,learningRate))
        
a=NeuralNetwork(layerInfo, layerTypes=[NetworkLayerWithAdaptiveWeights]*(len(layerInfo)-1))
#a=NeuralNetwork([3,8,3])
print("________________________________________________________")
for loop in range(0):
    learningRate=1
    print(a.backwardPropagate(inputs,outputs,learningRate))