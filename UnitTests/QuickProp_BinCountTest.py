import numpy as np
from NeuralNetworkClass import batchNeuralNetwork, NeuralNetwork
from QuickProp import QuickPropLayer
##Binarry counter test
inputs=[[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
outputs=inputs[1:]+[inputs[0]]

inputs=np.array(inputs)
outputs=np.array(outputs)

layerInfo=[3,20,3]

qnn=NeuralNetwork(layerInfo, layerTypes=[QuickPropLayer]*(len(layerInfo)-1),mu=1.75)
    
    
stagecounter=0
stage=0
for loop in range(700):
    learningRate=1.2#np.exp(-loop/50000-2)
    
    #stagecounter+=1
    #if (stagecounter==1000):
    #    stage=stage+1
    #    inds=np.arange(stage,stage+1)%(len(inputs))
    #    sudoInputs=inputs[inds]
    #    sudoOutputs=outputs[inds]
    
    print(qnn.backwardPropagate(inputs,outputs,learningRate))
    
    
