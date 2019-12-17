import numpy as np
from NeuralNetworkClass import batchNeuralNetwork, NeuralNetwork
import GeneticAlgorithmMatrix as GA
from PSO import PSO
##Binarry counter test
inputs=[[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
outputs=inputs[1:]+[inputs[0]]

inputs=np.array([inputs])
outputs=np.array([outputs])

layerInfo=[3,40,3]
popSize=4096

bnn=batchNeuralNetwork(layerInfo,popSize,lowerBound=-2,upperBound=2)
pso=PSO(1.49618,1.49618,0.7298,bnn.weights,bnn.getBatchError,-2,2)

for i in range(60):
    print(pso.iterate(inputs,outputs,bnn.weights))
    
trainedNetwork2=NeuralNetwork(layerInfo,pso.getBestWeightsAndError()[1])