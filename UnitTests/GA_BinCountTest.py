import numpy as np
from NeuralNetworkClass import batchNeuralNetwork, NeuralNetwork
import GeneticAlgorithmMatrix as GA

##Binarry counter test
inputs=[[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
outputs=inputs[1:]+[inputs[0]]

inputs=np.array([inputs])
outputs=np.array([outputs])

layerInfo=[3,40,3]
popSize=4096
a=batchNeuralNetwork(layerInfo, popSize)

seeds=a.weights.view()
chromesoneLen=len(seeds[0])

CrossOverClass=GA._2NearestNeighborsCrossover(popSize, chromesoneLen)
MutationClass=GA.SimpleMutation(popSize, chromesoneLen, 5, 0.2)
b=GA.GeneticAlgorithmMatrix(seeds, a.getBatchError, GA.TopN, CrossOverClass, MutationClass)

#for loop in range(10000):
#    print(b.iterateNextGeneration(inputs,outputs))
#    for i,o in zip(inputs,outputs):
#        b.updateDE_Dw(i,o)
#        print(NeuralNetwork.getError(b.neuralNetwork.forwardPropagate(i),o))
#    b.updateWeights()
    
trainedNetwork=NeuralNetwork(layerInfo,b.globalBestWeights.copy())


a=batchNeuralNetwork(layerInfo, popSize)

seeds=a.weights.view()
chromesoneLen=len(seeds[0])

CrossOverClass=GA.allCombos(popSize, chromesoneLen)
MutationClass=GA.DampedMutation(popSize, chromesoneLen, 5, 1, 0.99,0.2)
b=GA.GeneticAlgorithmMatrix(seeds, a.getBatchError, GA.TopN, CrossOverClass, MutationClass)

for loop in range(10000):
    print(b.iterateNextGeneration(inputs,outputs))
#    for i,o in zip(inputs,outputs):
#        b.updateDE_Dw(i,o)
#        print(NeuralNetwork.getError(b.neuralNetwork.forwardPropagate(i),o))
#    b.updateWeights()
    
trainedNetwork2=NeuralNetwork(layerInfo,b.globalBestWeights)