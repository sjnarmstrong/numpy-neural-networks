#example text network from https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
from ...Classes.NeuralNetworkClass import NeuralNetwork, batchNeuralNetwork
import numpy as np
  
initWeights=[0.15,0.2,0.25,0.3,0.35,0.35,0.4,0.45,0.5,0.55,0.6,0.6]
       
a=NeuralNetwork([2,2,2],initWeights)

print(a.forwardPropagate([[0.05],[0.1]]))
a.backwardPropagate(np.array([[0.05],[0.1]]),np.array([[0.01],[0.99]]),0.5)
print(a)

weights2=np.tile(initWeights,4).reshape((2,-1))
#print(weights2)
#weights2=np.append(weights2,[[1,1,1,1,1,0.35,0.4,0.45,0.5,0.55,0.6,0.6]],axis=0)
a=batchNeuralNetwork([2,2,2],4,weights2.flatten())

print(a.forwardPropagate(np.array([[[0.05,0.1],[0.05,0.1],[0.05,0.1],[0.05,0.1],[0.05,0.1],[0.05,0.1]]])))
print(a)

