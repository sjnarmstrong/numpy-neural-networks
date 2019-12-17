import numpy as np
from NeuralNetworkClass import NeuralNetwork
from RPROP import baseRPROP

#This class tests that RPROP sets DE_Dw(t) to the same values as a neural network update size with eta=1
# as the update in backprop = -eta*DE_Dw(t)

inputV=np.array([[1],[1],[1]])
outputV=np.array([[1],[1],[1]])

a=NeuralNetwork([3,5,5,3])

initialWeights=a.weights.copy()

b=baseRPROP(a,1,1,1,1,1)
b.populateDE_Dw(inputV,outputV)

a.backwardPropagate(inputV,outputV,1)

backPropStepSize=a.weights-initialWeights

diffBetweenMethods=np.sum(b.dE_dw_t+backPropStepSize)

assert diffBetweenMethods<1e-10, "Methods provided different answers. If this persists, ensure that floating point errors arnt playing a role."

print("Test Passed: Methods produce identical results (Up to floating point errors)")