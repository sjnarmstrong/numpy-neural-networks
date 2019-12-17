from QuickProp import QuickPropLayer
from NeuralNetworkClass import NeuralNetwork

NN=NeuralNetwork([2,3,3], layerTypes=[QuickPropLayer,QuickPropLayer], mu=1.4)

a=NN.forwardPropagate([[1,1],[0,0],[1,0],[1,0],[1,0],[1,0]])

b=NN.layers[-1]