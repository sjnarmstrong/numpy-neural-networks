import numpy as np
from .NeuralNetworkClass import BatchInputNetworkLayer
class NetworkLayerWithMomentum(BatchInputNetworkLayer):
    def __init__(self, InputDimentions, OutputDimentions, WeightArray, BiasArray, Momentum):
        """
        Initialise a neural network layer. This represents a row in the fully connected neural network.

        :param InputDimentions: The number of inputs to this layer
        :param OutputDimentions: The number of outputs from this layer
        :param WeightArray: The weights associated with this array in a linear form. Note, These values are viewed
            rather than copied. This allows external changes to effect the result of this class.
        :param BiasArray: The weights associated to the internal bias terms of this layer. These are also viewed.
        """
        BatchInputNetworkLayer.__init__(self, InputDimentions, OutputDimentions, WeightArray, BiasArray)
        
        self.prevDW = np.zeros(self.WeightMatrixT.shape)
        self.prevDbias = np.zeros(self.BiasVector.shape)
        self.Momentum = Momentum
    
    def backwardPropagate(self, relevantDeltaData, learningRate):

        deltaM = self.calcDelta(relevantDeltaData)
        holdOut = np.dot(deltaM, self.WeightMatrixT)
        
        deltaBias = -learningRate*deltaM
        
        self.prevDW = np.dot(deltaBias.T, self.prevX) + self.Momentum*self.prevDW 
        self.prevDbias = np.sum(deltaBias, axis=0) + self.Momentum*self.prevDbias
        
        self.WeightMatrixT += self.prevDW
        self.BiasVector += self.prevDbias
     
            
        return holdOut
    
#    def backwardPropagate(self, relevantDeltaData, learningRate):
#        """
#        Updated the weights of this node through the rules of backward propagation.
#
#        :param relevantDeltaData: This is the data required by this layer to calculate the deltas. For the output layer, 
#            this refers to the target data, and in the hidden layers this refers to the weighted deltas of the previous
#            layer.
#        :param learningRate: This is the parameter which controls how fast the algorithm learns at the expense of
#            stability.
#        :return: returns the weighted deltas required for updating the layer which comes before this one.
#        """
#        deltaM = self.calcDelta(relevantDeltaData)
#        
#        deltaBias = -learningRate*deltaM
#        
#        holdOut = np.dot(deltaM, self.WeightMatrixT)
#        
#        
#        self.prevDW = np.dot(deltaBias.T, self.prevX) + (self.Momentum*self.prevDW)
#        self.prevDbias = np.sum(deltaM, axis=0) + (self.Momentum*self.prevDbias)
#        
#        self.WeightMatrixT += self.prevDW
#        self.BiasVector += self.prevDbias
#        return holdOut
    
class NetworkLayerWithAdaptiveWeights(BatchInputNetworkLayer):
    def __init__(self, InputDimentions, OutputDimentions, WeightArray, BiasArray, **kwargs):
        """
        Initialise a neural network layer. This represents a row in the fully connected neural network.

        :param InputDimentions: The number of inputs to this layer
        :param OutputDimentions: The number of outputs from this layer
        :param WeightArray: The weights associated with this array in a linear form. Note, These values are viewed
            rather than copied. This allows external changes to effect the result of this class.
        :param BiasArray: The weights associated to the internal bias terms of this layer. These are also viewed.
        """
        BatchInputNetworkLayer.__init__(self, InputDimentions, OutputDimentions, WeightArray, BiasArray)
        
        self.HdEdW = np.zeros(self.WeightMatrixT.shape)
        self.HdEdBias = np.zeros(self.BiasVector.shape)
      
    def resetHistory(self):
        self.HdEdW = np.zeros(self.WeightMatrixT.shape)
        self.HdEdBias = np.zeros(self.BiasVector.shape)
    
    def backwardPropagate(self, relevantDeltaData, learningRate):
        """
        Updated the weights of this node through the rules of backward propagation.

        :param relevantDeltaData: This is the data required by this layer to calculate the deltas. For the output layer, 
            this refers to the target data, and in the hidden layers this refers to the weighted deltas of the previous
            layer.
        :param learningRate: This is the parameter which controls how fast the algorithm learns at the expense of
            stability.
        :return: returns the weighted deltas required for updating the layer which comes before this one.
        """
        deltaM = self.calcDelta(relevantDeltaData)
        
        deltaBiasM = -learningRate*deltaM
        
        holdOut = np.dot(deltaM, self.WeightMatrixT)
        holdDw = np.dot(deltaBiasM.T, self.prevX)
        
        self.HdEdW += holdDw**2
        self.HdEdBias +=np.sum(deltaBiasM, axis=0)**2
        
        self.WeightMatrixT += (holdDw / (1e-10 + np.sqrt(self.HdEdW)))
        self.BiasVector += (np.sum(deltaBiasM, axis=0) / (1e-10 + np.sqrt(self.HdEdBias)))
        return holdOut