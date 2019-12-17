import numpy as np
from .NeuralNetworkClass import NetworkLayer, NeuralNetwork

class QuickPropLayer():
    def __init__(self, InputDimentions, OutputDimentions, WeightArray, BiasArray, mu):
        """
        Initialise a neural network layer. This represents a row in the fully connected neural network.

        :param InputDimentions: The number of inputs to this layer
        :param OutputDimentions: The number of outputs from this layer
        :param WeightArray: The weights associated with this array in a linear form. Note, These values are viewed
            rather than copied. This allows external changes to effect the result of this class.
        :param BiasArray: The weights associated to the internal bias terms of this layer. These are also viewed.
        """
        self.WeightMatrixT = WeightArray
        # We reshape the array to allow easy matrix multiplication. Using this method rather than ".reshape()", ensures
        # that the view is kept in tact
        self.WeightMatrixT.shape = (OutputDimentions, InputDimentions)
        self.BiasVector = BiasArray

        # Initialise space for data related to back-propagation

        self.prevZ = None
        self.prevX = None

        # Set function used to calculate the delta for this layer

        self.calcDelta = self.calcDeltaHiddenLayer
        
    
        self.pDw=np.zeros(self.WeightMatrixT.shape).flatten()
        self.pDbias=np.zeros(self.BiasVector.shape).flatten()
        self.pdEdW=np.zeros(self.WeightMatrixT.shape).flatten()
        self.pdEdBias=np.zeros(self.BiasVector.shape).flatten()
        self.mu=mu
        
    def backwardPropagate(self, relevantDeltaData, learningRate):

        deltaM = self.calcDelta(relevantDeltaData)
        holdOut = np.dot(deltaM, self.WeightMatrixT)
        
        dEdW = np.dot(deltaM.T, self.prevX).flatten()
        dEdBias = np.sum(deltaM, axis=0).flatten()
        
        scale = np.nan_to_num(dEdW/(self.pdEdW-dEdW))
        np.clip(scale, -self.mu, self.mu, scale)
        self.pDw = scale*self.pDw
        inds = np.where(np.logical_or(np.sign(dEdW)==np.sign(self.pdEdW), self.pdEdW<1e-15))
        self.pDw[inds] -= learningRate*dEdW[inds]
        
        scale = np.nan_to_num(dEdBias/(self.pdEdBias-dEdBias))
        np.clip(scale, -self.mu, self.mu, scale)
        self.pDbias = scale*self.pDbias
        inds = np.where(np.logical_or(np.sign(dEdBias)==np.sign(self.pdEdBias), self.pdEdBias<1e-15))
        self.pDbias[inds] -= learningRate*dEdBias[inds]
        
        #print(self.pDw)
        
#        self.pDw = -learningRate*dEdW
#        self.pDbias = -learningRate*dEdBias
        
        self.pdEdW=dEdW.copy()
        self.pdEdBias=dEdBias.copy()
        self.WeightMatrixT.flat += self.pDw
        self.BiasVector += self.pDbias
     
            
        return holdOut
    

    def calcDeltaOutputLayer(self, Target):
        """
        Calculates the deltas for an output layer.

        :param Target: A (N, M) matrix, where:
            -N is the number of items in the batch,
            -M is the number of outputs of the layer.
        This contains the ideal results for the forward propagated inputs.
        
        :return: A (N, M) matrix. This contains
        the deltas required for the backward-propagation update, associated with this layer.
        """
        return self.prevZ*(1.0-self.prevZ)*(self.prevZ-Target)
    
    def calcDeltaHiddenLayer(self, WeightedDelta):
        """
        Calculates the deltas for this layer as a hidden layer.

        :param WeightedDelta: A (n, 1) matrix, where n is the number of outputs of this layer, weighted deltas
            back-propagated.
        :return: A (i, 1) matrix, where i is the number of inputs. This contains the deltas required for the
            backward-propagation update, associated with this layer.
        """
        return self.prevZ*(1.0-self.prevZ)*(WeightedDelta)
  
    def zStoreForwardPropagate(self, inputMatrix):
        """
        Propagates the inputMatrix through this layer of the Neural Network system,
        storing all the data needed for forward propogation.
    
        :param inputMatrix: A numpy matrix containing all inputs to be forward Propagated
            An (l, i) matrix where:
                l - is the number of input rows in this input batch,
                i - is the number of connections to layer (Excluding bias)
        :return: A numpy matrix containing the output of this neuron layer
            An (l, j) matrix where:
                l - is the number of input rows in this input batch,
                j - is the number of connections from this layer (Excluding bias)
        Notes
        -------
        self.WeightMatrixT : A numpy matrix containing the weights
            An (j, i) matrix where:
                j - is the number of connections from this layer (Excluding bias),
                i - is the number of connections to layer (Excluding bias)
        """
        self.prevX=inputMatrix

        self.prevZ=1.0/(1.0+np.exp(-np.einsum('ji, li->lj', self.WeightMatrixT, inputMatrix)-self.BiasVector))
        return self.prevZ 
    
    def forwardPropagate(self, inputMatrix):
        return 1.0/(1.0+np.exp(-np.einsum('ji, li->lj', self.WeightMatrixT, inputMatrix)-self.BiasVector))