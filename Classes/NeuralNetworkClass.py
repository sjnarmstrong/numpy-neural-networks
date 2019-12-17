import numpy as np


def oneHotThreshold(value):
    ind=np.argmax(value)
    value[:]=0.0
    value[ind]=1.0
    return value
    
def thresholdValue(value):
    np.rint(value, out=value)
    return value

def noThresholdValue(value):
    return value

def oneHotThresholdBatch(value):
    ind=np.argmax(value, axis=1)
    value[:,:]=0.0
    value[range(len(value)),ind]=1.0
    return value

def oneHotThresholdBatch2(value):
    ind=np.argmax(value, axis=2)
    value[:,:]=0.0
    value[np.repeat(range(len(value)),len(value[0])),np.tile(range(len(value[0])),len(value)), ind.flat]=1.0
    return value

class BatchInputNetworkLayer():
    def __init__(self, InputDimentions, OutputDimentions, WeightArray, BiasArray, **kwargs):
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
        
    def backwardPropagate(self, relevantDeltaData, learningRate):

        deltaM = self.calcDelta(relevantDeltaData)
        holdOut = np.dot(deltaM, self.WeightMatrixT)
        
        self.WeightMatrixT -= learningRate*np.dot(deltaM.T, self.prevX)
        self.BiasVector -= learningRate*np.sum(deltaM, axis=0)
     
            
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
  
            
    def forwardPropagate(self, inputMatrix):
        """
        This function propagates the input through this layer of the network

        :param inputVector: (n, 1) input vector where n corresponds to the number of inputs
        :return: (p, 1) output vector where p corresponds to the number of outputs
        """
        return 1.0/(1.0+np.exp(-np.einsum('ji, li->lj', self.WeightMatrixT, inputMatrix)-self.BiasVector))
    
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

class NetworkLayer:
    def __init__(self, InputDimentions, OutputDimentions, WeightArray, BiasArray, **kwargs):
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
        self.BiasVector.shape = (OutputDimentions, 1)

        # Initialise space for data related to back-propagation

        self.prevZ = np.zeros((OutputDimentions, 1))
        self.prevX = np.zeros((OutputDimentions, 1))

        # Set function used to calculate the delta for this layer

        self.calcDelta = self.calcDeltaHiddenLayer
            
    def forwardPropagate(self, inputVector):
        """
        This function propagates the input through this layer of the network

        :param inputVector: (n, 1) input vector where n corresponds to the number of inputs
        :return: (p, 1) output vector where p corresponds to the number of outputs
        """
        return 1.0/(1.0+np.exp(-np.dot(self.WeightMatrixT, inputVector)-self.BiasVector))
    
    def zStoreForwardPropagate(self, inputVector):
        """
        This function propagates the input through this layer of the network and stores the relevant information needed
        to perform back-propagation

        :param inputVector: (n, 1) input vector where n corresponds to the number of inputs
        :return: (p, 1) output vector where p corresponds to the number of outputs
        """
        self.prevX = inputVector
        self.prevZ = 1.0/(1.0+np.exp(-np.dot(self.WeightMatrixT, inputVector)-self.BiasVector))
        return self.prevZ
    
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
        delta = self.calcDelta(relevantDeltaData)
        
        deltaBias = -learningRate*delta
        
        holdOut = np.dot(self.WeightMatrixT.T, delta)
        
        self.WeightMatrixT += np.dot(deltaBias, self.prevX.T)
        self.BiasVector += deltaBias
        return holdOut
         
    def calcDeltaOutputLayer(self, Target):
        """
        Calculates the deltas for an output layer.

        :param Target: A (n, 1) matrix, where n is the number of outputs, containing the ideal results for the forward
            propagated inputs.
        :return: A (i, 1) matrix, where i is the number of inputs. This contains the deltas required for the
            backward-propagation update, associated with this layer.
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
    
class NeuralNetwork:
    def __repr__(self):
        hold = ""
        for i, layer in enumerate(self.layers):
            hold+= "_______Layer: "+str(i+1)+"\n"
            hold+= "Weights: "+"\n"
            hold+= str(layer.WeightMatrixT)+"\n"
            hold+= "BiasVector: "+"\n"
            hold+= str(layer.BiasVector)+"\n"
        return hold
    
    def __init__(self, layerNeurons, initialWeights = None, layerTypes=None, **kwargs):
        """
        Creates a fully connected neural network containing an array of neural network layers.
        :param layerNeurons: An array containing the amount of neurons in each layer of the network.
            Example: layerNeurons = [5, 6, 8, 3] will produce a neural network with the following layers:
                - An input layer with 5 neurons
                - A hidden layer with 6 neurons
                - A hidden layer with 8 neurons
                - A output layer with 3 neurons
        :param initialWeights: An optional parameter used to initialise the neural network. This can be used
            to initialise a trained network with other weights, or to start the training in an arbitrary location.
        :param layerTypes: The layers to be used in the neural network. (Default: All sigmoidal layers)
        :param **kwargs: Any additional parameters needed to initialise the NeuralNetworkLayer
        """
        
        # Ensure that there is at-least one input and one output layer in the network
        assert len(layerNeurons)>1, "At least one input layer and one output layer is needed"
        
        # Get the total number of weights needed in the network
        totalWeightCount = NeuralNetwork.getSizeOfWeightVector(layerNeurons)
        
        # Initialise the weights with the initializer or random values
        if initialWeights is None:
            self.weights = np.random.uniform(-1/np.sqrt(layerNeurons[0]), 1/np.sqrt(layerNeurons[0]), totalWeightCount)
        else:
            assert len(initialWeights) == totalWeightCount, ("Length of initial weight matrix incorrect. You need "+str(totalWeightCount)+" weights")
            self.weights = np.array(initialWeights, dtype = np.float64)  
        
        # create an empty array of layers
        self.layers = []
        layerBlockStart = 0
        
        if layerTypes is None or len(layerTypes)<(len(layerNeurons)-1):
            layerTypes=[NetworkLayer]*(len(layerNeurons)-1)
        
        for layerInputDimention, layerOutputDimention, layerType in zip(layerNeurons, layerNeurons[1:], layerTypes):
            # initialise each layer with its input and output dimentions and bi-directional pointers to the relivant weights
            layerBlockEnd = layerBlockStart+(layerInputDimention*layerOutputDimention)
            layerBiasEnd = layerBlockEnd+layerOutputDimention
            newLayer = layerType(layerInputDimention, layerOutputDimention, 
                                       self.weights[..., layerBlockStart:layerBlockEnd], 
                                       self.weights[..., layerBlockEnd:layerBiasEnd], **kwargs)
            self.layers.append(newLayer)
            
            layerBlockStart = layerBiasEnd
        
        # Tell the output later to use a different function to calculate the delta 
        newLayer.calcDelta = newLayer.calcDeltaOutputLayer

    def getSizeOfWeightVector(layerNeurons):
        """
        Gets the number of weights needed for the network.

        :param layerNeurons: A list describing the layout of the Neural network. This is the same layer info used to
            initialize the neural network.
        :return: The number of weights required in this entire network.
            Example: For an N by M by Q by R Network the following weights are needed:
                - N*M for the fist layer with N inputs and M outputs. And an additional M for the bias.
                - Using similar logic, (M+1)*Q weights are needed, for the next layer
                - And finally (Q+1)*R for the last layer
        """
        return np.sum((np.array(layerNeurons[:-1])+1)*layerNeurons[1:])
    
    def forwardPropagate(self, inputVector):
        """
        Propagates the input through each layer in the network

        :param inputVector: An (n, 1) input vector. Where n is the number of input nodes.
        :return: An (o, 1) output vector. Where o is the number of output nodes.
        """
        # Preform the forward propagation through the layers
        # setting the output of one layer to the input of the next
        for layer in self.layers:
            inputVector = layer.forwardPropagate(inputVector)
        # The output of the last layer is returned    
        return inputVector 
    
    def zStoreForwardPropagate(self, inputVector):
        """
        Propagates the input through each layer in the network, storing the relivant information.

        :param inputVector: An (n, 1) input vector. Where n is the number of input nodes.
        :return: An (o, 1) output vector. Where o is the number of output nodes.
        """
        # Preform the forward propagation through the layers
        # setting the output of one layer to the input of the next
        for layer in self.layers:
            inputVector = layer.zStoreForwardPropagate(inputVector)
        # The output of the last layer is returned    
        return inputVector
    
    def getError(outputVector, targetVector):
        """
        Get the squared error of the network for a given input

        :param outputVector: An (o, 1) output vector, where o is the number of output nodes. This contains the output of
            the network.
        :param targetVector: An (o, 1) target vector, where o is the number of output nodes. This contains the ideal
            output of the network.
        :return: A float containing the squared error
        """
        return np.sum((outputVector-targetVector)**2)
        
    
    def backwardPropagate(self, inputVector, targetVector, learningRate):
        """
        Perform backward propagation through the network.
        :param inputVector: An (n, 1) input vector. Where n is the number of input nodes.
        :param targetVector: An (o, 1) target vector, where o is the number of output nodes. This contains the ideal
            output of the network.
        :param learningRate: This is the parameter which controls how fast the algorithm learns at the expense of
            stability.
        :return:
        """
        # Preform the forward propogation through the layers 
        # setting the output of one layer to the input of the next
        # Note that this function stores data in each layer required for an upcoming back-propogation
        inputVector = self.zStoreForwardPropagate(inputVector)
        
        # Initialise the relavant data needed for calculating the delta in back propodation as the target vector
        # Then backwards Propagate the weighted deltas and update the weights on the way
        relivantDeltaData = targetVector
        for layer in reversed(self.layers):
            relivantDeltaData = layer.backwardPropagate(relivantDeltaData, learningRate)
        
        # Return the error before the weight update
        return 0.5*np.sum((inputVector-targetVector)**2)


class batchNetworkLayer:
    def __init__(self, InputDimentions, OutputDimentions, numberOfLayers, WeightArray, BiasArray):
        """
        Create a layer in a batch neural network where multiple neural networks can be forward Propagated at the same
            time with multiple inputs.

        :param InputDimentions: The number of inputs to this layer
        :param OutputDimentions: The number of outputs from this layer
        :param numberOfLayers: Number of neural networks to be processed in parallel.
        :param WeightArray: The weights associated with this array in a linear form. Note, These values are viewed
            rather than copied. This allows external changes to effect the result of this class.
        :param BiasArray: The weights associated to the internal bias terms of this layer. These are also viewed.
        """
        self.WeightMatrixT = WeightArray
        self.WeightMatrixT.shape = (numberOfLayers, OutputDimentions, InputDimentions)
        self.BiasVector = BiasArray
        self.BiasVector.shape = (numberOfLayers, 1, OutputDimentions)
            
    def forwardPropagate(self, inputMatrix):
        """
        Propagates the inputMatrix through this layer of the Neural Network system

        :param inputMatrix: A numpy matrix containing all inputs to be forward Propagated
            An (k, l, i) matrix where:
                k - is the number of neural networks in this system,
                l - is the number of input rows in this input batch,
                i - is the number of connections to layer (Excluding bias)
        :return: A numpy matrix containing the output of this neuron layer
            An (k, l, j) matrix where:
                k - is the number of neural networks in this system,
                l - is the number of input rows in this input batch,
                j - is the number of connections from this layer (Excluding bias)
        Notes
        -------
        self.WeightMatrixT : A numpy matrix containing the weights
            An (k, j, i) matrix where:
                k - is the number of neural networks in this system,
                j - is the number of connections from this layer (Excluding bias),
                i - is the number of connections to layer (Excluding bias)
        """

        return 1.0/(1.0+np.exp(-np.einsum('kji, kli->klj', self.WeightMatrixT, inputMatrix)-self.BiasVector))
            
    def forwardPropagateBest(self, inputMatrix, index):
        """
        Propagates the inputMatrix through this layer of the Neural Network system

        :param inputMatrix: A numpy matrix containing all inputs to be forward Propagated
            An (k, l, i) matrix where:
                k - is the number of neural networks in this system,
                l - is the number of input rows in this input batch,
                i - is the number of connections to layer (Excluding bias)
        :return: A numpy matrix containing the output of this neuron layer
            An (k, l, j) matrix where:
                k - is the number of neural networks in this system,
                l - is the number of input rows in this input batch,
                j - is the number of connections from this layer (Excluding bias)
        Notes
        -------
        self.WeightMatrixT : A numpy matrix containing the weights
            An (k, j, i) matrix where:
                k - is the number of neural networks in this system,
                j - is the number of connections from this layer (Excluding bias),
                i - is the number of connections to layer (Excluding bias)
        """
        return 1.0/(1.0+np.exp(-np.einsum('ji, li->lj', self.WeightMatrixT[index], inputMatrix)-self.BiasVector[index]))
    
    

class batchNeuralNetwork:    
    def __init__(self, layerNeurons, numberOfLayers, initialWeights = None, lowerBound = None, upperBound = None):
        """
        Creates a fully connected neural network containing an array of neural network layers.
        :param layerNeurons: An array containing the amount of neurons in each layer of the network.
            Example: layerNeurons = [5, 6, 8, 3] will produce a neural network with the following layers:
                - An input layer with 5 neurons
                - A hidden layer with 6 neurons
                - A hidden layer with 8 neurons
                - A output layer with 3 neurons
        
        :param numberOfLayers: Number of neural networks to be processed in parallel.
        :param initialWeights: An optional parameter used to initialise the neural network. This can be used
            to initialise a trained network with other weights, or to start the training in an arbitrary location.
        :param lowerBound: Lower bound for randomly generated weights. (Default: -sqrt(nodes in input layer))
        :param upperBound: Upper bound for randomly generated weights. (Default: sqrt(nodes in input layer))
        """
                    
        # Ensure that there is at-least one input and one output layer in the network
        assert len(layerNeurons) > 1, "At least one input layer and one output layer is needed"
        
        # Get the total number of weights needed in the network
        totalWeightCount = NeuralNetwork.getSizeOfWeightVector(layerNeurons)*numberOfLayers
        
        # Initialise the weights with the initialiser or random values
        if initialWeights is None:
            if lowerBound is None:
                lowerBound=-1/np.sqrt(layerNeurons[0])
            if upperBound is None:
                upperBound=1/np.sqrt(layerNeurons[0])
            self.weights = np.random.uniform(lowerBound, upperBound, totalWeightCount)
        else:
            assert initialWeights.size == totalWeightCount, ("Length of initial weight matrix incorrect. You need "+str(totalWeightCount)+" weights")
            self.weights = initialWeights.view()
        
        self.weights.shape = (numberOfLayers, -1)
        # create an empty array of layers
        self.layers = []
        layerBlockStart = 0
        
        for layerInputDimention, layerOutputDimention in zip(layerNeurons, layerNeurons[1:]):
            # initialise each layer with its input and output dimentions and bi-directional pointers to the relivant weights
            layerBlockEnd = layerBlockStart+(layerInputDimention*layerOutputDimention)
            layerBiasEnd = layerBlockEnd+layerOutputDimention
            newLayer = batchNetworkLayer(layerInputDimention, layerOutputDimention, numberOfLayers, 
                                       self.weights[..., :, layerBlockStart:layerBlockEnd], 
                                       self.weights[..., :, layerBlockEnd:layerBiasEnd])
            self.layers.append(newLayer)
            
            layerBlockStart = layerBiasEnd
        
    def forwardPropagate(self, inputMatrix):
        """
        Propagates the input through each layer in the network

        :param inputMatrix: A numpy matrix containing all inputs to be forward Propagated
            An (1, l, i) matrix where:
                l - is the number of input rows in this input batch,
                i - is the number of connections to layer (Excluding bias)
        :return: A numpy matrix containing the output of this neuron layer
            An (k, l, j) matrix where:
                k - is the number of neural networks in this system,
                l - is the number of input rows in this input batch,
                j - is the number of connections from this layer (Excluding bias)
        """
        # Preform the forward propogation through the layers 
        # setting the output of one layer to the input of the next
        for layer in self.layers:
            inputMatrix = layer.forwardPropagate(inputMatrix)
        # The output of the last layer is returned    
        return inputMatrix
        
    def forwardPropagateBest(self, inputMatrix, index):
        """
        Propagates the input through each layer in the network

        :param inputMatrix: A numpy matrix containing all inputs to be forward Propagated
            An (1, l, i) matrix where:
                l - is the number of input rows in this input batch,
                i - is the number of connections to layer (Excluding bias)
        :return: A numpy matrix containing the output of this neuron layer
            An (k, l, j) matrix where:
                k - is the number of neural networks in this system,
                l - is the number of input rows in this input batch,
                j - is the number of connections from this layer (Excluding bias)
        """
        # Preform the forward propogation through the layers 
        # setting the output of one layer to the input of the next
        for layer in self.layers:
            inputMatrix = layer.forwardPropagateBest(inputMatrix, index)
        # The output of the last layer is returned    
        return inputMatrix
    
    def getBatchError(self, inputMatrix, targetMatrix):
        """
        Get the error of each network in the system

        :param inputMatrix: A numpy matrix containing all inputs to be forward Propagated
            An (1, l, i) matrix where:
                l - is the number of input rows in this input batch,
                i - is the number of connections to layer (Excluding bias)
        :param targetMatrix: A numpy matrix containing the ideal output of this network
            An (1, l, j) matrix where:
                l - is the number of input rows in this input batch,
                j - is the number of connections from this layer (Excluding bias)
        :return: A numpy array containing the error for each neural network in the system.
        """
        outputmatrix = self.forwardPropagate(inputMatrix)
        return np.sum((outputmatrix-targetMatrix)**2, axis=(1, 2))
