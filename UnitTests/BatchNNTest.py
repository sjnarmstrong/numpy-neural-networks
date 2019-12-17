from NeuralNetworkClass import NeuralNetwork, batchNeuralNetwork
import numpy as np

LayerInfo=(100,1000,1000,1000,1000,10000,5)
NumberOfNetworks=3
NumberOfInputs=10

a=batchNeuralNetwork(LayerInfo,NumberOfNetworks)

randInput=np.random.uniform(0,1,NumberOfInputs*LayerInfo[0])
randInput=randInput.reshape(1,NumberOfInputs,LayerInfo[0])

batchOut=a.forwardPropagate(randInput)

for inputIndex in range(NumberOfInputs):
    for weightIndex in range(NumberOfNetworks):
        testNetwork=NeuralNetwork(LayerInfo,a.weights[weightIndex].flatten())
        testOut=testNetwork.forwardPropagate( randInput[0,inputIndex].reshape(1,-1) )
        assert np.allclose(batchOut[weightIndex,inputIndex],testOut.flatten()), "Discrepency found, Error has occured"
        
print("All tests have sucessfully passed")
        