from QuickProp import QuickPropLayer
from NeuralNetworkClass import NeuralNetwork
import numpy as np

QNN=NeuralNetwork([2, 3, 4, 5], layerTypes=[QuickPropLayer,QuickPropLayer,QuickPropLayer], mu=1.4)
NN=NeuralNetwork([2, 3, 4, 5],initialWeights=QNN.weights.copy())

inputs=np.random.uniform(0,1,(6,2))
targets=np.random.uniform(0,1,(6,5))

a=QNN.zStoreForwardPropagate(inputs)

for i,out in zip(inputs,a):
    b=NN.forwardPropagate(i.reshape(-1,1))
    assert np.allclose(out,b.flat), "Forward propogate does not match"

print("Test 1 passed")

Rdata=targets
holdout=[]
for layer in reversed(QNN.layers):
    Rdata=layer.calcDelta(Rdata)
    holdout.append(np.sum(Rdata,axis=0))
    Rdata=np.dot(Rdata, layer.WeightMatrixT)


accumulators=[np.zeros(len(l)) for l in holdout]
for inputV, Tv in zip(inputs, targets):
    a=NN.zStoreForwardPropagate(inputV.reshape((-1,1)))
    Rdata=Tv.reshape((-1,1))
    for layer, acc in zip(reversed(NN.layers), accumulators):
        Rdata=layer.calcDelta(Rdata)
        acc+=Rdata.flat
        Rdata=np.dot(layer.WeightMatrixT.T, Rdata)
    

test=np.all([np.allclose(a,b) for a,b in zip(accumulators, holdout)])

assert test, "Networks do not produce the same accumulate deltas"
print("Test 2 passed")

weightsBefore=QNN.weights.copy()
QNN.backwardPropagate(inputs[0].reshape(1,-1), targets[0].reshape(1,-1), 0.01)
dWQNN=weightsBefore-QNN.weights

weightsBefore=NN.weights.copy()
NN.backwardPropagate(inputs[0].reshape(-1,1), targets[0].reshape(-1,1), 0.01)
dWNN=weightsBefore-NN.weights

assert np.allclose(dWQNN,dWNN), "NN and QNN dont do same update step for first iteration"
print("Test 3 passed")