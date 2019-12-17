from Classes.NeuralNetworkClass import NeuralNetwork, BatchInputNetworkLayer
from Classes.MnistReader import generateData
import numpy as np
from os.path import join

def oneHotThresholdBatch(value):
    return np.argmax(value, axis=1)

weights=np.loadtxt(join("Outputs","MNIST","RPROP","784x100x10","MNIST-RPROP-784x100x10-Weights.gz"))
nn=NeuralNetwork([784,100,10],weights, layerTypes=[BatchInputNetworkLayer]*(2))

d=generateData()

indsGuessed=oneHotThresholdBatch(nn.forwardPropagate(d.TestImages))
indsTrue=oneHotThresholdBatch(d.Testlabels)
ConMat=np.zeros((10,10), dtype=np.int32)

for i,j in zip(indsGuessed,indsTrue):
    ConMat[j,i]+=1
    
for row in ConMat:
    hold="";
    for c in row:
        hold+='&'+str(c)
    hold+="\\\ \\cline{2-12} "
    print(hold)