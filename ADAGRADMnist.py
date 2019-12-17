from Classes.NeuralNetworkClass import NeuralNetwork, oneHotThresholdBatch
from Classes.MnistReader import generateData
import numpy as np
from time import time
from Classes.DataReporter import reportStats
import Classes.progressbar as pb
from math import ceil
import Classes.MomentumLayer as ML

algName="ADAGRAD"

stopStripLen=5
stopAlpha=0.5

NumTrials=30
epochCountMax=200
minEpoch=6

endStep=0
datasets=['MNIST']
hiddenLayers=[[[30]]]

#datasets=datasets[3:4]
#hiddenLayers=hiddenLayers[3:4]
#learningRates=learningRates[3:4]
assert len(hiddenLayers) == len(datasets), "Please ensure that there is as many hidden layers as there are datasets"

IdealFracMinibatch=20.0
maxBatchSize=200

'''
Setup Stop Info
'''
stopAlphaPrime=10*stopAlpha

'''
Setup progress Information
'''
TotalItCounter=NumTrials*np.sum(len(l) for l in hiddenLayers)
loopCounter=0
bar = pb.ProgressBar(max_value=TotalItCounter, widgets=[
    pb.Bar(),
    ' (', pb.ETA(), ') ',
])

bar.update(loopCounter)
#learningRate=0.08

for dataset, datasetHiddenLayers in zip(datasets, hiddenLayers):
    d=generateData()
    ThreshFunction=oneHotThresholdBatch
    
    errorScaleTr=100.0/(10*d.TrainCount)
    errorScaleV=100.0/(10*d.ValCount)
    
    minibatchSize=min(maxBatchSize,int(np.ceil(d.TrainCount/IdealFracMinibatch)))
    invTrueFracMiniBatch=1.0/np.ceil(d.TrainCount/minibatchSize)
    stopStripLen=int(1.0/(invTrueFracMiniBatch*5.0))
    trainingDataIndex=np.arange(d.TrainCount)+1
    lastTrainItemIndex=trainingDataIndex[-1]
    
    for currentHiddenLayer in datasetHiddenLayers:
        l=[d.numInputs]+currentHiddenLayer+[10]
        
        TrErrors=[[] for Trial in range(NumTrials)]
        VErrors=[[] for Trial in range(NumTrials)]
        
        bestGenerationValidationErrors=[]
        bestGenerationEpochs=[]
        bestGenerationTimes=[]
        bestGenerationWeightedEpochs=[]
        GenerationStopEpochs=[]
        epochCount=epochCountMax
    
        for Trial in range(NumTrials):   
            
            bestGenerationValidationError=float('inf')
            bestGenerationEpoch=0
            bestGenerationTime=0
            bestGenerationWeightedEpoch=0
            GenerationStopEpoch=0
            bestGenerationTestError=float('inf')
            bestGenWeights=None
            
            NN=NeuralNetwork(l, layerTypes=[ML.NetworkLayerWithAdaptiveWeights]*(len(l)-1))
            
            TrainingInputs=d.TrainData
            TrainingOutputs=d.TrainLabels
            ValInputs=d.ValData
            ValOutputs=d.ValLabels
            TestInputs=d.TestImages
            TestOutputs=d.Testlabels
            
            TrainingError=errorScaleTr*np.sum((ThreshFunction(NN.forwardPropagate(TrainingInputs))-TrainingOutputs)**2)
            TrErrors[Trial].append(TrainingError)    
            VError=errorScaleV*np.sum((ThreshFunction(NN.forwardPropagate(ValInputs))-ValOutputs)**2)
            VErrors[Trial].append(VError)     

            notTerminated=True
            
            timebefore=time()
            epoch=0
            while epoch < epochCount:
                
                TrainingInputs,TrainingOutputs=d.GetNewTrainWithNoise()
                epoch+=1 ##Note preincrement as epoch is after train
                
                #learningRate=np.exp(np.interp(epoch, [0,25], np.log([3,0.001])))
                #print(learningRate)
                learningRate=10
                
                batchIndexStart=0
                for batchIndexEnd in range(minibatchSize, d.TrainCount+minibatchSize, minibatchSize):
                    #print(TrainingInputs[batchIndexStart:batchIndexEnd])
                    NN.backwardPropagate(TrainingInputs[batchIndexStart:batchIndexEnd], TrainingOutputs[batchIndexStart:batchIndexEnd], learningRate)
                    batchIndexStart=batchIndexEnd
                    TrainingError=errorScaleTr*np.sum((ThreshFunction(NN.forwardPropagate(TrainingInputs))-TrainingOutputs)**2)
                    TrErrors[Trial].append(TrainingError)    
                    VError=errorScaleV*np.sum((ThreshFunction(NN.forwardPropagate(ValInputs))-ValOutputs)**2)
                    VErrors[Trial].append(VError)      
                    
                    print(VError)
                    if VError<=bestGenerationValidationError:
                        bestGenerationValidationError=VError
                        bestGenerationEpoch=epoch+(batchIndexEnd/d.TrainCount)
                        bestGenerationTime=time()-timebefore
                        bestGenerationWeightedEpoch=bestGenerationEpoch*len(NN.weights)
                        bestGenWeights=NN.weights.copy()
            
                    if (notTerminated and epoch>5):
                        GlPrime=(VError/bestGenerationValidationError)-1
                        PkPrime=np.sum(TrErrors[Trial][-stopStripLen:len(TrErrors[Trial])])/(stopStripLen*min(TrErrors[Trial][-stopStripLen:len(TrErrors[Trial])]))-1
                        
                        if (GlPrime>stopAlphaPrime*PkPrime):
                            
                            notTerminated=False
                            GenerationStopEpoch=epoch+(batchIndexEnd/d.TrainCount)
                            if (Trial==0):
                                epochCount=max(ceil(epoch*5/3),minEpoch)

            bestGenerationValidationErrors.append(bestGenerationValidationError)
            bestGenerationEpochs.append(bestGenerationEpoch)
            bestGenerationTimes.append(bestGenerationTime)
            bestGenerationWeightedEpochs.append(bestGenerationWeightedEpoch)
            GenerationStopEpochs.append(GenerationStopEpoch)
            
            loopCounter+=1
            bar.update(loopCounter)   
        layerstring=str(l[0])
        for v in l[1:]:
            layerstring+="x"+str(v)
        reportStats(None, VErrors, TrErrors, bestGenerationValidationErrors, bestGenerationEpochs,
                    bestGenerationTimes, bestGenerationWeightedEpochs, None, GenerationStopEpochs, algName, dataset, layerstring, invTrueFracMiniBatch)
