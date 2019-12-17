from Classes.NeuralNetworkClass import NeuralNetwork, oneHotThresholdBatch
from Classes.MnistReader import generateData
import numpy as np
from time import time
from Classes.DataReporterMNIST import reportStats
from math import ceil
import Classes.RPROPLayer as RPL

algName="RPROP"

stopAlpha=0.6

NumTrials=1
epochCountMax=2000

endStep=0
datasets=['MNIST']
hiddenLayers=[[[130]]]

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


for dataset, datasetHiddenLayers in zip(datasets, hiddenLayers):
    d=generateData()
    ThreshFunction=oneHotThresholdBatch
    
    errorScaleTr=100.0/(10*d.TrainCount)
    errorScaleV=100.0/(10*d.ValCount)
    
    minibatchSize=min(maxBatchSize,int(np.ceil(d.TrainCount/IdealFracMinibatch)))
    invTrueFracMiniBatch=1.0/np.ceil(d.TrainCount/minibatchSize)
    stopStripLen=int(0.5/(invTrueFracMiniBatch*5.0))
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
        BestGenWeightsAllTrials=[]
        epochCount=epochCountMax
    
        for Trial in range(NumTrials):   
            
            bestGenerationValidationError=float('inf')
            bestGenerationEpoch=0
            bestGenerationTime=0
            bestGenerationWeightedEpoch=0
            GenerationStopEpoch=0
            bestGenerationTestError=float('inf')
            bestGenWeights=None
            
            NN=NeuralNetwork(l, layerTypes=[RPL.iRProp_plus_Layer]*(len(l)-1))
            
            TrainingInputs=d.TrainData/255.0
            TrainingOutputs=d.TrainLabels
            ValInputs=d.ValData/255.0
            ValOutputs=d.ValLabels
            TestInputs=d.TestImages/255.0
            TestOutputs=d.Testlabels
            
            TrainingError=errorScaleTr*np.sum((ThreshFunction(NN.forwardPropagate(TrainingInputs))-TrainingOutputs)**2)
            TrErrors[Trial].append(TrainingError)    
            VError=errorScaleV*np.sum((ThreshFunction(NN.forwardPropagate(ValInputs))-ValOutputs)**2)
            VErrors[Trial].append(VError)     
            
            timebefore=time()
            epoch=-1
            while epoch < epochCount:
                
                #TrainingInputs,TrainingOutputs=d.GetNewTrainWithNoise()
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
            
                    if (len(TrErrors[Trial])>stopStripLen*3):
                        GlPrime=(VError/bestGenerationValidationError)-1
                        PkPrime=np.sum(TrErrors[Trial][-stopStripLen:len(TrErrors[Trial])])/(stopStripLen*min(TrErrors[Trial][-stopStripLen:len(TrErrors[Trial])]))-1
                        
                        if (GlPrime>stopAlphaPrime*PkPrime):
                            
                            GenerationStopEpoch=epoch+(batchIndexEnd/d.TrainCount)
                            
                            if (Trial==0):
                                epochCount=epoch

            bestGenerationValidationErrors.append(bestGenerationValidationError)
            bestGenerationEpochs.append(bestGenerationEpoch)
            bestGenerationTimes.append(bestGenerationTime)
            bestGenerationWeightedEpochs.append(bestGenerationWeightedEpoch)
            GenerationStopEpochs.append(GenerationStopEpoch)
            BestGenWeightsAllTrials.append(bestGenWeights)
              
        layerstring=str(l[0])
        for v in l[1:]:
            layerstring+="x"+str(v)
        reportStats(BestGenWeightsAllTrials,VErrors,TrErrors, bestGenerationValidationErrors,
                    bestGenerationEpochs, bestGenerationTimes, bestGenerationWeightedEpochs,
                    GenerationStopEpochs, algName, dataset, layerstring, invTrueFracMiniBatch)
        