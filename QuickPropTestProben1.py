from Classes.NeuralNetworkClass import NeuralNetwork, oneHotThresholdBatch, thresholdValue, noThresholdValue
from Classes.GenerateData import generateData
import numpy as np
from time import time
from Classes.DataReporter import reportStats
import Classes.progressbar as pb
from math import ceil
import Classes.QuickProp as QP

algName="QuickProp"

stopAlpha=0.5

NumTrials=30
epochCountMax=200
minEpoch=10

endStep=0
datasets=['cancer','card','flare','gene','horse','heartc']
hiddenLayers=[[[4,2],[6]],  # Cancer
              [[4,4],[6]],  # Card
              [[4]],    # Flare
              [[4,2],[4,4],[9]],  # Gene
              [[4],[9]],    # Horse
              [[8,8],[6]]]  # Heartc
learningRates=[0.1,0.05,0.05,0.1,0.1,0.001]
#datasets=datasets
#hiddenLayers=hiddenLayers
#learningRates=learningRates
learningRate=0.8
assert len(hiddenLayers) == len(datasets), "Please ensure that there is as many hidden layers as there are datasets"

IdealFracMinibatch=10.0
maxBatchSize=30

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

for dataset, datasetHiddenLayers, learningRateR in zip(datasets, hiddenLayers, learningRates):
    d=generateData(dataset)
    ThreshFunction=noThresholdValue if not d.booleanOuts else oneHotThresholdBatch if d.oneHotOut else thresholdValue
    
    errorScaleTr=100.0/(d.numOutputs*d.TrainCount)
    errorScaleV=100.0/(d.numOutputs*d.ValidationCount)
    errorScaleTe=100.0/(d.numOutputs*d.TestCount)
    
    minibatchSize=min(maxBatchSize,int(np.ceil(d.TrainCount/IdealFracMinibatch)))
    invTrueFracMiniBatch=1.0/np.ceil(d.TrainCount/minibatchSize)
    stopStripLen=int(5.0/invTrueFracMiniBatch)
    trainingDataIndex=np.arange(d.TrainCount)+1
    lastTrainItemIndex=trainingDataIndex[-1]
    
    for currentHiddenLayer in datasetHiddenLayers:
        l=[d.numInputs]+currentHiddenLayer+[d.numOutputs]
        
        TrErrors=[[] for Trial in range(NumTrials)]
        VErrors=[[] for Trial in range(NumTrials)]
        TeErrors=[[] for Trial in range(NumTrials)]
        
        bestGenerationValidationErrors=[]
        bestGenerationEpochs=[]
        bestGenerationTimes=[]
        bestGenerationWeightedEpochs=[]
        bestGenerationTestErrors=[]
        GenerationStopEpochs=[]
        epochCount=epochCountMax
    
        for Trial in range(NumTrials):   
            
            bestGenerationValidationError=float('inf')
            bestGenerationEpoch=0
            bestGenerationTime=0
            bestGenerationWeightedEpoch=0
            GenerationStopEpoch=0
            bestGenerationTestError=float('inf')
            
            TrainingInputs,TrainingOutputs,ValInputs,ValOutputs,TestInputs,TestOutputs=d.getDatasets()
            NN=NeuralNetwork(l, layerTypes=[QP.QuickPropLayer]*(len(l)-1), mu=1.00)
            
            TrainingError=errorScaleTr*np.sum((ThreshFunction(NN.forwardPropagate(TrainingInputs))-TrainingOutputs)**2)
            TrErrors[Trial].append(TrainingError)    
            VError=errorScaleV*np.sum((ThreshFunction(NN.forwardPropagate(ValInputs))-ValOutputs)**2)
            VErrors[Trial].append(VError)      
            TestError=errorScaleV*np.sum((ThreshFunction(NN.forwardPropagate(TestInputs))-TestOutputs)**2)       
            TeErrors[Trial].append(TestError)

            notTerminated=True
            
            timebefore=time()
            epoch=-1
            while epoch < epochCount:
                epoch+=1 ##Note preincrement as epoch is after train
                
                batchIndexStart=0
                for batchIndexEnd in range(minibatchSize, d.TrainCount+minibatchSize, minibatchSize):
                    NN.backwardPropagate(TrainingInputs[batchIndexStart:batchIndexEnd], TrainingOutputs[batchIndexStart:batchIndexEnd], learningRate)
                    batchIndexStart=batchIndexEnd
                    TrainingError=errorScaleTr*np.sum((ThreshFunction(NN.forwardPropagate(TrainingInputs))-TrainingOutputs)**2)
                    TrErrors[Trial].append(TrainingError)    
                    VError=errorScaleV*np.sum((ThreshFunction(NN.forwardPropagate(ValInputs))-ValOutputs)**2)
                    VErrors[Trial].append(VError)      
                    TestError=errorScaleV*np.sum((ThreshFunction(NN.forwardPropagate(TestInputs))-TestOutputs)**2)       
                    TeErrors[Trial].append(TestError)
                    
                    #print(TrainingError)
                    
                    if VError<=bestGenerationValidationError:
                        bestGenerationValidationError=VError
                        bestGenerationEpoch=epoch+(batchIndexEnd/d.TrainCount)
                        bestGenerationTime=time()-timebefore
                        bestGenerationWeightedEpoch=bestGenerationEpoch*len(NN.weights)
                        bestGenerationTestError=TestError
            
                    if (notTerminated and epoch>5):
                        GlPrime=(VError/bestGenerationValidationError)-1
                        PkPrime=np.sum(TrErrors[Trial][-stopStripLen:len(TrErrors[Trial])])/(stopStripLen*min(TrErrors[Trial][-stopStripLen:len(TrErrors[Trial])]))-1
                        
                        if (GlPrime>stopAlphaPrime*PkPrime):
                            
                            notTerminated=False
                            GenerationStopEpoch=epoch+(batchIndexEnd/d.TrainCount)
                            if (Trial==0):
                                epochCount=max(ceil(epoch*6/3),minEpoch)
            if notTerminated:
                GenerationStopEpoch=epoch
            bestGenerationValidationErrors.append(bestGenerationValidationError)
            bestGenerationEpochs.append(bestGenerationEpoch)
            bestGenerationTimes.append(bestGenerationTime)
            bestGenerationWeightedEpochs.append(bestGenerationWeightedEpoch)
            bestGenerationTestErrors.append(bestGenerationTestError)
            GenerationStopEpochs.append(GenerationStopEpoch)
            
            loopCounter+=1
            bar.update(loopCounter)   
        layerstring=str(l[0])
        for v in l[1:]:
            layerstring+="x"+str(v)
        reportStats(TeErrors, VErrors, TrErrors, bestGenerationValidationErrors, bestGenerationEpochs,
                    bestGenerationTimes, bestGenerationWeightedEpochs, bestGenerationTestErrors, GenerationStopEpochs, algName, dataset, layerstring, invTrueFracMiniBatch)
