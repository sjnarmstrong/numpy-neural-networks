from Classes.NeuralNetworkClass import NeuralNetwork, oneHotThreshold, thresholdValue, noThresholdValue
from Classes.GenerateData import generateData
import numpy as np
from time import time
from Classes.DataReporter import reportStats
import Classes.progressbar as pb
from math import ceil


stopAlpha=0.5

NumTrials=30
epochCountMax=200
minEpoch=3

endStep=0
datasets=['cancer','card','flare','gene','horse','heartc']
hiddenLayers=[[[4,2],[6]],  # Cancer
              [[4,4],[6]],  # Card
              [[4]],    # Flare
              [[4,2],[4,4],[9]],  # Gene
              [[4],[9]],    # Horse
              [[8,8],[6]]]  # Heartc

dataMinMaxLearingRates=[[[2,0.005],[2,0.005]],    # Cancer
                        [[2,0.005],[2,0.005]],    # Card
                        [[0.8,0.005]],    # Flare
                        [[0.8,0.005],[0.8,0.005],[0.8,0.005]], 	# Gene
                        [[0.8,0.005],[0.8,0.005]],    # Horse
                        [[0.8,0.005],[0.8,0.005]]]    # Heartc

#datasets=datasets[3:]
#hiddenLayers=hiddenLayers[3:]
#dataMinMaxLearingRates=dataMinMaxLearingRates[3:]
assert len(hiddenLayers) == len(datasets), "Please ensure that there is as many hidden layers as there are datasets"

IdealFracMinibatch=10.0
maxBatchSize=30.0

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

for dataset, datasetHiddenLayers, minMaxLearingRates in zip(datasets, hiddenLayers, dataMinMaxLearingRates):
    d=generateData(dataset,reshape=True)
    ThreshFunction=noThresholdValue if not d.booleanOuts else oneHotThreshold if d.oneHotOut else thresholdValue
    
    errorScaleTr=100.0/(d.numOutputs*d.TrainCount)
    errorScaleV=100.0/(d.numOutputs*d.ValidationCount)
    errorScaleTe=100.0/(d.numOutputs*d.TestCount)
    
    minibatchSize=min(maxBatchSize,int(np.ceil(d.TrainCount/IdealFracMinibatch)))
    invTrueFracMiniBatch=1.0/np.ceil(d.TrainCount/minibatchSize)
    stopStripLen=int(5.0/invTrueFracMiniBatch)
    trainingDataIndex=np.arange(d.TrainCount)+1
    lastTrainItemIndex=trainingDataIndex[-1]
    
    for currentHiddenLayer, minMaxLearingRate in zip(datasetHiddenLayers, np.log(minMaxLearingRates)):
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
            NN=NeuralNetwork(l)
            
            TrainingError=errorScaleTr*np.sum([NeuralNetwork.getError(ThreshFunction(NN.forwardPropagate(i)),o) for i,o in zip(TrainingInputs,TrainingOutputs)])
            TrErrors[Trial].append(TrainingError)             
            VError=errorScaleV*np.sum([NeuralNetwork.getError(ThreshFunction(NN.forwardPropagate(i)),o) for i,o in zip(ValInputs,ValOutputs)])
            VErrors[Trial].append(VError)             
            TestError=errorScaleTe*np.sum([NeuralNetwork.getError(ThreshFunction(NN.forwardPropagate(i)),o) for i,o in zip(TestInputs,TestOutputs)])
            TeErrors[Trial].append(TestError)

            notTerminated=True
            
            timebefore=time()
            epoch=-1
            while epoch < epochCount:
                epoch+=1 
                learningRate=np.exp(np.interp(epoch, [0,25], minMaxLearingRate))
                for i,o, indexTrainingItem in zip(TrainingInputs,TrainingOutputs,trainingDataIndex):
                    NN.backwardPropagate(i, o, learningRate)
                    
                    if (indexTrainingItem%minibatchSize==0 or indexTrainingItem==lastTrainItemIndex):
                        TrainingError=errorScaleTr*np.sum([NeuralNetwork.getError(ThreshFunction(NN.forwardPropagate(i)),o) for i,o in zip(TrainingInputs,TrainingOutputs)])
                        TrErrors[Trial].append(TrainingError)             
                        VError=errorScaleV*np.sum([NeuralNetwork.getError(ThreshFunction(NN.forwardPropagate(i)),o) for i,o in zip(ValInputs,ValOutputs)])
                        VErrors[Trial].append(VError)             
                        TestError=errorScaleTe*np.sum([NeuralNetwork.getError(ThreshFunction(NN.forwardPropagate(i)),o) for i,o in zip(TestInputs,TestOutputs)])
                        TeErrors[Trial].append(TestError)
                        
                        #print(VError)
                        
                        if VError<=bestGenerationValidationError:
                            bestGenerationValidationError=VError
                            bestGenerationEpoch=epoch+invTrueFracMiniBatch*(indexTrainingItem/minibatchSize)
                            bestGenerationTime=time()-timebefore
                            bestGenerationWeightedEpoch=bestGenerationEpoch*len(NN.weights)
                            bestGenerationTestError=TestError
                
                        if (notTerminated and len(TrErrors[Trial])>stopStripLen):
                            GlPrime=(VError/bestGenerationValidationError)-1
                            PkPrime=(np.sum(TrErrors[Trial][-stopStripLen:len(TrErrors[Trial])])/(stopStripLen*min(TrErrors[Trial][-stopStripLen:len(TrErrors[Trial])])))-1
                            #print(GlPrime/PkPrime)
                            if (GlPrime>stopAlphaPrime*PkPrime):
                                
                                notTerminated=False
                                GenerationStopEpoch=epoch+invTrueFracMiniBatch*(indexTrainingItem/minibatchSize)
                                if (Trial==0):
                                    #epochCount=max(ceil(epoch*5/3),minEpoch)
                                    epochCount=max(ceil(epoch*8/5),minEpoch)
            if notTerminated:
                GenerationStopEpoch=epoch#note some may not terminate as early as origional
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
                    bestGenerationTimes, bestGenerationWeightedEpochs, bestGenerationTestErrors, GenerationStopEpochs, "back-propogation", dataset, layerstring, invTrueFracMiniBatch)
