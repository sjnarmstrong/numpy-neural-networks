from Classes.NeuralNetworkClass import batchNeuralNetwork, oneHotThresholdBatch2, oneHotThresholdBatch, thresholdValue, noThresholdValue
from Classes.GenerateData import generateData
import numpy as np
from time import time
from Classes.DataReporter import reportStats
import Classes.progressbar as pb
from math import ceil
import Classes.GeneticAlgorithmMatrix as GA

algName="GA4"

PopulationSize=64

stopStripLen=20
stopAlpha=8

NumTrials=200
generationMaxCount=30
generationMinCount=10

endStep=0
datasets=['cancer','card','flare','gene','horse','heartc']
hiddenLayers=[[[4,2],[6]],  # Cancer
              [[4,4],[6]],  # Card
              [[4]],    # Flare
              [[4,2],[4,4],[9]],  # Gene
              [[4],[9]],    # Horse
              [[8,8],[6]]]  # Heartc

#datasets=datasets[3:4]
#hiddenLayers=hiddenLayers[3:4]
assert len(hiddenLayers) == len(datasets), "Please ensure that there is as many hidden layers as there are datasets"


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

for dataset, datasetHiddenLayers in zip(datasets, hiddenLayers):
    d=generateData(dataset)
    ThreshFunction=noThresholdValue if not d.booleanOuts else oneHotThresholdBatch2 if d.oneHotOut else thresholdValue
    ThreshFunction2=noThresholdValue if not d.booleanOuts else oneHotThresholdBatch if d.oneHotOut else thresholdValue
    
    errorScaleTr=100.0/(d.numOutputs*d.TrainCount)
    errorScaleV=100.0/(d.numOutputs*d.ValidationCount)
    errorScaleTe=100.0/(d.numOutputs*d.TestCount)
    
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
        generationCount=generationMaxCount
    
        for Trial in range(NumTrials):   
            
            bestGenerationValidationError=float('inf')
            bestGenerationEpoch=0
            bestGenerationTime=0
            bestGenerationWeightedEpoch=0
            GenerationStopEpoch=0
            bestGenerationTestError=float('inf')
            
            TrainingInputs,TrainingOutputs,ValInputs,ValOutputs,TestInputs,TestOutputs=d.getDatasets()
            TrainingInputs.shape=(1,)+TrainingInputs.shape
            TrainingOutputs.shape=(1,)+TrainingOutputs.shape
            ValInputs.shape=(1,)+ValInputs.shape
            ValOutputs.shape=(1,)+ValOutputs.shape
            TestInputs.shape=(1,)+TestInputs.shape
            TestOutputs.shape=(1,)+TestOutputs.shape
            
            BNN=batchNeuralNetwork(l, PopulationSize)
            seeds=BNN.weights.view()
            chromesoneLen=len(seeds[0])
            CrossOverClass=GA._2NearestNeighborsCrossover(PopulationSize, chromesoneLen)
            MutationClass=GA.DampedMutation(PopulationSize, chromesoneLen, ceil(PopulationSize*chromesoneLen*25/100), 1.5, 0.999,0.05)
            ga=GA.GeneticAlgorithmMatrix(seeds, BNN.getBatchError, GA.SelectionProbabilityWeights, CrossOverClass, MutationClass)
            TestBNN=batchNeuralNetwork(l,1,ga.globalBestWeights)
            
            weightPenalty=np.sum(BNN.weights.shape)
            
            #trainingErrors=np.sum((ThreshFunction(TestBNN.forwardPropagate(TrainingInputs))-TrainingOutputs)**2, axis=(1,2))
            #bestTrainErrorindex=np.argmin(trainingErrors)
            TrainingError=errorScaleTr*np.sum((ThreshFunction2(TestBNN.forwardPropagateBest(TrainingInputs[0],0))-TrainingOutputs[0])**2, axis=(0,1))
            #trainingErrors[bestTrainErrorindex]
            TrErrors[Trial].append(TrainingError)    
            VError=errorScaleV*np.sum((ThreshFunction2(TestBNN.forwardPropagateBest(ValInputs[0],0))-ValOutputs[0])**2, axis=(0,1))
            VErrors[Trial].append(VError)      
            TestError=errorScaleV*np.sum((ThreshFunction2(TestBNN.forwardPropagateBest(TestInputs[0],0))-TestOutputs[0])**2, axis=(0,1))    
            TeErrors[Trial].append(TestError)
            
            notTerminated=True
            
            timebefore=time()
            epoch=-1
            while epoch < generationCount:
                epoch+=1 ##Note preincrement as epoch is after train
                
                
                ga.iterateNextGeneration(TrainingInputs,TrainingOutputs)
                
                TrainingError=errorScaleTr*np.sum((ThreshFunction2(TestBNN.forwardPropagateBest(TrainingInputs[0], 0))-TrainingOutputs[0])**2, axis=(0,1))
                TrErrors[Trial].append(TrainingError)    
                VError=errorScaleV*np.sum((ThreshFunction2(TestBNN.forwardPropagateBest(ValInputs[0], 0))-ValOutputs[0])**2, axis=(0,1))
                VErrors[Trial].append(VError)      
                TestError=errorScaleV*np.sum((ThreshFunction2(TestBNN.forwardPropagateBest(TestInputs[0], 0))-TestOutputs[0])**2, axis=(0,1))     
                TeErrors[Trial].append(TestError)
            
                
                
                if VError<=bestGenerationValidationError:
                    bestGenerationValidationError=VError
                    bestGenerationEpoch=epoch
                    bestGenerationTime=time()-timebefore
                    bestGenerationWeightedEpoch=bestGenerationEpoch*weightPenalty
                    bestGenerationTestError=TestError
        
                if (notTerminated and epoch>stopStripLen):
                    GlPrime=(VError/bestGenerationValidationError)-1
                    PkPrime=np.sum(TrErrors[Trial][-stopStripLen:len(TrErrors[Trial])])/(stopStripLen*min(TrErrors[Trial][-stopStripLen:len(TrErrors[Trial])]))-1
                    
                    
                    if (GlPrime>stopAlphaPrime*PkPrime):
                        
                        notTerminated=False
                        GenerationStopEpoch=epoch
                        if (Trial==0):
                            generationCount=max(ceil(epoch*2),generationMinCount)
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
                    bestGenerationTimes, bestGenerationWeightedEpochs, bestGenerationTestErrors, GenerationStopEpochs, algName, dataset, layerstring, 1, reportGenerateion=True)

