import numpy as np

class GeneticAlgorithmMatrix:
    def __init__(self, weightMatrix, Evaluation, Selection, CrossoverClass, MutationClass):
        self.seedMatrix=weightMatrix
        self.PopulationSize = len(weightMatrix)
        self.Evaluation=Evaluation
        self.Selection=Selection
        self.CrossoverClass=CrossoverClass
        self.MutationClass=MutationClass
        
        self.globalBestWeights=self.seedMatrix[0].copy()
        self.bestScore=float('inf')
        
    def iterateNextGeneration(self, inputData, targetData):
        errorValues=self.Evaluation(inputData, targetData)
        selectedSeeds=self.Selection(self.seedMatrix, errorValues, self.CrossoverClass.selectionCount)
        self.CrossoverClass.Crossover(self.seedMatrix, selectedSeeds)
        self.MutationClass.Mutation(self.seedMatrix)
        
        genMinIndex=np.argmin(errorValues)
        genMinStats=errorValues[genMinIndex]
        if genMinStats<self.bestScore:
            self.bestScore=genMinStats
            self.globalBestWeights[:]=self.seedMatrix[genMinIndex]
        return genMinStats


######Selection Functions
def SelectionProbabilityWeights(seeds, error, selectionSize):
    inverseError=1.0/(error+0.000001)
    return seeds[np.random.choice(np.arange(len(seeds)),
                                  size=selectionSize,
                                  replace=False,
                                  p=inverseError/np.sum(inverseError))]
    
def TopN(seeds, error, selectionSize):
    indicies=np.argpartition(error, selectionSize)[:selectionSize]
    return seeds[indicies] 
   
def TopNSorted(seeds, error, selectionSize):
    indicies=(error[np.argpartition(error, selectionSize)[:selectionSize]]).argsort()
    return seeds[indicies]

#####CrossoverClasses
class nocrossover:
    def __init__(self, numberOfSeed):
        self.selectionCount=numberOfSeed
    def Crossover(self, WeightMatrix, SelectedSeeds):
        WeightMatrix[:] = SelectedSeeds
    

class allCombos:
    def __init__(self, numberOfSeed, chromesoneLength):
        self.selectionCount=np.sqrt(numberOfSeed)
        assert float(self.selectionCount).is_integer(), "The population size must be a squared number"
        self.selectionCount=int(self.selectionCount)
        
        self.weightMidway=int((chromesoneLength+1)/2.0)
    def Crossover(self, WeightMatrix, SelectedSeeds):
        holdStart=np.repeat(SelectedSeeds[:,:self.weightMidway],self.selectionCount,axis=0)
        holdEnd=np.tile(SelectedSeeds.T[self.weightMidway:],self.selectionCount).T
        WeightMatrix[:]=np.append(holdStart,holdEnd,axis=1)
        
class _2NearestNeighborsCrossover:
    def __init__(self, numberOfSeed, chromesoneLength):
        self.selectionCount=numberOfSeed/2.0
        assert float(self.selectionCount).is_integer(), "The population size must be an even number"
        self.selectionCount=int(self.selectionCount)
        
        self.weightMidway=int((chromesoneLength+1)/2.0)
                
    def Crossover(self, WeightMatrix, SelectedSeeds):
        WeightMatrix[:,:self.weightMidway]=SelectedSeeds[:,:self.weightMidway].repeat(2,axis=0)
        aend=SelectedSeeds[:,self.weightMidway:]
        WeightMatrix[::2,self.weightMidway:]=np.roll(aend, 1, axis=0)
        WeightMatrix[1::2,self.weightMidway:]=np.roll(aend, -1, axis=0)
    
######MutationClasses
class SimpleMutation:
    def __init__(self, numberOfSeeds, chromesoneLength, numberToMutate, mutationValueStd):
        self.numberToMutate=numberToMutate
        self.mutationValueStd=mutationValueStd
        self.GeneCount=numberOfSeeds*chromesoneLength
        
    def Mutation(self, WeightMatrix):
        MutationIndicies=np.random.randint(0,self.GeneCount,self.numberToMutate)
        WeightMatrix.flat[MutationIndicies]+=np.random.normal(0,self.mutationValueStd,self.numberToMutate)
        
class DampedMutation(SimpleMutation):
    def __init__(self, numberOfSeeds, chromesoneLength, numberToMutate, mutationValueStd, dampingConst, minStd):
        SimpleMutation.__init__(self, numberOfSeeds, chromesoneLength, numberToMutate, mutationValueStd)
        self.dampingConst=dampingConst
        self.minStd=minStd
        
    def Mutation(self, WeightMatrix):
        super().Mutation(WeightMatrix);
        self.mutationValueMean=max(self.mutationValueStd*self.dampingConst,self.minStd)
        
class GaussMutation:
    def __init__(self, numberOfSeeds, chromesoneLength, averageNumberMutated, numberMutatedDecrease, minAvgMutationCount, mutationValueStd):
        self.averageNumberMutated=averageNumberMutated
        self.numberMutatedDecrease=numberMutatedDecrease
        self.mutationValueStd=mutationValueStd
        self.minAvgMutationCount=minAvgMutationCount
        
        self.GeneCount=numberOfSeeds*chromesoneLength
        
    def Mutation(self, WeightMatrix):
        numberToMutate=int(np.random.exponential(self.averageNumberMutated))
        self.averageNumberMutated-=self.numberMutatedDecrease
        self.averageNumberMutated=max(self.averageNumberMutated,self.minAvgMutationCount)
        
        MutationIndicies=np.random.randint(0,self.GeneCount,numberToMutate)
        WeightMatrix.flat[MutationIndicies]+=np.random.normal(0,self.mutationValueStd, numberToMutate)
