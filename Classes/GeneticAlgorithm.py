from NeuralNetworkClass import NeuralNetwork 
import numpy as np
import itertools as iterT

class GeneticAlgorithm:
    def __init__(self, seedGenerator, PopulationSize):
        self.seeds=[seedGenerator() for i in range(PopulationSize)]
        self.PopulationSize = PopulationSize
        
    def iterateNextGeneration(self, inputData, targetData):
        errorValues=self.evaluateSeeds(inputData, targetData)
        selectedSeeds=self.Selection(self.seeds, errorValues, selectionSize)
        self.Crossover(selectedSeeds, self.seeds)
        self.Mutation()
        return iterationStats


######Evaluation Functions
def L2NormEvaluation(x, inputData, targetData):
    return NeuralNetwork.getError(x.forwardPropagate(inputData), targetData)

def invL2NormEvaluation(x, inputData, targetData):
    return 1.0/(NeuralNetwork.getError(x.forwardPropagate(inputData), targetData) + 1.0)

invL2NormEvaluationVector = np.vectorize(invL2NormEvaluation, excluded=[1,2])
L2NormEvaluationVector = np.vectorize(L2NormEvaluation, excluded=[1,2])


######Selection Functions
def SelectionProbabilityWeights(seeds, inverseError, selectionSize):
    return np.random.choice(seeds,size=selectionSize,replace=False,p=inverseError/np.sum(inverseError))


#####CrossoverClasses
class nocrossover:
    def __init__(self, parent: GeneticAlgorithm):
        self.selectionSize=parent.PopulationSize
        self.parent = parent
    def crossover(self, SelectedSeeds):
        self.parent.seeds = SelectedSeeds
    

class allCombos:
    def __init__(self, parent: GeneticAlgorithm):
        self.selectionSize=np.sqrt(parent.PopulationSize)
        assert float(self.selectionSize).is_integer(), "The population size must be a squared number"
        self.parent = parent
        
        self.weightMidway=int(parent.seeds[0].totalWeightCount/2.0)
    def crossover(self, SelectedSeeds):
        for seed,CrossoverSeeds in zip(self.parent.seeds,iterT.product(SelectedSeeds,repeat=2)):      
            seed.weights[:self.weightMidway]=(CrossoverSeeds[0]).weights[:self.weightMidway]
            seed.weights[self.weightMidway:]=(CrossoverSeeds[1]).weights[self.weightMidway:]
    
######Mutation
class GaussMutation:
    def __init__(self, parent: GeneticAlgorithm, averageNumberMutated, numberMutatedDecrease, mutationValueMean, mutationValueStd):
        self.seedCount=len(parent.seeds)
        self.seedCountLen=len(parent.seeds[0].weights)
        self.parent=parent
        self.averageNumberMutated=averageNumberMutated
        self.numberMutatedDecrease=numberMutatedDecrease
        self.mutationValueMean=mutationValueMean
        self.mutationValueStd=mutationValueStd
        
    def Mutation(self):
        numberToMutate=int(np.random.exponential(self.averageNumberMutated))
        self.averageNumberMutated-=self.numberMutatedDecrease
        MutationIndicies=np.random.randint(0,self.seedCount,numberToMutate)
        
        for i in MutationIndicies:
            mutationIndex=np.random.randint(0,self.seedCountLen)
            self.parent.seeds[int(i)].weights[mutationIndex]+=np.random.normal(self.mutationValueMean,self.mutationValueStd)
    
        
    
        