import GeneticAlgorithmMatrix as GA
import numpy as np


'''

Start of unit tests for selection algorithms

'''

populationSize=16
chromesoneLen=5

testSeeds=np.repeat(np.arange(populationSize, dtype=np.float64),chromesoneLen).reshape((-1,chromesoneLen))

chromesoneLen2=8
testSeeds2=np.arange(populationSize*chromesoneLen2).reshape((-1,chromesoneLen2))

print("""
      #########################################################
      #       SelectionProbabilityWeights Tests               #
      #########################################################
      """)
'''

Tests on SelectionProbabilityWeights
This is a probabilistic algorithm and hence may occasionally fail tests
Test 1 ensures that seeds with high errors dont get selected and seeds with low errors do
Test 2 ensures that seeds with lowwer errors get selected first
Test 3 checks to see that items with equil weighting get selected equily as often and further checks that there is no selection mistakes
Test 4 checks that the probability of selection alighns with the error rate

'''

#Start Test 1

error=np.repeat(1000000000,populationSize)

idealSelected=[0,3,4,5,6]
idealNotSelected=np.delete(np.arange(populationSize),idealSelected)
error[idealSelected]=10000
error[0]=0
Selected=GA.SelectionProbabilityWeights(testSeeds, error, 5)

assert np.all(np.isin(idealSelected,Selected)) and np.all(np.isin(idealNotSelected,Selected)==False), "Test 1 Failed... NB this test can fail with a very low probability (~1/100000)"
print("Test 1 passed!!")
assert Selected[0,0]==0, "Test 2 Failed... NB this test can fail with a very low probability (~1/10000)"
print("Test 2 passed!!")

error=np.repeat(100000000000.0,populationSize)
error[0]=0
error[1]=0

count0=0
count1=0

for i in range(1000):
    Selected=GA.SelectionProbabilityWeights(testSeeds, error, 2)
    if (Selected[0,0]==0):
        count0+=1
        assert Selected[1,0] == 1, "Test 3 Failed... NB this test can fail with a very low probability (~1/100000000)"
    else:
        count1+=1
        assert Selected[1,0] == 0, "Test 3 Failed... NB this test can fail with a very low probability (~1/100000000)"

assert abs((count0/1000.0)-0.5)<0.1, "Test 3 Failed"
assert abs((count1/1000.0)-0.5)<0.1, "Test 3 Failed"

print("Test 3 passed!!")

pop2=np.repeat(np.arange(2),chromesoneLen).reshape((-1,chromesoneLen))
error=np.repeat(0,2)
error[0]=400
error[1]=600
count0=0
count1=0

for i in range(10000):
    Selected=GA.SelectionProbabilityWeights(pop2, error, 2)
    if (Selected[0,0]==0):
        count0+=1
    else:
        count1+=1    
assert abs((count0/10000.0)-0.6)<0.05, "Test 4 Failed"
assert abs((count1/10000.0)-0.4)<0.05, "Test 4 Failed"

print("Test 4 passed!!")

print("""
      #########################################################
      #                       TopN Tests                      #
      #########################################################
      """)
'''

Tests on TopN
This is a probabilistic algorithm and hence may occasionally fail tests
Test 1 ensures that seeds with high errors dont get selected and seeds with low errors do

'''

#Start Test 1

error=np.repeat(1000000000,populationSize)

idealSelected=[0,3,4,5,6]
idealNotSelected=np.delete(np.arange(populationSize),idealSelected)
error[idealSelected]=0
Selected=GA.TopN(testSeeds, error, 5)

assert np.all(np.isin(idealSelected,Selected)) and np.all(np.isin(idealNotSelected,Selected)==False), "Test 1 Failed..."
print("Test 1 passed!!")




'''

Start of unit tests for crossover classes

'''

print("""
      #########################################################
      #       _2NearestNeighborsCrossover Tests               #
      #########################################################
      """)

'''

Tests on _2NearestNeighborsCrossover
Test 1 checks is the algorithm is able to sucessfully preform chrossover on a chromesome with an even number of genes
Test 2 checks is the algorithm is able to sucessfully preform chrossover on a chromesome with an odd number of genes

'''

_2NN=GA._2NearestNeighborsCrossover(populationSize,chromesoneLen)

Selected=testSeeds[:_2NN.selectionCount].copy()
weightMat=testSeeds.copy()
_2NN.crossover(weightMat,Selected)
testMat=np.array(
        [[0,0,0,7,7],
        [0,0,0,1,1],
        [1,1,1,0,0],
        [1,1,1,2,2],
        [2,2,2,1,1],
        [2,2,2,3,3],
        [3,3,3,2,2],
        [3,3,3,4,4],
        [4,4,4,3,3],
        [4,4,4,5,5],
        [5,5,5,4,4],
        [5,5,5,6,6],
        [6,6,6,5,5],
        [6,6,6,7,7],
        [7,7,7,6,6],
        [7,7,7,0,0]])
    
assert np.all(weightMat==testMat),"Test 1 failed..."
print("Test 1 passed!!")

_2NN=GA._2NearestNeighborsCrossover(populationSize,chromesoneLen2)

Selected=testSeeds2[:_2NN.selectionCount].copy()
weightMat=testSeeds2.copy()
_2NN.crossover(weightMat,Selected)

testMat=np.array([[ 0,  1,  2,  3, 60, 61, 62, 63],
       [ 0,  1,  2,  3, 12, 13, 14, 15],
       [ 8,  9, 10, 11,  4,  5,  6,  7],
       [ 8,  9, 10, 11, 20, 21, 22, 23],
       [16, 17, 18, 19, 12, 13, 14, 15],
       [16, 17, 18, 19, 28, 29, 30, 31],
       [24, 25, 26, 27, 20, 21, 22, 23],
       [24, 25, 26, 27, 36, 37, 38, 39],
       [32, 33, 34, 35, 28, 29, 30, 31],
       [32, 33, 34, 35, 44, 45, 46, 47],
       [40, 41, 42, 43, 36, 37, 38, 39],
       [40, 41, 42, 43, 52, 53, 54, 55],
       [48, 49, 50, 51, 44, 45, 46, 47],
       [48, 49, 50, 51, 60, 61, 62, 63],
       [56, 57, 58, 59, 52, 53, 54, 55],
       [56, 57, 58, 59,  4,  5,  6,  7]])
    
    
assert np.all(weightMat==testMat),"Test 2 failed..."
print("Test 2 passed!!")




print("""
      #########################################################
      #                    allCombos Tests                    #
      #########################################################
      """)

'''

Tests on allCombos
Test 1 checks is the algorithm is able to sucessfully preform chrossover on a chromesome with an even number of genes
Test 2 checks is the algorithm is able to sucessfully preform chrossover on a chromesome with an odd number of genes

'''

allComboCO=GA.allCombos(populationSize,chromesoneLen)

Selected=testSeeds[:allComboCO.selectionCount].copy()
weightMat=testSeeds.copy()
allComboCO.crossover(weightMat,Selected)
testMat=np.array([[0, 0, 0, 0, 0],
       [0, 0, 0, 1, 1],
       [0, 0, 0, 2, 2],
       [0, 0, 0, 3, 3],
       [1, 1, 1, 0, 0],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 2, 2],
       [1, 1, 1, 3, 3],
       [2, 2, 2, 0, 0],
       [2, 2, 2, 1, 1],
       [2, 2, 2, 2, 2],
       [2, 2, 2, 3, 3],
       [3, 3, 3, 0, 0],
       [3, 3, 3, 1, 1],
       [3, 3, 3, 2, 2],
       [3, 3, 3, 3, 3]])
    
assert np.all(weightMat==testMat),"Test 1 failed..."
print("Test 1 passed!!")

allComboCO=GA.allCombos(populationSize,chromesoneLen2)

Selected=testSeeds2[:allComboCO.selectionCount].copy()
weightMat=testSeeds2.copy()
allComboCO.crossover(weightMat,Selected)

testMat=np.array([[ 0,  1,  2,  3,  4,  5,  6,  7],
       [ 0,  1,  2,  3, 12, 13, 14, 15],
       [ 0,  1,  2,  3, 20, 21, 22, 23],
       [ 0,  1,  2,  3, 28, 29, 30, 31],
       [ 8,  9, 10, 11,  4,  5,  6,  7],
       [ 8,  9, 10, 11, 12, 13, 14, 15],
       [ 8,  9, 10, 11, 20, 21, 22, 23],
       [ 8,  9, 10, 11, 28, 29, 30, 31],
       [16, 17, 18, 19,  4,  5,  6,  7],
       [16, 17, 18, 19, 12, 13, 14, 15],
       [16, 17, 18, 19, 20, 21, 22, 23],
       [16, 17, 18, 19, 28, 29, 30, 31],
       [24, 25, 26, 27,  4,  5,  6,  7],
       [24, 25, 26, 27, 12, 13, 14, 15],
       [24, 25, 26, 27, 20, 21, 22, 23],
       [24, 25, 26, 27, 28, 29, 30, 31]])
    
    
assert np.all(weightMat==testMat),"Test 2 failed..."
print("Test 2 passed!!")





'''

Start of unit tests for mutation classes

'''

print("""
      #########################################################
      #                   GaussMutation Tests                 #
      #########################################################
      """)


weightMat=np.zeros((populationSize, chromesoneLen), dtype=np.float64)
GM=GA.GaussMutation(populationSize,chromesoneLen,5,0,4,1000)

GM.Mutation(weightMat)
print(weightMat)
print("""
      #########################################################
      #                  SimpleMutation Tests                 #
      #########################################################
      """)

weightMat=np.zeros((populationSize, chromesoneLen), dtype=np.float64)
SM=GA.SimpleMutation(populationSize, chromesoneLen, 5, 1000)

SM.Mutation(weightMat)
assert np.count_nonzero(weightMat) == 5, "Test 1 failed"
print("Test 1 passed!!")
print("""
      #########################################################
                        DampedMutation Tests                 #
      #########################################################
      """)

weightMat=np.zeros((populationSize, chromesoneLen), dtype=np.float64)
DM=GA.DampedMutation(populationSize, chromesoneLen, 5, 1000, 0.01, 1)

DM.Mutation(weightMat)
print(weightMat)
DM.Mutation(weightMat)
print(weightMat)
