from os.path import join, isfile
from os import getcwd
import numpy as np

TestFiles=['t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte']
TrainFiles=['train-images-idx3-ubyte','train-labels-idx1-ubyte']

class generateData:
    def __init__(self, basepath='mnist', precentValidation=10):
        
        dataPath=join(basepath, TrainFiles[0])
        labelPath=join(basepath, TrainFiles[1])
        
        assert isfile(dataPath), ("Dataset file could not be found. Please ensure"+
        " that the following path exists: \n" + join(getcwd(),dataPath))
        
        assert isfile(labelPath), ("Dataset file could not be found. Please ensure"+
        " that the following path exists: \n" + join(getcwd(),labelPath))        
        
        with open(dataPath, 'rb') as imageFile:
          with open(labelPath, 'rb') as labelFile:  
              dt = np.dtype(np.int32)
              dt = dt.newbyteorder('>')
             
              
              np.frombuffer(labelFile.read(4), dtype=dt)
              self.numDigets=np.frombuffer(labelFile.read(4), dtype=dt)[0]
              lbl = np.fromfile(labelFile, dtype=np.int8)
              
              labels=np.zeros((self.numDigets,10))
              labels[np.arange(self.numDigets),lbl]=1
              
              np.frombuffer(imageFile.read(4), dtype=dt)
              np.frombuffer(imageFile.read(4), dtype=dt)
              self.numInputs=np.product(np.frombuffer(imageFile.read(8), dtype=dt))
              
              images=np.fromfile(imageFile, dtype=np.uint8).reshape(np.append(-1,self.numInputs))

        self.ValCount=int(precentValidation*self.numDigets/100)
        self.TrainCount=self.numDigets-self.ValCount
        self.randomizeIndicies = np.arange(self.TrainCount)
        
        self.ValData=images[:self.ValCount]
        self.ValLabels=labels[:self.ValCount]
        self.TrainData=images[self.ValCount:]
        self.TrainLabels=labels[self.ValCount:]
        
        
        self.Reshuffel()
        self.getTest(basepath=basepath)
        
    def Reshuffel(self):        
        randomizeIndicies = np.arange(self.numDigets)
        np.random.shuffle(randomizeIndicies)
        images=np.append(self.ValData,self.TrainData,axis=0)[randomizeIndicies]
        labels=np.append(self.ValLabels,self.TrainLabels,axis=0)[randomizeIndicies]
        self.ValData=images[:self.ValCount]
        self.ValLabels=labels[:self.ValCount]
        self.TrainData=images[self.ValCount:]
        self.TrainLabels=labels[self.ValCount:]
        
    def GetNewTrainWithNoise(self):
        np.random.shuffle(self.randomizeIndicies)
        HoldOutImages=self.TrainData[self.randomizeIndicies]
        HoldOutLables=self.TrainLabels[self.randomizeIndicies]
        return np.abs(HoldOutImages+np.random.normal(0,15,HoldOutImages.shape)),HoldOutLables

    def getTest(self, basepath='mnist'):
        
        dataPath=join(basepath, TestFiles[0])
        labelPath=join(basepath, TestFiles[1])
        
        assert isfile(dataPath), ("Dataset file could not be found. Please ensure"+
        " that the following path exists: \n" + join(getcwd(),dataPath))
        
        assert isfile(labelPath), ("Dataset file could not be found. Please ensure"+
        " that the following path exists: \n" + join(getcwd(),labelPath))        
        
        with open(dataPath, 'rb') as imageFile:
          with open(labelPath, 'rb') as labelFile:  
              dt = np.dtype(np.int32)
              dt = dt.newbyteorder('>')
             
              
              np.frombuffer(labelFile.read(4), dtype=dt)
              numDigets=np.frombuffer(labelFile.read(4), dtype=dt)[0]
              self.Testlbl = np.fromfile(labelFile, dtype=np.int8)
              
              self.Testlabels=np.zeros((numDigets,10))
              self.Testlabels[np.arange(numDigets),self.Testlbl]=1
              
              np.frombuffer(imageFile.read(4), dtype=dt)
              np.frombuffer(imageFile.read(4), dtype=dt)
              imageDimention=np.product(np.frombuffer(imageFile.read(8), dtype=dt))
              
              self.TestImages=np.fromfile(imageFile, dtype=np.uint8).reshape(np.append(-1,imageDimention))  
        self.TestCount=len(self.TestImages)
