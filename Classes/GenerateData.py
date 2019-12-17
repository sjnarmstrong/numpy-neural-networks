from os.path import join, isfile
from os import getcwd
import numpy as np

Datasets={
 'flare':join('flare','flare1.dt'),
 'card':join('card','card1.dt'),
 'diabetes':join('diabetes','diabetes1.dt'),
 'heart':join('heart','heart1.dt'),
 'heartc':join('heart','heartc1.dt'),
 'hearta':join('heart','hearta1.dt'),
 'heartac':join('heart','heartac1.dt'),
 'glass':join('glass','glass1.dt'),
 'thyroid':join('thyroid','thyroid1.dt'),
 'gene':join('gene','gene1.dt'),
 'building':join('building','building1.dt'),
 'mushroom':join('mushroom','mushroom1.dt'),
 'soybean':join('soybean','soybean1.dt'),
 'horse':join('horse','horse1.dt'),
 'cancer':join('cancer','cancer1.dt')
}

class generateData:
    def __init__(self, dataset, basepath='proben1', reshape=False):
        path=join(basepath, Datasets[dataset])
        
        assert isfile(path), ("Dataset file could not be found. Please ensure"+
        " that the following path exists: \n" + join(getcwd(),path))
        
        with open(path, 'r') as f:
            boolIn=int(f.readline().split("=")[1])
            realIn=int(f.readline().split("=")[1])
            boolOut=int(f.readline().split("=")[1])
            realOut=int(f.readline().split("=")[1])
            self.TrainCount=int(f.readline().split("=")[1])
            self.ValidationCount=int(f.readline().split("=")[1])
            self.Vend=self.ValidationCount+self.TrainCount
            self.TestCount=int(f.readline().split("=")[1])
            
            self.numInputs=boolIn+realIn
            self.numOutputs=boolOut+realOut
            
            holdArray=np.fromfile(f,sep=" ").reshape(-1,self.numInputs+self.numOutputs)
            self.inputs,self.outputs=np.split(holdArray,[self.numInputs],axis=1)
            if reshape:
                self.inputs.shape+=(1,)
                self.outputs.shape+=(1,)
            self.randomizeIndicies = np.arange(len(self.inputs))
            
            self.booleanOuts=(realOut==0)
            self.oneHotOut=(len(self.outputs)==np.count_nonzero(self.outputs))
    def getDatasets(self):
        np.random.shuffle(self.randomizeIndicies)
        self.inputs=self.inputs[self.randomizeIndicies]
        self.outputs=self.outputs[self.randomizeIndicies]
        
        return (self.inputs[:self.TrainCount],self.outputs[:self.TrainCount],
            self.inputs[self.TrainCount:self.Vend],self.outputs[self.TrainCount:self.Vend],
            self.inputs[self.Vend:],self.outputs[self.Vend:])
