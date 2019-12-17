import numpy as np
from .NeuralNetworkClass import NeuralNetwork

class baseRPROP:
    def __init__(self, neuralNetwork: NeuralNetwork, eta_minus=0.5, eta_plus=1.2, delta_min=0, delta_max=50, delta_init=0.0125):
        assert isinstance(neuralNetwork, NeuralNetwork), "This class expects instance of NeuralNetwork to be used"
        self.totalWeightCount=len(neuralNetwork.weights)
        
        #Allocate space for weights now as apposed to multiple times later
        self.dE_dw_t=np.zeros(self.totalWeightCount) 
        self.dE_dw_t_m1=np.zeros(self.totalWeightCount) 
        
        self.neuralNetwork=neuralNetwork
        
        self.eta_minus=eta_minus
        self.eta_plus=eta_plus
        self.delta_min=delta_min
        self.delta_max=delta_max
        self.delta=np.random.uniform(0.005,0.02,self.totalWeightCount)
        #np.repeat(delta_init,self.totalWeightCount)
        
    def populateDE_Dw(self, inputVector, targetVector):
        for layer in self.neuralNetwork.layers:
            inputVector=layer.zStoreForwardPropagate(inputVector)
            
        endPosition=self.totalWeightCount
        relivantDeltaData=targetVector
        for layer in reversed(self.neuralNetwork.layers):
            dE_DTheta=layer.calcDelta(relivantDeltaData)
            
            startPosition=endPosition-len(dE_DTheta)
            self.dE_dw_t[startPosition:endPosition]=dE_DTheta.flat
            endPosition=startPosition
            
            dE_DW=np.dot(dE_DTheta,layer.prevX.T)
            startPosition=endPosition-len(dE_DW.flat)
            self.dE_dw_t[startPosition:endPosition]=dE_DW.flat
            endPosition=startPosition
            
            relivantDeltaData=np.dot(layer.WeightMatrixT.T,dE_DTheta)
            
class baseRPROP_batch(baseRPROP):
    def __init__(self, neuralNetwork: NeuralNetwork, eta_minus=0.5, eta_plus=1.2, delta_min=0, delta_max=50, delta_init=0.0125):
        baseRPROP.__init__(self, neuralNetwork, eta_minus, eta_plus, delta_min, delta_max, delta_init)
        self.updateDE_Dw=self.populateDE_Dw
        
    def populateDE_Dw(self, inputVector, targetVector):
        super().populateDE_Dw(inputVector, targetVector)
        self.updateDE_Dw=self.addToDE_Dw
        
    def addToDE_Dw(self, inputVector, targetVector):
        for layer in self.neuralNetwork.layers:
            inputVector=layer.zStoreForwardPropagate(inputVector)
            
        endPosition=self.totalWeightCount
        relivantDeltaData=targetVector
        for layer in reversed(self.neuralNetwork.layers):
            dE_DTheta=layer.calcDelta(relivantDeltaData)
            
            startPosition=endPosition-len(dE_DTheta)
            self.dE_dw_t[startPosition:endPosition]+=dE_DTheta.flat
            endPosition=startPosition
            
            dE_DW=np.dot(dE_DTheta,layer.prevX.T)
            startPosition=endPosition-len(dE_DW.flat)
            self.dE_dw_t[startPosition:endPosition]+=dE_DW.flat
            endPosition=startPosition
            
            relivantDeltaData=np.dot(layer.WeightMatrixT.T,dE_DTheta)
            
    def resetDE_Dw(self):
        hold=self.dE_dw_t_m1
        self.dE_dw_t_m1=self.dE_dw_t
        self.dE_dw_t=hold
        self.updateDE_Dw=self.populateDE_Dw
        
class RProp_minus(baseRPROP):
    def __init__(self, neuralNetwork: NeuralNetwork, eta_minus=0.5, eta_plus=1.2, delta_min=0, delta_max=50, delta_init=0.0125):
        baseRPROP.__init__(self, neuralNetwork, eta_minus, eta_plus, delta_min, delta_max, delta_init)
    
    def updateWeights(self, inputVector, targetVector):
        self.populateDE_Dw(inputVector, targetVector)
        prevDw_times_Dw=self.dE_dw_t*self.dE_dw_t_m1

        #implement prevDw_times_Dw>0 ->self.eta_plus
        #prevDw_times_Dw<0 ->self.eta_minus
        #else 1.0
        deltaUpdate=np.where(prevDw_times_Dw>0, self.eta_plus,
                 np.where(prevDw_times_Dw<0,self.eta_minus,1.0))
        self.delta=np.clip(self.delta*deltaUpdate,self.delta_min,self.delta_max)
        
        self.neuralNetwork.weights-= np.sign(self.dE_dw_t)*self.delta
        
        hold=self.dE_dw_t_m1
        self.dE_dw_t_m1=self.dE_dw_t
        self.dE_dw_t=hold
        
        

class RProp_minus_batch(baseRPROP_batch):
    def __init__(self, neuralNetwork: NeuralNetwork, eta_minus=0.5, eta_plus=1.2, delta_min=0, delta_max=50, delta_init=0.0125):
        baseRPROP_batch.__init__(self, neuralNetwork, eta_minus, eta_plus, delta_min, delta_max, delta_init)
        
    def updateWeights(self):
        
        prevDw_times_Dw=self.dE_dw_t*self.dE_dw_t_m1
        #implement prevDw_times_Dw>0 ->self.eta_plus
        #prevDw_times_Dw<0 ->self.eta_minus
        #else 1.0
        deltaUpdate=np.where(prevDw_times_Dw>0, self.eta_plus,
                 np.where(prevDw_times_Dw<0,self.eta_minus,1.0))
        self.delta=np.clip(self.delta*deltaUpdate,self.delta_min,self.delta_max)
        
        self.neuralNetwork.weights-= np.sign(self.dE_dw_t)*self.delta
        
        self.resetDE_Dw()
            

class RProp_plus_batch(baseRPROP_batch):
    def __init__(self, neuralNetwork: NeuralNetwork, eta_minus=0.5, eta_plus=1.2, delta_min=0, delta_max=50, delta_init=0.0125):
        baseRPROP_batch.__init__(self, neuralNetwork, eta_minus, eta_plus, delta_min, delta_max, delta_init)
        self.delta_W=-self.delta.copy()
        
    def updateWeights(self):
        
        prevDw_times_Dw=self.dE_dw_t*self.dE_dw_t_m1
        prevBiggerZero=prevDw_times_Dw>0
        prevLessZero=prevDw_times_Dw<0

        deltaUpdate=np.where(prevBiggerZero, self.eta_plus,
                 np.where(prevLessZero,self.eta_minus,1.0))
        self.delta=np.clip(self.delta*deltaUpdate,self.delta_min,self.delta_max)
        
        self.delta_W=np.where(prevLessZero, self.delta_W, np.sign(self.dE_dw_t)*self.delta)
        
        self.neuralNetwork.weights-= self.delta_W
        
        self.dE_dw_t=np.where(prevLessZero,0,self.dE_dw_t)
        self.resetDE_Dw()
        

class iRProp_minus_batch(baseRPROP_batch):
    def __init__(self, neuralNetwork: NeuralNetwork, eta_minus=0.5, eta_plus=1.2, delta_min=0, delta_max=50, delta_init=0.0125):
        baseRPROP_batch.__init__(self, neuralNetwork, eta_minus, eta_plus, delta_min, delta_max, delta_init)
        
    def updateWeights(self):
        
        prevDw_times_Dw=self.dE_dw_t*self.dE_dw_t_m1
        prevLessZero=prevDw_times_Dw<0
        #implement prevDw_times_Dw>0 ->self.eta_plus
        #prevDw_times_Dw<0 ->self.eta_minus
        #else 1.0
        deltaUpdate=np.where(prevDw_times_Dw>0, self.eta_plus,
                 np.where(prevLessZero,self.eta_minus,1.0))
        self.delta=np.clip(self.delta*deltaUpdate,self.delta_min,self.delta_max)
        
        self.dE_dw_t=np.where(prevLessZero,0,self.dE_dw_t)
        
        self.neuralNetwork.weights-= np.sign(self.dE_dw_t)*self.delta
        
        self.resetDE_Dw()
        
class iRProp_plus_batch(baseRPROP_batch):
    def __init__(self, neuralNetwork: NeuralNetwork, eta_minus=0.5, eta_plus=1.2, delta_min=0, delta_max=50, delta_init=0.0125):
        baseRPROP_batch.__init__(self, neuralNetwork, eta_minus, eta_plus, delta_min, delta_max, delta_init)
        self.delta_W=-self.delta.copy()
        self.prevErr=0
        self.currentErr=0
        
    def populateDE_Dw(self, inputVector, targetVector):
        super().populateDE_Dw(inputVector, targetVector)
        self.currentErr=NeuralNetwork.getError(self.neuralNetwork.forwardPropagate(inputVector),targetVector)
        
    def addToDE_Dw(self, inputVector, targetVector):
        super().addToDE_Dw(inputVector, targetVector)
        self.currentErr+=NeuralNetwork.getError(self.neuralNetwork.forwardPropagate(inputVector),targetVector)
    
    def resetDE_Dw(self):  
        super().resetDE_Dw()
        self.prevErr=self.currentErr
        
    def updateWeights(self):
        
        prevDw_times_Dw=self.dE_dw_t*self.dE_dw_t_m1
        prevBiggerZero=prevDw_times_Dw>0
        prevLessZero=prevDw_times_Dw<0

        deltaUpdate=np.where(prevBiggerZero, self.eta_plus,
                 np.where(prevLessZero,self.eta_minus,1.0))
        self.delta=np.clip(self.delta*deltaUpdate,self.delta_min,self.delta_max)
        
        self.delta_W=np.where(prevLessZero, self.delta_W if self.currentErr>self.prevErr else 0, np.sign(self.dE_dw_t)*self.delta)
        
        self.neuralNetwork.weights-= self.delta_W
        
        self.dE_dw_t=np.where(prevLessZero,0,self.dE_dw_t)
        self.resetDE_Dw()