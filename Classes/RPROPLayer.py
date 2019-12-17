from .NeuralNetworkClass import BatchInputNetworkLayer
import numpy as np

class iRProp_plus_Layer(BatchInputNetworkLayer):
    def __init__(self, InputDimentions, OutputDimentions, WeightArray, BiasArray, eta_minus=0.5, eta_plus=1.2, delta_min=0, delta_max=50, delta_init=0.0125):
        BatchInputNetworkLayer.__init__(self, InputDimentions, OutputDimentions, WeightArray, BiasArray)
        
        self.NumberOfWeights=self.WeightMatrixT.size+self.BiasVector.size
        self.EndWeightMat=self.WeightMatrixT.size
        
        self.dE_dw_t_m1=np.zeros(self.NumberOfWeights)
        
        self.eta_minus=eta_minus
        self.eta_plus=eta_plus
        self.delta_min=delta_min
        self.delta_max=delta_max
        self.delta=np.random.uniform(0.005,0.02,self.NumberOfWeights)
        
        self.delta_W=-self.delta.copy()
        self.prevErr=0
        self.currentErr=0
    
    def backwardPropagate(self, relevantDeltaData, learningRate):

        deltaM = self.calcDelta(relevantDeltaData)
        holdOut = np.dot(deltaM, self.WeightMatrixT)
        
        dE_dw_t=np.append(np.dot(deltaM.T, self.prevX),np.sum(deltaM, axis=0))
        prevDw_times_Dw=dE_dw_t*self.dE_dw_t_m1
        prevBiggerZero=prevDw_times_Dw>0
        prevLessZero=prevDw_times_Dw<0
        
        deltaUpdate=np.where(prevBiggerZero, self.eta_plus,
                 np.where(prevLessZero,self.eta_minus,1.0))
        self.delta=np.clip(self.delta*deltaUpdate,self.delta_min,self.delta_max)
        
        self.delta_W=np.where(prevLessZero, self.delta_W if self.currentErr>self.prevErr else 0, np.sign(dE_dw_t)*self.delta)
        
        self.WeightMatrixT.flat -= self.delta_W[:self.EndWeightMat]
        self.BiasVector.flat -= self.delta_W[self.EndWeightMat:]
     
        self.dE_dw_t_m1=np.where(prevLessZero,0,dE_dw_t)
            
        return holdOut
        
#    def updateWeights(self):
#        
#        prevDw_times_Dw=self.dE_dw_t*self.dE_dw_t_m1
#        prevBiggerZero=prevDw_times_Dw>0
#        prevLessZero=prevDw_times_Dw<0
#
#        deltaUpdate=np.where(prevBiggerZero, self.eta_plus,
#                 np.where(prevLessZero,self.eta_minus,1.0))
#        self.delta=np.clip(self.delta*deltaUpdate,self.delta_min,self.delta_max)
#        
#        self.delta_W=np.where(prevLessZero, self.delta_W if self.currentErr>self.prevErr else 0, np.sign(self.dE_dw_t)*self.delta)
#        
#        self.neuralNetwork.weights-= self.delta_W
#        
#        self.dE_dw_t=np.where(prevLessZero,0,self.dE_dw_t)
#        self.resetDE_Dw()