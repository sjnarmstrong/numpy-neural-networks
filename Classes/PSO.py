import numpy as np

class PSO:
    def __init__(self, Phi_g, Phi_p, omega, weights, getError, searchLow, searchUp):
        dataRange=searchUp-searchLow
        self.Phi_g=Phi_g
        self.Phi_p=Phi_p
        self.omega=omega
        self.bestPos=weights.copy()
        self.bestVals=np.repeat(float('inf'),weights.shape[0])

        self.particalVelocity=np.random.uniform(-dataRange,dataRange,weights.shape)
        self.getError=getError
        
    def iterate(self, inputData, targetData, weights):
        currentError=self.getError(inputData, targetData)
        minInds=np.where(currentError<self.bestVals)[0]
        self.bestVals[minInds]=currentError[minInds]
        self.bestPos[minInds]=weights[minInds]
        
        globalbestIndex=np.argmin(self.bestVals)
    
        r_p=np.random.uniform(0,self.Phi_p)
        self.r_g=np.random.uniform(0,self.Phi_g)
        
        self.particalVelocity=self.omega*self.particalVelocity+r_p*(self.bestPos-weights)+r_p*(self.bestPos[globalbestIndex]-weights)
        weights+=self.particalVelocity
        
        return self.bestVals[globalbestIndex]

    def getBestWeightsAndError(self):
        globalbestIndex=np.argmin(self.bestVals)
        
        return self.bestVals[globalbestIndex], self.bestPos[globalbestIndex]

def getError(inputData, targetData, weights):
    return np.abs(np.sum(weights,axis=1))