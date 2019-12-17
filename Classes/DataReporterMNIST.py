import numpy as np
import matplotlib.pyplot as plt
from os.path import join, exists
from os import makedirs

plt.rcParams.update({'font.size': 13})

def reportStats(Weights,VErrors,TrErrors, bestGenerationValidationErrors, bestGenerationEpochs, bestGenerationTimes, bestGenerationWeightedEpochs, GenerationStopEpochs, algName, dataset, networkStructure, invMiniBatchSize=1):
    
    baseDir=join("Outputs",dataset,algName,networkStructure)
    
    if not exists(baseDir):
        makedirs(baseDir)
        
    VErrors=np.array(VErrors)
    TrErrors=np.array(TrErrors)
    
    np.savetxt(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-Weights.gz"),Weights)
    np.savetxt(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-VErrors.gz"),VErrors)
    np.savetxt(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-TrErrors.gz"),TrErrors)
    np.savetxt(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-bestGenerationValidationErrors.gz"),bestGenerationValidationErrors)
    np.savetxt(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-bestGenerationEpochs.gz"),bestGenerationEpochs)
    np.savetxt(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-bestGenerationTimes.gz"),bestGenerationTimes)
    np.savetxt(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-bestGenerationWeightedEpochs.gz"),bestGenerationWeightedEpochs)
    np.savetxt(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-GenerationStopEpochs.gz"),GenerationStopEpochs)
    
    AvgVErrors=np.nanmean(VErrors,axis=0)
    AvgTrErrors=np.nanmean(TrErrors,axis=0)
    
    epochs=np.arange(len(AvgTrErrors))*invMiniBatchSize
    f, axarr = plt.subplots(1, sharex=True,figsize=(15,8))
    
    axarr.set_xlabel("Epochs")  
    axarr.set_ylabel("Average Squared Error")
    axarr.set_title("Average Error vs Epochs for "+algName+" on the "+dataset+" dataset using an "+networkStructure+" FFNN")
    
    line, = axarr.plot(epochs,AvgTrErrors, color='g', lw=1, label="Average Training Error")
    
    line, = axarr.plot(epochs,AvgVErrors, color='b', lw=1, label="Average Validation Error")
    
    
    plt.legend()
    plt.savefig(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-Avg.pdf"), format='pdf', dpi=800,bbox_inches="tight")
    #plt.show()
    plt.close()
    
