import numpy as np
import matplotlib.pyplot as plt
from os.path import join, exists
from os import makedirs
from math import ceil

plt.rcParams.update({'font.size': 13})

def reportStats(TeErrors,VErrors,TrErrors, bestGenerationValidationErrors, bestGenerationEpochs, bestGenerationTimes, bestGenerationWeightedEpochs, bestGenerationTestErrors, GenerationStopEpochs, algName, dataset, networkStructure, invMiniBatchSize=1, reportGenerateion=False):
    
    baseDir=join("Outputs",dataset,algName,networkStructure)
    
    if not exists(baseDir):
        makedirs(baseDir)
        
    TeErrors=np.array(TeErrors)
    VErrors=np.array(VErrors)
    TrErrors=np.array(TrErrors)
    
    np.savetxt(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-TeErrors.gz"),TeErrors)
    np.savetxt(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-VErrors.gz"),VErrors)
    np.savetxt(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-TrErrors.gz"),TrErrors)
    np.savetxt(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-Loop.gz"),np.arange(len(TeErrors[0]))*invMiniBatchSize)
    
    AvgTeErrors=np.nanmean(TeErrors,axis=0)
    AvgVErrors=np.nanmean(VErrors,axis=0)
    AvgTrErrors=np.nanmean(TrErrors,axis=0)
    
    StdTeErrors=np.nanstd(TeErrors,axis=0)
    StdVErrors=np.nanstd(VErrors,axis=0)
    StdTrErrors=np.nanstd(TrErrors,axis=0)
    
    
    indexM=bestGenerationEpochs.index(np.percentile(bestGenerationEpochs,50,interpolation='nearest'))
    indexUpper=bestGenerationEpochs.index(np.percentile(bestGenerationEpochs,25,interpolation='nearest'))
    indexLower=bestGenerationEpochs.index(np.percentile(bestGenerationEpochs,75,interpolation='nearest'))
    
    indexUpperQuartile=ceil(np.percentile(GenerationStopEpochs,75)/invMiniBatchSize)
    
    epochs=np.arange(len(AvgTrErrors))*invMiniBatchSize
    f, axarr = plt.subplots(1, sharex=True,figsize=(15,8))
    
    if reportGenerateion:
        axarr.set_xlabel("Generation")
    else:
        axarr.set_xlabel("Epochs")  
    axarr.set_ylabel("Average Squared Error")
    axarr.set_title("Average Error vs Epochs for "+algName+" on the "+dataset+" dataset using a "+networkStructure+" FFNN")
    
    line, = axarr.plot(epochs[:indexUpperQuartile],AvgTrErrors[:indexUpperQuartile], color='g', lw=1, label="Average Training Error")
    line, = axarr.plot(epochs[:indexUpperQuartile],AvgTrErrors[:indexUpperQuartile]+StdTrErrors[:indexUpperQuartile], color='g', lw=1, ls="--", label="Training Error Std Dev.")
    line, = axarr.plot(epochs[:indexUpperQuartile],AvgTrErrors[:indexUpperQuartile]-StdTrErrors[:indexUpperQuartile], color='g', lw=1, ls="--")
    
    line, = axarr.plot(epochs[:indexUpperQuartile],AvgVErrors[:indexUpperQuartile], color='b', lw=1, label="Average Validation Error")
    line, = axarr.plot(epochs[:indexUpperQuartile],AvgVErrors[:indexUpperQuartile]+StdVErrors[:indexUpperQuartile], color='b', lw=1, ls="--", label="Test Validation Std Dev.")
    line, = axarr.plot(epochs[:indexUpperQuartile],AvgVErrors[:indexUpperQuartile]-StdVErrors[:indexUpperQuartile], color='b', lw=1, ls="--")
    
    line, = axarr.plot(epochs[:indexUpperQuartile],AvgTeErrors[:indexUpperQuartile], color='r', lw=1, label="Average Test Error")
    line, = axarr.plot(epochs[:indexUpperQuartile],AvgTeErrors[:indexUpperQuartile]+StdTeErrors[:indexUpperQuartile], color='r', lw=1, ls="--", label="Test Error Std Dev.")
    line, = axarr.plot(epochs[:indexUpperQuartile],AvgTeErrors[:indexUpperQuartile]-StdTeErrors[:indexUpperQuartile], color='r', lw=1, ls="--")
    
    
    plt.legend()
    plt.savefig(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-Avg.pdf"), format='pdf', dpi=800,bbox_inches="tight")
    #plt.show()
    plt.close()
    
    
    
    
    f, axarr = plt.subplots(1, sharex=True,figsize=(15,8))
    
    if reportGenerateion:
        axarr.set_xlabel("Generation")
    else:
        axarr.set_xlabel("Epochs") 
    axarr.set_ylabel("Average Squared Error")
    axarr.set_title("Error Statistics vs Epochs for "+algName+" on the "+dataset+" dataset using a "+networkStructure+" FFNN")
    
    epochsM=np.arange(len(TeErrors[indexM]))*invMiniBatchSize
    epochsU=np.arange(len(TeErrors[indexUpper]))*invMiniBatchSize
    epochsL=np.arange(len(TeErrors[indexLower]))*invMiniBatchSize
    
    line, = axarr.plot(epochsM[:indexUpperQuartile],np.percentile(TrErrors,50,axis=0)[:indexUpperQuartile], color='g', lw=1, label="Median Training Error")
    line, = axarr.plot(epochsU[:indexUpperQuartile],np.percentile(TrErrors,25,axis=0)[:indexUpperQuartile], color='g', lw=1, ls="--", label="Upper Quartile Test Error")
    line, = axarr.plot(epochsL[:indexUpperQuartile],np.percentile(TrErrors,75,axis=0)[:indexUpperQuartile], color='g', lw=1, ls=":", label="Lower Quartile Test Error") 
    
    line, = axarr.plot(epochsM[:indexUpperQuartile],np.percentile(VErrors,50,axis=0)[:indexUpperQuartile], color='b', lw=1, label="Median Validation Error")
    line, = axarr.plot(epochsU[:indexUpperQuartile],np.percentile(VErrors,25,axis=0)[:indexUpperQuartile], color='b', lw=1, ls="--", label="Upper Quartile Test Error")
    line, = axarr.plot(epochsL[:indexUpperQuartile],np.percentile(VErrors,75,axis=0)[:indexUpperQuartile], color='b', lw=1, ls=":", label="Lower Quartile Test Error") 
    
    line, = axarr.plot(epochsM[:indexUpperQuartile],np.percentile(TeErrors,50,axis=0)[:indexUpperQuartile], color='r', lw=1, label="Median Test Error")
    line, = axarr.plot(epochsU[:indexUpperQuartile],np.percentile(TeErrors,25,axis=0)[:indexUpperQuartile], color='r', lw=1, ls="--", label="Upper Quartile Test Error")
    line, = axarr.plot(epochsL[:indexUpperQuartile],np.percentile(TeErrors,75,axis=0)[:indexUpperQuartile], color='r', lw=1, ls=":", label="Lower Quartile Test Error")
    
    
    plt.legend()
    plt.savefig(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-Stat2.pdf"), format='pdf', dpi=800,bbox_inches="tight")
    #plt.show()
    plt.close()
    
    
    
    
    f, axarr = plt.subplots(1, sharex=True,figsize=(15,8))
    
    if reportGenerateion:
        axarr.set_xlabel("Generation")
    else:
        axarr.set_xlabel("Epochs") 
    axarr.set_ylabel("Average Squared Error")
    axarr.set_title("Error Statistics vs Epochs for "+algName+" on the "+dataset+" dataset using a "+networkStructure+" FFNN")
    
    epochsM=np.arange(len(TeErrors[indexM]))*invMiniBatchSize
    epochsU=np.arange(len(TeErrors[indexUpper]))*invMiniBatchSize
    epochsL=np.arange(len(TeErrors[indexLower]))*invMiniBatchSize
    
    line, = axarr.plot(epochsM[:indexUpperQuartile],TrErrors[indexM][:indexUpperQuartile], color='g', lw=1, label="Median Training Error")
    line, = axarr.plot(epochsU[:indexUpperQuartile],TrErrors[indexUpper][:indexUpperQuartile], color='g', lw=1, ls="--", label="Upper Quartile Test Error")
    line, = axarr.plot(epochsL[:indexUpperQuartile],TrErrors[indexLower][:indexUpperQuartile], color='g', lw=1, ls=":", label="Lower Quartile Test Error") 
    
    line, = axarr.plot(epochsM[:indexUpperQuartile],VErrors[indexM][:indexUpperQuartile], color='b', lw=1, label="Median Validation Error")
    line, = axarr.plot(epochsU[:indexUpperQuartile],VErrors[indexUpper][:indexUpperQuartile], color='b', lw=1, ls="--", label="Upper Quartile Test Error")
    line, = axarr.plot(epochsL[:indexUpperQuartile],VErrors[indexLower][:indexUpperQuartile], color='b', lw=1, ls=":", label="Lower Quartile Test Error") 
    
    line, = axarr.plot(epochsM[:indexUpperQuartile],TeErrors[indexM][:indexUpperQuartile], color='r', lw=1, label="Median Test Error")
    line, = axarr.plot(epochsU[:indexUpperQuartile],TeErrors[indexUpper][:indexUpperQuartile], color='r', lw=1, ls="--", label="Upper Quartile Test Error")
    line, = axarr.plot(epochsL[:indexUpperQuartile],TeErrors[indexLower][:indexUpperQuartile], color='r', lw=1, ls=":", label="Lower Quartile Test Error")
    
    
    plt.legend()
    plt.savefig(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-Stat.pdf"), format='pdf', dpi=800,bbox_inches="tight")
    #plt.show()
    plt.close()
    
    
    bestGenerationValidationErrors=np.array(bestGenerationValidationErrors)
    MbestGenerationValidationErrors=np.percentile(bestGenerationValidationErrors,50)
    UQbestGenerationValidationErrors=np.percentile(bestGenerationValidationErrors,75)
    LQbestGenerationValidationErrors=np.percentile(bestGenerationValidationErrors,25)
    UbestGenerationValidationErrors=np.percentile(bestGenerationValidationErrors,100)
    LbestGenerationValidationErrors=np.percentile(bestGenerationValidationErrors,0)
    
    MbestGenerationEpochs=np.percentile(bestGenerationEpochs,50)
    UQbestGenerationEpochs=np.percentile(bestGenerationEpochs,75)
    LQbestGenerationEpochs=np.percentile(bestGenerationEpochs,25)
    UbestGenerationEpochs=np.percentile(bestGenerationEpochs,100)
    LbestGenerationEpochs=np.percentile(bestGenerationEpochs,0)
    
    MbestGenerationTimes=np.percentile(bestGenerationTimes,50)
    UQbestGenerationTimes=np.percentile(bestGenerationTimes,75)
    LQbestGenerationTimes=np.percentile(bestGenerationTimes,25)
    UbestGenerationTimes=np.percentile(bestGenerationTimes,100)
    LbestGenerationTimes=np.percentile(bestGenerationTimes,0)
    
    MbestGenerationWeightedEpochs=np.percentile(bestGenerationWeightedEpochs,50)
    UQbestGenerationWeightedEpochs=np.percentile(bestGenerationWeightedEpochs,75)
    LQbestGenerationWeightedEpochs=np.percentile(bestGenerationWeightedEpochs,25)
    UbestGenerationWeightedEpochs=np.percentile(bestGenerationWeightedEpochs,100)
    LbestGenerationWeightedEpochs=np.percentile(bestGenerationWeightedEpochs,0)
    
    bestGenerationTestErrors=np.array(bestGenerationTestErrors)
    MbestGenerationTestErrors=np.percentile(bestGenerationTestErrors,50)
    UQbestGenerationTestErrors=np.percentile(bestGenerationTestErrors,75)
    LQbestGenerationTestErrors=np.percentile(bestGenerationTestErrors,25)
    UbestGenerationTestErrors=np.percentile(bestGenerationTestErrors,100)
    LbestGenerationTestErrors=np.percentile(bestGenerationTestErrors,0)
    
    MGenerationStopEpochs=np.percentile(GenerationStopEpochs,50)
    UQGenerationStopEpochs=np.percentile(GenerationStopEpochs,75)
    LQGenerationStopEpochs=np.percentile(GenerationStopEpochs,25)
    UGenerationStopEpochs=np.percentile(GenerationStopEpochs,100)
    LGenerationStopEpochs=np.percentile(GenerationStopEpochs,0)
    
    
    print("bestGenerationValidationError stats:")
    print("Median: "+str(MbestGenerationValidationErrors))
    print("Upper Quartile: "+str(UQbestGenerationValidationErrors))
    print("Lower Quartile: "+str(LQbestGenerationValidationErrors))
    print("Max: "+str(UbestGenerationValidationErrors))
    print("Min: "+str(LbestGenerationValidationErrors))
    print("___________________________________________________________")
    print("bestGenerationEpochs stats:")
    print("Median: "+str(MbestGenerationEpochs))
    print("Upper Quartile: "+str(UQbestGenerationEpochs))
    print("Lower Quartile: "+str(LQbestGenerationEpochs))
    print("Max: "+str(UbestGenerationEpochs))
    print("Min: "+str(LbestGenerationEpochs))
    print("___________________________________________________________")
    print("bestGenerationTimes stats:")
    print("Median: "+str(MbestGenerationTimes))
    print("Upper Quartile: "+str(UQbestGenerationTimes))
    print("Lower Quartile: "+str(LQbestGenerationTimes))
    print("Max: "+str(UbestGenerationTimes))
    print("Min: "+str(LbestGenerationTimes))
    print("___________________________________________________________")
    print("bestGenerationWeightedEpochs stats:")
    print("Median: "+str(MbestGenerationWeightedEpochs))
    print("Upper Quartile: "+str(UQbestGenerationWeightedEpochs))
    print("Lower Quartile: "+str(LQbestGenerationWeightedEpochs))
    print("Max: "+str(UbestGenerationWeightedEpochs))
    print("Min: "+str(LbestGenerationWeightedEpochs))
    print("___________________________________________________________")
    print("bestGenerationTestErrors stats:")
    print("Median: "+str(MbestGenerationTestErrors))
    print("Upper Quartile: "+str(UQbestGenerationTestErrors))
    print("Lower Quartile: "+str(LQbestGenerationTestErrors))
    print("Max: "+str(UbestGenerationTestErrors))
    print("Min: "+str(LbestGenerationTestErrors))
    print("___________________________________________________________")
    
    outString = """
    %bestGenerationValidationError
    \\addplot+[
    boxplot prepared={{
    	median={:.2f},
    	upper quartile={:.2f},
    	lower quartile={:.2f},
    	upper whisker={:.2f},
    	lower whisker={:.2f}
    }},
    ] coordinates {{}};
    %bestGenerationEpochs
    \\addplot+[
    boxplot prepared={{
    	median={:.2f},
    	upper quartile={:.2f},
    	lower quartile={:.2f},
    	upper whisker={:.2f},
    	lower whisker={:.2f}
    }},
    ] coordinates {{}};
    %bestGenerationTimes
    \\addplot+[
    boxplot prepared={{
    	median={:.2f},
    	upper quartile={:.2f},
    	lower quartile={:.2f},
    	upper whisker={:.2f},
    	lower whisker={:.2f}
    }},
    ] coordinates {{}};
    %bestGenerationWeightedEpochs
    \\addplot+[
    boxplot prepared={{
    	median={:.2f},
    	upper quartile={:.2f},
    	lower quartile={:.2f},
    	upper whisker={:.2f},
    	lower whisker={:.2f}
    }},
    ] coordinates {{}};
    %bestGenerationTestErrors
    \\addplot+[
    boxplot prepared={{
    	median={:.2f},
    	upper quartile={:.2f},
    	lower quartile={:.2f},
    	upper whisker={:.2f},
    	lower whisker={:.2f}
    }},
    ] coordinates {{}};
    %GenerationStopEpochs
    \\addplot+[
    boxplot prepared={{
    	median={:.2f},
    	upper quartile={:.2f},
    	lower quartile={:.2f},
    	upper whisker={:.2f},
    	lower whisker={:.2f}
    }},
    ] coordinates {{}};
    """.format(MbestGenerationValidationErrors,UQbestGenerationValidationErrors,LQbestGenerationValidationErrors,UbestGenerationValidationErrors,LbestGenerationValidationErrors,
               MbestGenerationEpochs,UQbestGenerationEpochs,LQbestGenerationEpochs,UbestGenerationEpochs,LbestGenerationEpochs,
               MbestGenerationTimes,UQbestGenerationTimes,LQbestGenerationTimes,UbestGenerationTimes,LbestGenerationTimes,
               MbestGenerationWeightedEpochs,UQbestGenerationWeightedEpochs,LQbestGenerationWeightedEpochs,UbestGenerationWeightedEpochs,LbestGenerationWeightedEpochs,
               MbestGenerationTestErrors,UQbestGenerationTestErrors,LQbestGenerationTestErrors,UbestGenerationTestErrors,LbestGenerationTestErrors,
               MGenerationStopEpochs,UQGenerationStopEpochs,LQGenerationStopEpochs,UGenerationStopEpochs,LGenerationStopEpochs)
    
    
    with open(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-Stat.dta"), 'w') as f:
        f.write(outString)
    
