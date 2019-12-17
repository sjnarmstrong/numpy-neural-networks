import numpy as np
import matplotlib.pyplot as plt
from os.path import join, exists
from os import makedirs
from math import ceil

plt.rcParams.update({'font.size': 13})
colors = ['#FF0000','#00FF00','#0000FF',	'#FFFF00','#00FFFF','#00FFFF','#FF00FF','#008080','#800080','#008000','#800000']
def PlotDetailedGraphs(dataset,algName,networkStructure, reportGenerateion=False):
    
    baseDir=join("Outputs",dataset,algName,networkStructure)
    
    TeErrors=np.loadtxt(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-TeErrors.gz"))
    VErrors=np.loadtxt(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-VErrors.gz"))
    TrErrors=np.loadtxt(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-TrErrors.gz"))
    epochs=np.loadtxt(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-Loop.gz"))
    
    invMiniBatchSize=epochs[1]
    
    indexUpperQuartile=open(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-Stat.dta")).read().splitlines()[55]
    indexUpperQuartile=float(indexUpperQuartile.split("=")[1].split(',')[0])
    indexUpperQuartile=int(indexUpperQuartile/invMiniBatchSize)
    
    AvgTeErrors=np.nanmean(TeErrors,axis=0)
    AvgVErrors=np.nanmean(VErrors,axis=0)
    AvgTrErrors=np.nanmean(TrErrors,axis=0)
    
    StdTeErrors=np.nanstd(TeErrors,axis=0)
    StdVErrors=np.nanstd(VErrors,axis=0)
    StdTrErrors=np.nanstd(TrErrors,axis=0)
    
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
    plt.savefig(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-Avg----.pdf"), format='pdf', dpi=800,bbox_inches="tight")
    #plt.show()
    plt.close()
    
    
    f, axarr = plt.subplots(1, sharex=True,figsize=(15,8))
    
    if reportGenerateion:
        axarr.set_xlabel("Generation")
    else:
        axarr.set_xlabel("Epochs") 
    axarr.set_ylabel("Average Squared Error")
    axarr.set_title("Error Statistics vs Epochs for "+algName+" on the "+dataset+" dataset using a "+networkStructure+" FFNN")
    
    line, = axarr.plot(epochs[:indexUpperQuartile],np.percentile(TrErrors,50,axis=0)[:indexUpperQuartile], color='g', lw=1, label="Median Training Error")
    line, = axarr.plot(epochs[:indexUpperQuartile],np.percentile(TrErrors,25,axis=0)[:indexUpperQuartile], color='g', lw=1, ls="--", label="Upper Quartile Test Error")
    line, = axarr.plot(epochs[:indexUpperQuartile],np.percentile(TrErrors,75,axis=0)[:indexUpperQuartile], color='g', lw=1, ls=":", label="Lower Quartile Test Error") 
    
    line, = axarr.plot(epochs[:indexUpperQuartile],np.percentile(VErrors,50,axis=0)[:indexUpperQuartile], color='b', lw=1, label="Median Validation Error")
    line, = axarr.plot(epochs[:indexUpperQuartile],np.percentile(VErrors,25,axis=0)[:indexUpperQuartile], color='b', lw=1, ls="--", label="Upper Quartile Test Error")
    line, = axarr.plot(epochs[:indexUpperQuartile],np.percentile(VErrors,75,axis=0)[:indexUpperQuartile], color='b', lw=1, ls=":", label="Lower Quartile Test Error") 
    
    line, = axarr.plot(epochs[:indexUpperQuartile],np.percentile(TeErrors,50,axis=0)[:indexUpperQuartile], color='r', lw=1, label="Median Test Error")
    line, = axarr.plot(epochs[:indexUpperQuartile],np.percentile(TeErrors,25,axis=0)[:indexUpperQuartile], color='r', lw=1, ls="--", label="Upper Quartile Test Error")
    line, = axarr.plot(epochs[:indexUpperQuartile],np.percentile(TeErrors,75,axis=0)[:indexUpperQuartile], color='r', lw=1, ls=":", label="Lower Quartile Test Error")
    
    
    plt.legend()
    plt.savefig(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-Stat2----.pdf"), format='pdf', dpi=800,bbox_inches="tight")
    #plt.show()
    plt.close()

def PlotSummaryGraphs(dataset,algNames,networkStructure, reportGenerateion=False):
    AverageTEErrors=[]
    epochsList=[]
    maxLen=[]
    minLen=[]
    
    ytick=str(list(range(1,len(algNames)+1))).replace('[','{').replace(']','}').replace('\'','')
    yticklables=str(algNames[::-1]).replace('[','{').replace(']','}').replace('\'','')
    ytick="ytick="+ytick+",\n"
    yticklables="yticklabels="+yticklables+",\n"
    header="\\begin{figure}[H]\n\\begin{tikzpicture}\n\\begin{axis}\n[\n"+ytick+yticklables+"width=0.8\\textwidth\n]\n"
    footer="\\end{axis}\n\\end{tikzpicture}\n"
    
    holdStringbestGenEpochs="\t%bestGenerationEpochs\n"
    holdStringbestGenTimes="\t%bestGenerationTimes\n"
    holdStringbestGenerationTestErrors="\t%bestGenerationTestErrors\n"
    holdStringGenerationStopEpochs="\t%GenerationStopEpochs\n"
    for algName in algNames:
        
        baseDir=join("Outputs",dataset,algName,networkStructure)
        
        TeErrors=np.loadtxt(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-TeErrors.gz"))
        AverageTEErrors.append(np.nanmean(TeErrors,axis=0))
        maxLen.append(len(AverageTEErrors[-1]))
        epochs=np.loadtxt(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-Loop.gz"))
        epochsList.append(epochs)
        infoFile=open(join(baseDir,dataset+"-"+algName+"-"+networkStructure+"-Stat.dta")).read().splitlines()
        indexUpperQuartile=float(infoFile[55].split("=")[1].split(',')[0])
        indexUpperQuartile=int(indexUpperQuartile/epochs[1])
        minLen.append(indexUpperQuartile)
        
        
        holdStringbestGenEpochs+='\n'.join(infoFile[12:21])+'\n'
        holdStringbestGenTimes+='\n'.join(infoFile[22:31])+'\n'
        holdStringbestGenerationTestErrors+='\n'.join(infoFile[42:51])+'\n'
        holdStringGenerationStopEpochs+='\n'.join(infoFile[52:61])+'\n'
    plotEnd=min(min(maxLen),max(minLen))
    
    f, axarr = plt.subplots(1, sharex=True,figsize=(15,8))
    
    if reportGenerateion:
        axarr.set_xlabel("Generation")
    else:
        axarr.set_xlabel("Epochs") 
    axarr.set_ylabel("Average Squared Error")
    axarr.set_title("Test Error vs Epochs summary on the "+dataset+" dataset using a "+networkStructure+" FFNN")
    
    for algname, epochs, AverageTeError, c in zip(algNames, epochsList, AverageTEErrors, colors):
        line, = axarr.plot(epochs[:plotEnd],AverageTeError[:plotEnd], color=c, lw=1, label=algname)

    plt.legend()
    holdname=dataset+"-"
    for alg in algNames:
        holdname+=alg+"-"
    plt.savefig(join("Outputs",dataset,holdname+"-"+networkStructure+"Summary.pdf"), format='pdf', dpi=800,bbox_inches="tight")
    #plt.show()
    plt.close()
    
    with open(join("Outputs",dataset,holdname+"-"+networkStructure+"Summary.dta"), 'w') as f:
        f.write("\n\n\n\n\n\n\n\n\n\n".join( [holdStringbestGenEpochs,holdStringbestGenTimes,
                                              holdStringbestGenerationTestErrors,holdStringGenerationStopEpochs]))
    
    holdStringbestGenEpochs=header+holdStringbestGenEpochs+footer+"\\caption{Number of epochs required to achieve best results on the validation dataset}\n\\end{figure}\n"
    holdStringbestGenTimes=header+holdStringbestGenTimes+footer+"\\caption{Time required to achieve best results on the validation dataset}\n\\end{figure}\n"
    holdStringbestGenerationTestErrors=header+holdStringbestGenerationTestErrors+footer+"\\caption{Best test errors achieved}\n\\end{figure}\n"
    holdStringGenerationStopEpochs=header+holdStringGenerationStopEpochs+footer+"\\caption{Number of epochs until algorithm terminated}\n\\end{figure}\n"
    with open(join("Outputs",dataset,holdname+"-"+networkStructure+"BestEpochs.tex"), 'w') as f:
        f.write(holdStringbestGenEpochs)
    with open(join("Outputs",dataset,holdname+"-"+networkStructure+"Time.tex"), 'w') as f:
        f.write(holdStringbestGenTimes)
    with open(join("Outputs",dataset,holdname+"-"+networkStructure+"Errors.tex"), 'w') as f:
        f.write(holdStringbestGenerationTestErrors)
    with open(join("Outputs",dataset,holdname+"-"+networkStructure+"TermEpochs.tex"), 'w') as f:
        f.write(holdStringGenerationStopEpochs)

datasets=['cancer','card','flare','gene','horse','heartc']
hiddenLayers=[[[4,2],[6]],  # Cancer
              [[4,4],[6]],  # Card
              [[4]],    # Flare
              [[4,2],[4,4],[9]],  # Gene
              [[4],[9]],    # Horse
              [[8,8],[6]]]  # Heartc
inputsOutputs=[[9,2],[51,2],[24,3],[120,3],[58,3],[35,2]]
   
for dataset, arch, io in zip(datasets,hiddenLayers, inputsOutputs):
    for hl in arch:
        l=[io[0]]+hl+[io[1]]
        layerstring=str(l[0])
        for v in l[1:]:
            layerstring+="x"+str(v)
    
        PlotSummaryGraphs(dataset,["GA1","GA2","GA3"],layerstring,reportGenerateion=True)
        PlotSummaryGraphs(dataset,["RPROP-","RPROP+","iRPROP-","iRPROP+"],layerstring)
        PlotSummaryGraphs(dataset,["back-propogation","iRPROP+","QuickProp","Backprop with Momentum","ADAGRAD"],layerstring)
        PlotSummaryGraphs(dataset,["PSO","GA1"],layerstring,reportGenerateion=True)
        PlotSummaryGraphs(dataset,["back-propogation","RPROP-","RPROP+","iRPROP-","iRPROP+","QuickProp","Backprop with Momentum","ADAGRAD","GA1","GA2","GA3","PSO"],layerstring)
#PlotSummaryGraphs("cancer",["RPROP+","RPROP-"],"9x4x2x2")
#PlotDetailedGraphs("cancer","RPROP+","9x4x2x2")
