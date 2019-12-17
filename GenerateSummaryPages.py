from os.path import join

def getPage(dataset,arch):
    return """
    \\newpage
    $\\mathbf{{{arch2}}}$\\textbf{{ Architecture:}}\\\\
    \\begin{{figure}}[H]
    	\\centering
    	\\includegraphics[width=1\\textwidth]{{"Outputs/{dataset}/{dataset}-back-propogation-iRPROP+-QuickProp-Backprop with Momentum-ADAGRAD--{arch}Summary"}}
    	\\caption{{Graph of test error vs epochs for the gradient based algorithms}}
    \\end{{figure}}
    \\begin{{figure}}[H]
    	\\centering
    	\\includegraphics[width=1\\textwidth]{{"Outputs/{dataset}/{dataset}-RPROP--RPROP+-iRPROP--iRPROP+--{arch}Summary"}}
    	\\caption{{Graph of test error vs epochs for the various version of RPROP}}
    \\end{{figure}}
    \\begin{{figure}}[H]
    	\\centering
    	\\includegraphics[width=1\\textwidth]{{"Outputs/{dataset}/{dataset}-GA1-GA2-GA3--{arch}Summary"}}
    	\\caption{{Graph of test error vs epochs for the genetic algorithms}}
    \\end{{figure}}
    \\begin{{figure}}[H]
    	\\centering
    	\\includegraphics[width=1\\textwidth]{{"Outputs/{dataset}/{dataset}-PSO-GA1--{arch}Summary"}}
    	\\caption{{Graph comparing GA and PSO}}
    \\end{{figure}}
    \\input{{"Outputs/{dataset}/{dataset}-back-propogation-RPROP--RPROP+-iRPROP--iRPROP+-QuickProp-Backprop with Momentum-ADAGRAD-GA1-GA2-GA3-PSO--{arch}Errors"}}
    \\input{{"Outputs/{dataset}/{dataset}-back-propogation-RPROP--RPROP+-iRPROP--iRPROP+-QuickProp-Backprop with Momentum-ADAGRAD-GA1-GA2-GA3-PSO--{arch}TermEpochs"}}
    """.format(**{"dataset":dataset,"arch":arch,"arch2":arch.replace('x','\\times')})


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
        
        strHold=getPage(dataset,layerstring)
        with open(join("Pages",dataset+layerstring+"spage.tex"), 'w') as f:
            f.write(strHold)