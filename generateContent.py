from os.path import join

def getDatasetHeader(dataset):
    return """
    \\subsection{{{dataset}}}
    This section displays the results obtained in the {dataset} dataset.
    """.format(**{"dataset":dataset})
def getAlgHeader(alg):
    return """
    \\subsubsection{{{alg}}}
    This section displays the results obtained using the {alg} algorithm.\\\\
    """.format(**{"alg":alg})
def getContent(dataset,alg,arch):
    return """
    $\\mathbf{{{arch2}}}$\\textbf{{ Architecture:}}\\\\
    Fig. \\ref{{tbl:out-{dataset}-{alg}-{arch}-avg}} shows the average and standard deviation of the test, training and validation errors. Fig. \\ref{{tbl:out-{dataset}-{alg}-{arch}-stat}} shows the median and quartiles of the test, training and validation errors based on the best epoch. Finally, Fig. \\ref{{tbl:out-{dataset}-{alg}-{arch}-stat2}} shows the median and quartiles of the test, training and validation errors per mini-batch.
    \\begin{{figure}}[H]
    	\\centering
    	\\includegraphics[width=1\\textwidth]{{"Outputs/{dataset}/{alg}/{arch}/{dataset}-{alg}-{arch}-Avg"}}
    	\\caption{{Graph of mean and standard deviation of errors vs epochs}}
    	\\label{{tbl:out-{dataset}-{alg}-{arch}-avg}}
    \\end{{figure}}
    \\begin{{figure}}[H]
    	\\centering
    	\\includegraphics[width=1\\textwidth]{{"Outputs/{dataset}/{alg}/{arch}/{dataset}-{alg}-{arch}-Stat"}}
    	\\caption{{Graph of test error vs epochs for the gradient based algorithms}}
    	\\label{{tbl:out-{dataset}-{alg}-{arch}-stat}}
    \\end{{figure}}
    \\begin{{figure}}[H]
    	\\centering
    	\\includegraphics[width=1\\textwidth]{{"Outputs/{dataset}/{alg}/{arch}/{dataset}-{alg}-{arch}-Stat2"}}
    	\\caption{{Graph of test error vs epochs for the gradient based algorithms}}
    	\\label{{tbl:out-{dataset}-{alg}-{arch}-stat2}}
    \\end{{figure}}\n\n
    """.format(**{"dataset":dataset,"arch":arch,"arch2":arch.replace('x','\\times'),"alg":alg})



datasets=['cancer','card','flare','gene','horse','heartc']
hiddenLayers=[[[4,2],[6]],  # Cancer
              [[4,4],[6]],  # Card
              [[4]],    # Flare
              [[4,2],[4,4],[9]],  # Gene
              [[4],[9]],    # Horse
              [[8,8],[6]]]  # Heartc
inputsOutputs=[[9,2],[51,2],[24,3],[120,3],[58,3],[35,2]]
algs=["ADAGRAD","Backprop with Momentum","back-propogation","GA1","GA2","GA3","iRPROP-","iRPROP+","PSO","QuickProp","RPROP-","RPROP+"]
   

outstring=""
for dataset, arch, io in zip(datasets,hiddenLayers, inputsOutputs):
    outstring+=getDatasetHeader(dataset)
    for alg in algs:
        outstring+=getAlgHeader(alg)
        for hl in arch:
            outstring+=getAlgHeader(dataset)
            l=[io[0]]+hl+[io[1]]
            layerstring=str(l[0])
            for v in l[1:]:
                layerstring+="x"+str(v)
            
            outstring+=getContent(dataset,alg,layerstring)
            
with open(join("Appendage.tex"), 'w') as f:
    f.write(outstring)