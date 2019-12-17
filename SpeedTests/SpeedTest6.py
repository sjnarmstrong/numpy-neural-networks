import timeit

setup1 = """\
import numpy as np
import pandas as pd
a=np.arange(1000)
"""

s1 = """\
c=pd.cut(a,[-float('inf'),500-1e-10,500+1e-10,float('inf')],labels=[-1,0,1])
b=c*a
"""
time1=timeit.timeit(setup=setup1,stmt=s1, number=10000)

setup2 = """\
import numpy as np
a=np.arange(1000)
"""

s2 = """\
c=np.where(a<500-1e-10, -1, np.where(a>500+1e-10,1,0))
b=c*a
"""
time2=timeit.timeit(setup=setup2,stmt=s2, number=10000)