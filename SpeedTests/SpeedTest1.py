import timeit

setup1 = """\
import numpy as np

a=np.random.uniform(-10,10,(100,1))
b=np.random.uniform(-10,10,(200,101))
"""

s1 = """\
np.dot(b,np.append(a,[[1]],axis=0))
"""
time1=timeit.timeit(setup=setup1,stmt=s1, number=100000)

setup2 = """\
import numpy as np

a=np.random.uniform(-10,10,(100,1))
b=np.random.uniform(-10,10,(200,100))
c=np.random.uniform(-10,10,(200,1))
"""

s2 = """\
np.dot(b,a)+c
"""
time2=timeit.timeit(setup=setup2,stmt=s2, number=100000)