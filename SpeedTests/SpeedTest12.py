import timeit

setup = """\
import numpy as np

errors=np.random.uniform(0,1,(2,1000000))
a=errors[0]
b=errors[1]

weights=np.repeat(99,20000000).reshape((2,1000000,-1))
weights[0]=0
c=weights[0]
d=weights[1]

holdRange=np.arange(1000000)

f=weights[1].copy()





"""

s1 = """\
i=np.where(a<b)[0]
f[i]=weights[0,i]
"""
time1=timeit.timeit(setup=setup,stmt=s1, number=100)

s2 = """\
i=np.argmin(errors,axis=0)
h=weights[i,holdRange]
"""
time2=timeit.timeit(setup=setup,stmt=s2, number=100)
