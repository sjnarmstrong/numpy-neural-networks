import timeit

setup = """\
import numpy as np

a=np.random.uniform(0,1,(10000,100000))
b=np.random.uniform(0,1,(10000,1))
c=np.random.uniform(0,1,(10000,1))


"""

s1 = """\
b=c
"""
time1=timeit.timeit(setup=setup,stmt=s1, number=100)

s2 = """\
a[:,0]=c.flat
"""
time2=timeit.timeit(setup=setup,stmt=s2, number=100)
