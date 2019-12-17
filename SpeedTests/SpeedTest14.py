import timeit

setup = """\
import numpy as np

a=np.random.uniform(0,1,(100,1000))
b=np.random.uniform(0,1,(1000,300))


"""

s1 = """\
c=np.dot(a,b)
"""
time1=timeit.timeit(setup=setup,stmt=s1, number=100)

s2 = """\
c=np.einsum('ij, jk->ik', a, b)
"""
time2=timeit.timeit(setup=setup,stmt=s2, number=100)
