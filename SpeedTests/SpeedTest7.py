import timeit

setup1 = """\
import numpy as np

"""

s1 = """\
a=np.random.uniform(-10,10,(10000,1))
b=np.random.uniform(-10,10,(10000,1))
b*=a
"""
time1=timeit.timeit(setup=setup1,stmt=s1, number=10000)

setup2 = """\
import numpy as np

"""

s2 = """\
a=np.random.uniform(-10,10,(10000,1))
b=np.random.uniform(-10,10,(10000,1))
np.multiply(a,b,out=b)
"""
time2=timeit.timeit(setup=setup2,stmt=s2, number=10000)