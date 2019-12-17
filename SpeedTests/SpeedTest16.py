import timeit


#from sklearn.utils import shuffle
setup = """\
import numpy as np

a=np.random.uniform(0,1,100000)
lena=10.0/len(a)
"""

s1 = """\
np.average(a)*10
"""
time1=timeit.timeit(setup=setup,stmt=s1, number=100)
print(time1)
s2 = """\
np.sum(a)*lena
"""
time2=timeit.timeit(setup=setup,stmt=s2, number=100)
print(time2)
