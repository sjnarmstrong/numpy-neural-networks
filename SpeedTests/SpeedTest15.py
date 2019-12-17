import timeit


#from sklearn.utils import shuffle
setup = """\
import numpy as np

a=np.random.uniform(0,1,(100000,1000))
b=np.random.uniform(0,1,(100000,300))
randomize = np.arange(len(a))

"""

#s1 = """\
#c,d=shuffle(a,b)
#"""
#time1=timeit.timeit(setup=setup,stmt=s1, number=100)

#s2 = """\
#np.random.shuffle(randomize)
#a = a[randomize]
#b = b[randomize]
#"""
#time2=timeit.timeit(setup=setup,stmt=s2, number=100)

s1 = """\
c=a[...,:1000]
"""
time1=timeit.timeit(setup=setup,stmt=s1, number=100)

s2 = """\
c=a[:1000]
"""
time2=timeit.timeit(setup=setup,stmt=s2, number=100)