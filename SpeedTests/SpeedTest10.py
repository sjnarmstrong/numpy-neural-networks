import timeit

setup = """\
import numpy as np
import heapq

a=np.random.uniform(-10,10,10000)
c=np.random.uniform(-10,10,10000)

"""

s1 = """\
b=a.argsort()[:1000]
"""
time1=timeit.timeit(setup=setup,stmt=s1, number=10000)

s2 = """\
b=np.argpartition(a, 1000)[:1000]
"""
time2=timeit.timeit(setup=setup,stmt=s2, number=10000)

s3 = """\
b=sorted(a[np.argpartition(a, 1000)[:1000]])
"""
time3=timeit.timeit(setup=setup,stmt=s3, number=10000)

s4 = """\
inds=(a[np.argpartition(a, 1000)[:1000]]).argsort()
b=c[inds]
"""
time4=timeit.timeit(setup=setup,stmt=s3, number=10000)


s5 = """\
heapq.nsmallest(1000,np.arange(len(a)),key=a.__getitem__)
"""
time5=timeit.timeit(setup=setup,stmt=s4, number=1000)