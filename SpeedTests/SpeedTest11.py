import timeit

setup = """\
import numpy as np

a=np.repeat(np.arange(1000),3000).reshape((1000,3000))

weightMidway=int((len(a[0])+1)/2.0)
rollCount=5



newshape=np.array(a.shape)
newshape[0]*=rollCount
b=np.empty(newshape, dtype=a.dtype)

b[:,:weightMidway]=a[:,:weightMidway].repeat(rollCount,axis=0)
aend=a[:,weightMidway:]


enumeratedRollCount=np.arange(rollCount)

def func(i, b, aend):
    b[i::rollCount,weightMidway:]=np.roll(aend, 1 if i == 0 else -i)
vfunc=np.vectorize(func, excluded=[1,2])



"""

s1 = """\
b[::rollCount,weightMidway:]=np.roll(aend, 1)
b[1::rollCount,weightMidway:]=np.roll(aend, -1)
b[2::rollCount,weightMidway:]=np.roll(aend, -2)
b[3::rollCount,weightMidway:]=np.roll(aend, -3)
b[4::rollCount,weightMidway:]=np.roll(aend, -4)
"""
time1=timeit.timeit(setup=setup,stmt=s1, number=1)

s2 = """\
vfunc(np.arange(rollCount),b,aend)
"""
time2=timeit.timeit(setup=setup,stmt=s2, number=1)