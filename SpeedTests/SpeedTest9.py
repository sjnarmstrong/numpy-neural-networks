import timeit

setup = """\
import numpy as np
import itertools as iterT

a=np.random.uniform(-10,10,(200,200))
c=np.random.uniform(-10,10,(100,200))

"""

s1 = """\
e=np.tile(c.T[:100],2).T
f=np.repeat(c[:,100:],2,axis=0)
a[:]=np.append(e,f,axis=1)
"""
time1=timeit.timeit(setup=setup,stmt=s1, number=10000)

s2 = """\
e=np.tile(c.T,2).T
f=np.repeat(c,2,axis=0)
a[:]=np.append(e[:,:100],f[:,100:],axis=1)
"""
time2=timeit.timeit(setup=setup,stmt=s2, number=10000)

s3 = """\
e=np.tile(c.T,2).T
f=np.repeat(c,2,axis=0)
a[:,:100]=e[:,:100]
a[:,100:]=f[:,100:]
"""
time3=timeit.timeit(setup=setup,stmt=s3, number=10000)


s4 = """\
for i,j in zip(a,iterT.product(c,repeat=2)):      
    i[:100]=j[0][:100]
    i[100:]=j[1][100:]
"""
time4=timeit.timeit(setup=setup,stmt=s4, number=10000)