import timeit

setup1 = """\
import numpy as np

a=np.random.uniform(-10,10,(101,200))
b=np.random.uniform(-10,10,(200,101))

class Test:
    def __init__(self):
        pass
        
    def MatM(self,a,b):
        return np.dot(a,b)
        
c=Test()
    
"""

s1 = """\
c.MatM(a,b)
"""
time1=timeit.timeit(setup=setup1,stmt=s1, number=100000)

setup2 = """\
import numpy as np

a=np.random.uniform(-10,10,(101,200))
b=np.random.uniform(-10,10,(200,101))

class Test:
    def __init__(self,a,b):
        self.a=a
        self.b=b
        
    def MatM(self):
        return np.dot(self.a,self.b)
        
c=Test(a,b)
"""

s2 = """\
c.MatM()
"""
time2=timeit.timeit(setup=setup2,stmt=s2, number=100000)