import timeit

setup1 = """\
def func1(argsArrV):
    return argsArrV[0]+argsArrV[1]
"""

s1 = """\
a=(1,5)
func1(a)
"""
time1=timeit.timeit(setup=setup1,stmt=s1, number=10000000)

setup2 = """\
def func1(a,b,c,d):
    return a+b
"""

s2 = """\
func1(1,5,None,None)
"""
time2=timeit.timeit(setup=setup2,stmt=s2, number=10000000)

setup3 = """\
def func1(a,b):
    return a+b
"""

s3 = """\
b=(1,5)
func1(b[0],b[1])
"""
time3=timeit.timeit(setup=setup3,stmt=s3, number=10000000)

setup4 = """\
class Test:
    def __init__(self):
        self.a=10
        self.b=10
    def sum(self):
        return self.a+self.b
tester=Test()
"""

s4 = """\
a=(1,5)
tester.a=a[0]
tester.b=a[1]
tester.sum()
"""
time4=timeit.timeit(setup=setup4,stmt=s4, number=10000000)