import timeit

setup1 = """\
class Test:
    def __init__(self):
        self.a=10
    def sum(self,b):
        return self.a+b
tester=Test()
"""

s1 = """\
tester.sum(5)
"""
time1=timeit.timeit(setup=setup1,stmt=s1, number=10000000)

setup2 = """\
class Test:
    def __init__(self):
        self.a=10
        self.b=5
    def sum(self):
        return self.a+self.b
tester=Test()
"""

s2 = """\
tester.sum()
"""
time2=timeit.timeit(setup=setup2,stmt=s2, number=10000000)