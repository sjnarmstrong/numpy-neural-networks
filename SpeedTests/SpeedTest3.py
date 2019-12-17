import timeit

setup1 = """\
class Test:
    def __init__(self):
        self.a=10
        self.b=10
    def sum(self):
        return self.a+self.b
tester=Test()
"""

s1 = """\
tester.sum()
"""
time1=timeit.timeit(setup=setup1,stmt=s1, number=10000000)

setup2 = """\
class Test:
    def __init__(self):
        self.a=10
        self.b=10
        self.dosum=self.sum
    def sum(self):
        return self.a+self.b
tester=Test()
"""

s2 = """\
tester.dosum()
"""
time2=timeit.timeit(setup=setup2,stmt=s2, number=10000000)