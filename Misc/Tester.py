import numpy as np

errors=np.array([[1,2,3,4,5,6,7,8,9,0],[0,1,5,1,1,2,9,9,10,11]])
a=errors[0]
b=errors[1]

weights=np.repeat(99,100).reshape((2,10,5))
weights[0]=0
c=weights[0]
d=weights[1]

holdRange=np.arange(10)

f=weights[1].copy()

i=np.where(a<b)[0]
f[i]=weights[0,i]


i=np.argmin(errors,axis=0)
h=weights[i,holdRange]