print('HI')

x = [1,2,3]
#print(3*x)

import numpy as np

y = np.array([1,2,3])

#print(3*y)

#print('this is 3y',3*y)

import matplotlib.pyplot as plt

X = np.linspace(0,2*np.pi,100)
Ya = np.sin(X)
Yb = np.cos(X)

#plt.plot(X,Ya)
#plt.plot(X,Yb)
#plt.xlabel('x')
#plt.ylabel('y')
#plt.show()

x = np.linspace(1,10,10)
y = np.arange(1,11)

print(x[:3])

print('The first three entries of x are',x[:3])

w = 10**(-np.linspace(1,10,10))
print(w)
