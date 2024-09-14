#!/usr/bin/env python
# coding: utf-8

# In[65]:


import matplotlib.pylab as plt
import numpy as np
import math
import random


# ## Problem 2b - Condition number of A

# In[89]:


A = np.matrix([[.5,.5],[.5*(1+10**-10),.5*(1-10**-10)]])
A_inv = np.matrix([[1-10**10,10**10],[1+10**10,-10**10]])

print(np.linalg.norm(A,ord=2),np.linalg.norm(A_inv, ord=2))


# ## Problem 2c

# In[108]:


delta_b = np.matrix([[10**-5,10**-5]])
rel_error = np.matmul(A_inv, np.transpose(delta_b))
rel_error


# ## Problem 3b

# In[128]:


x = 9.999999995000000*10**-10
y = np.e**x
y-1


# ## Problem 3c

# In[135]:


y2 = x+0.5*x**2+(1/6)*x**3
y2-1


# ## Problem 4a

# In[49]:


t = np.linspace(0,math.pi,31)
y = np.cos(t)
s = 0
for k in range(0,31):
    s+= (t[k]*(y[k]))
print(f'the sum is: {s}')


# ## Problem 4b

# In[61]:


R = 1.2
deltar = 0.1
f = 15
p = 0

theta = np.linspace(0,2*math.pi,400)
xvals = R*(1+deltar*np.sin(f*theta+p))*np.cos(theta)
yvals = R*(1+deltar*np.sin(f*theta+p))*np.sin(theta)
plt.plot(xvals,yvals)
plt.axis('equal')
plt.show()


# In[73]:


deltar = 0.05
p = random.uniform(0,2)
for i in range(10):
    R = i
    f = 2+i
    theta = np.linspace(0,2*math.pi,400)
    xvals = R*(1+deltar*np.sin(f*theta+p))*np.cos(theta)
    yvals = R*(1+deltar*np.sin(f*theta+p))*np.sin(theta)
    plt.plot(xvals,yvals)
plt.axis('equal')
plt.show()


# In[ ]:




