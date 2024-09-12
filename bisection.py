#!/usr/bin/env python
# coding: utf-8

# In[44]:


# import libraries
import numpy as np
def driver():
    # use routines
    #  f = lambda x: (x-1)**2*(x-3)
    #  a = 0
    # b = 2

    f = lambda x: np.sin(x)
    a = 0
    b = 0.1

    tol = 1e-5

    [astar,ier] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))


# In[45]:


# define routines
def bisection(f,a,b,tol):
    # Inputs:
        # f,a,b - function and endpoints of initial interval
        # tol - bisection stops when interval length < tol
    # Returns:
        # astar - approximation of root
        # ier - error message
        # - ier = 1 => Failed
        # - ier = 0 == success
    # first verify there is a root we can find in the interval
    fa = f(a)
    fb = f(b);
    
    if (fa*fb>0):
        ier = 1
        astar = a
        return [astar, ier]

# verify end points are not a root
    if (fa == 0):
        astar = a
        ier =0
        return [astar, ier]
    
    if (fb ==0):
        astar = b
        ier = 0
        return [astar, ier]
    
    count = 0
    d = 0.5*(a+b)
    
    
    while (abs(d-a)> tol):
        fd = f(d)
        if (fd ==0):
            astar = d
            ier = 0
            return [astar, ier]
        if (fa*fd<0):
            b = d
        else:
            a = d
            fa = fd
        d = 0.5*(a+b)
        count = count +1
        # print('abs(d-a) = ', abs(d-a))
    astar = d
    ier = 0
    return [astar, ier]
driver()


# 1) A was successful. For b, there was an error message because the root was too close to the end point. C was successful. It is not possible for bisection for find the root x=0. 

# 2a) The code was successful and the approximate root for A is 1. The desired accuracy was achieved. 
# 2b) The code was not successful. An error occured because the root was close to the end point. Error was extremely large.  
# 2c) 
