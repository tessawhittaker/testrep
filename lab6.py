import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 

def pre_lab(): 
    h = 0.01*2.0**(-np.arange(0,10))
    f = lambda x: np.cos(x)
    s = np.pi/2

    forward_difference = (f(s+h)-f(s))/h
    centered_difference = (f(s+h)-f(s-h))/(2*h)

    print(f'Foward Difference: {forward_difference}')
    print(f'Centered Difference: {centered_difference}')

def driver():

    x0 = np.array([1,0])
    
    Nmax = 100
    tol = 1e-10
     
    t = time.time()
    for j in range(20):
      [xstar,ier,its] =  LazyNewton(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Lazy Newton: the error message reads:',ier)
    print('Lazy Newton: took this many seconds:',elapsed/20)
    print('Lazy Newton: number of iterations is:',its)
     
def evalF(x): 

    F = np.zeros(2)
    
    F[0] = 4*x[0]**2+x[1]**2-4
    F[1] = x[0]+x[1]-np.sin(x[0]-x[1])
    return F
    
def evalJ(x): 
    
    J = np.array([[8*x[0], 2*x[1]], 
        [1+np.cos(x[0]-x[1]), 1+np.cos(x[0]-x[1])]])
    return J

def LazyNewton(x0,tol,Nmax):

    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''
    if norm(x0) < 2:
        J = evalJ(x0)
        Jinv = inv(J)

    for its in range(Nmax):

       F = evalF(x0)
       x1 = x0 - Jinv.dot(F)
    
           
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier,its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]   
driver()
    