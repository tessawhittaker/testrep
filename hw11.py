import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import norm
import scipy


def driver():
    
    f = lambda x: np.cos(1/x)*x
    a = 1
    b = 1/(10^7)
    
    # exact integral
    I_ex = scipy.integrate.quad(f,a,b)[0]
    
#    N =100
#    ntest = np.arrange(0,N,step=2)
    
#    errorT = np.zeros(len(ntest))
#    errorS = np.zeros(len(ntest))
    
#    for j in range(0,len(ntest)):
#        n = ntest[j]

# for simpson's n must be even.        
# n+1 = number of pts.
    n=10
    I_trap = CompTrap(a,b,n,f)
    print('I_trap= ', I_trap)
    
    err = abs(I_ex-I_trap)  
    print('absolute error = ', err)  
    
    quad_trap = scipy.integrate.quad(f, a, b, epsabs=10**-4)[0]
    quad_err = abs(I_trap-quad_trap)
    print('Quadrature Error:', quad_err)
    
    #simpsons
    I_simp = CompSimp(a,b,n,f)
    print('I_simp= ', I_simp)
    
    err = abs(I_ex-I_simp)   
    print('absolute error = ', err)    

    quad_trap = scipy.integrate.quad(f, a, b, epsabs=10**-4)[0]
    quad_err2 = abs(I_simp-quad_trap)
    #print('Quadrature Error:', quad_err2)

        
def CompTrap(a,b,n,f):
    h = (b-a)/n
    xnode = a+np.arange(0,n+1)*h
    
    I_trap = h*f(xnode[0])*1/2
    
    for j in range(1,n):
         I_trap = I_trap+h*f(xnode[j])
    I_trap= I_trap + 1/2*h*f(xnode[n])
    
    return I_trap     

def CompSimp(a,b,n,f):
    h = (b-a)/n
    xnode = a+np.arange(0,n+1)*h
    I_simp = f(xnode[0])

    nhalf = n/2
    for j in range(1,int(nhalf)+1):
         # even part 
         I_simp = I_simp+2*f(xnode[2*j])
         # odd part
         I_simp = I_simp +4*f(xnode[2*j-1])
    I_simp= I_simp + f(xnode[n])
    
    I_simp = h/3*I_simp
    
    return I_simp     

    
driver()    
