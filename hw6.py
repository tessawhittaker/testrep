#libraries:
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import norm
import time
'''
def driver():

    x0= np.array([1,1,1])
    
    Nmax = 100
    tol = 1e-10
    
    t = time.time()
    for j in range(50):
      [xstar,ier,its] =  Newton(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Newton: the error message reads:',ier) 
    print('Newton: took this many seconds:',elapsed/50)
    print('Netwon: number of iterations is:',its)
     
    t = time.time()
    for j in range(20):
      [xstar,ier,its] =  LazyNewton(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Lazy Newton: the error message reads:',ier)
    print('Lazy Newton: took this many seconds:',elapsed/20)
    print('Lazy Newton: number of iterations is:',its)
     
    t = time.time()
    for j in range(20):
      [xstar,ier,its] = Broyden(x0, tol,Nmax)     
    elapsed = time.time()-t
    print(xstar)
    print('Broyden: the error message reads:',ier)
    print('Broyden: took this many seconds:',elapsed/20)
    print('Broyden: number of iterations is:',its)
     
def evalF(x): 

    F = np.zeros(3)
    F[0] = x[0]+np.cos(x[0]*x[1]*x[2])-1
    F[1] = (1-x[0])**(1/4)+x[1]+0.05*x[2]**2-0.15-1
    F[2] = -x[0]**2-0.1*x[1]**2+0.01*x[1]+x[2]-1
    return F
    
def evalJ(x): 
    J =np.array([[1-x[1]*x[2]*np.sin(x[0]*x[1]*x[2]),-x[0]*x[2]*np.sin(x[0]*x[1]*x[2]),-x[0]*x[1]*np.sin(x[0]*x[1]*x[2])],
          [(-1/4)*(1-x[0])**(-3/4),1,0.1*x[2]-0.15],
          [-2*x[0],-0.2*x[1]+0.01,1]])
    return J


def Newton(x0,tol,Nmax):

    for its in range(Nmax):
       J = evalJ(x0)
       Jinv = inv(J)
       F = evalF(x0)
       
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]
           
def LazyNewton(x0,tol,Nmax):

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
    
def Broyden(x0,tol,Nmax):
    A0 = evalJ(x0)

    v = evalF(x0)
    A = np.linalg.inv(A0)

    s = -A.dot(v)
    xk = x0+s
    for  its in range(Nmax):
       w = v
       v = evalF(xk)
       y = v-w;                   
       z = -A.dot(y)
       p = -np.dot(s,z)                 
       u = np.dot(s,A) 
       tmp = s+z
       tmp2 = np.outer(tmp,u)
       A = A+1./p*tmp2
       s = -A.dot(v)
       xk = xk+s
       if (norm(s)<tol):
          alpha = xk
          ier = 0
          return[alpha,ier,its]
    alpha = xk
    ier = 1
    return[alpha,ier,its]
     
        
#if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    #driver()       

#libraries:
'''

def driver():

    Nmax = 100
    x0= np.array([0,0,1])
    tol = 1e-6
    
    [xstar,gval,ier] = SteepestDescent(x0,tol,Nmax)
    print("the steepest descent code found the solution ",xstar)
    print("g evaluated at this point is ", gval)
    print("ier is ", ier	)

###########################################################
#functions:
def evalF(x): 

    F = np.zeros(3)
    F[0] = x[0]+np.cos(x[0]*x[1]*x[2])-1
    F[1] = (1-x[0])**(1/4)+x[1]+0.05*x[2]**2-0.15-1
    F[2] = -x[0]**2-0.1*x[1]**2+0.01*x[1]+x[2]-1
    return F
    
def evalJ(x): 
    J =np.array([[1-x[1]*x[2]*np.sin(x[0]*x[1]*x[2]),-x[0]*x[2]*np.sin(x[0]*x[1]*x[2]),-x[0]*x[1]*np.sin(x[0]*x[1]*x[2])],
          [(-1/4)*(1-x[0])**(-3/4),1,0.1*x[2]-0.15],
          [-2*x[0],-0.2*x[1]+0.01,1]])
    return J

def evalg(x):

    F = evalF(x)
    g = F[0]**2 + F[1]**2 + F[2]**2
    return g

def eval_gradg(x):
    F = evalF(x)
    J = evalJ(x)
    
    gradg = np.transpose(J).dot(F)
    return gradg


###############################
### steepest descent code

def SteepestDescent(x,tol,Nmax):
    
    for its in range(Nmax):
        g1 = evalg(x)
        z = eval_gradg(x)
        z0 = norm(z)

        if z0 == 0:
            print("zero gradient")
        z = z/z0
        alpha1 = 0
        alpha3 = 1
        dif_vec = x - alpha3*z
        g3 = evalg(dif_vec)

        while g3>=g1:
            alpha3 = alpha3/2
            dif_vec = x - alpha3*z
            g3 = evalg(dif_vec)
            
        if alpha3<tol:
            print("no likely improvement")
            ier = 0
            return [x,g1,ier]
        
        alpha2 = alpha3/2
        dif_vec = x - alpha2*z
        g2 = evalg(dif_vec)

        h1 = (g2 - g1)/alpha2
        h2 = (g3-g2)/(alpha3-alpha2)
        h3 = (h2-h1)/alpha3

        alpha0 = 0.5*(alpha2 - h1/h3)
        dif_vec = x - alpha0*z
        g0 = evalg(dif_vec)

        if g0<=g3:
            alpha = alpha0
            gval = g0

        else:
            alpha = alpha3
            gval =g3

        x = x - alpha*z

        if abs(gval - g1)<tol:
            ier = 0
            return [x,gval,ier]

    print('max iterations exceeded')    
    ier = 1        
    return [x,g1,ier]



if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()        
