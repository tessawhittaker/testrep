import numpy as np

def driver():

# test functions 
     f1 = lambda x: ((10/(x+4))**0.5)
     p = 1.3652300134140976
     Nmax = 100
     tol = 1e-10

# test f1 '''
     x0 = 1.5
     v1 = fixedpt(f1,x0,tol,Nmax)
     print(v1)
     compute_order(v1,p)

# define routines
def fixedpt(f,x0,tol,Nmax):
    x = np.zeros((Nmax, 1))
    
    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    while (count <Nmax):
       x[count] = x0
       count = count +1
       x1 = f(x0)
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          x[count] = xstar
          return x[:count+1]
          #return [xstar,ier]
       x0 = x1

       
    xstar = x1
    ier = 1
    
    #return [xstar,ier]
    return x




# Question 2.2.1: 

def compute_order(x,xstar):
    diff1 = np.abs(x[1::]-xstar)
    diff2 = np.abs(x[0:-1]-xstar)

    fit = np.polyfit(np.log(diff2.flatten()),np.log(diff1.flatten()),1)
    print('the order equation is')
    print('log|p_{n+1}-p|)=log(lambda)+alpha*log(|p_n-p|) where')
    print('lambda = '+str(np.exp(fit[1])))
    print('alpha = '+str(fit[0]))
    return [fit,diff1,diff2] 

#def aitkens(approximations, tol, Nmax):
#p_n+1 - p (p_n+1-p) = p_n+2-p (p_n -p)
#p_n+1^2 - p*p_n+1 - p*p_n+1 + p^2 = p_n*p_n+2 -p*p_n+2-p*p_n + p^2




driver()
