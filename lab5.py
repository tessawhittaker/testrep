# import libraries
import numpy as np

def driver():

# use routines    
    f = lambda x: np.e**(x**2+(7*x)-30)-1
    fp = lambda x: ((2*x)+7)*np.e**(x**2+(7*x)-30)
    fpp = lambda x: (4*x**2+(28*x)+51)*np.e**(x**2+(7*x)-30)
    a = 2
    b = 4.5

#    f = lambda 
# x: np.sin(x)
#    a = 0.1
#    b = np.pi+0.1

    tol = 1e-7

    [astar,iter] = bisection(f,fp,fpp,a,b,tol)
    print('the approximate root is',astar)
    print('number of iterations is ',iter)
    #print('the error message reads:',ier)
    print('f(astar) =', f(astar))


def newton(f,fp,p0,tol,Nmax):
    p = np.zeros(Nmax+1)
    p[0] = p0
    for it in range(Nmax):
        p1 = p0-f(p0)/fp(p0)
        p[it+1] = p1
        if (abs(p1-p0) < tol):
            pstar = p1
            info = 0
            return [pstar,it]
        p0 = p1
    pstar = p1
    info = 1
    return [pstar,it]

# define routines
def bisection(f,fp,fpp,a,b,tol):
    count = 0
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when interval length < tol

#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 

    fa = f(a)
    fb = f(b);

    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, count]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, count]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, count]

    d = 0.5*(a+b)


    while (abs(d-a)> tol):
        if ((f(d)*fpp(d))/(fp(d)**2) < 1):
            return newton(f,fp,d,tol,100)
      
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
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    return [astar, count]
      
driver()               

