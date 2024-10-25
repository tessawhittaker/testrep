import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import norm


def driver():
    
    f = lambda x: 1./(1.+x**2)
    a = -5
    b = 5
    
    
    ''' number of intervals'''
    Nint = [5,10,15,20]
    Neval = 1000
    err = [] 
    xeval  = np.linspace(a,b,Neval+1)
    fex = f(xeval)
    yeval = [] 

    for n in range(4): 
        xint = np.array([5*np.cos(((2*j+1)*np.pi)/(2*(Nint[n]+1))) for j in range(Nint[n]+1)])
        yint = f(xint)

        
        (M,C,D) = create_natural_spline(yint,xint,Nint[n])
        
    #    print('M =', M)
    #    print('C =', C)
    #    print('D=', D)
        
        yeval.append(eval_cubic_spline(xeval,Neval,xint,Nint[n],M,C,D))
        
        #print('yeval = ', yeval)
        
        ''' evaluate f at the evaluation points'''
        fex = f(xeval)
        err.append(abs(yeval[n]-fex))
        
    nerr = norm(fex-yeval)
    #print('nerr = ', nerr)
    
    plt.figure()    
    plt.plot(xeval,fex,'k',label='exact function')
    plt.plot(xeval,yeval[0],c='mediumpurple',label='N = 5')
    plt.plot(xeval,yeval[1],c='darkmagenta',label='N = 10')
    plt.plot(xeval,yeval[2],c='skyblue',label='N = 15') 
    plt.plot(xeval,yeval[3],c='navy',label='N = 20')
    plt.legend()
    plt.title('Cubic Splines Interpolation for varied N with Chebychev Nodes')
    plt.show()
     
    plt.figure() 
    plt.semilogy(xeval,err[0],c='mediumpurple',label='N = 5')
    plt.semilogy(xeval,err[1],c='darkmagenta',label='N = 10')
    plt.semilogy(xeval,err[2],c='skyblue',label='N = 15')
    plt.semilogy(xeval,err[3],c='navy',label='N = 20')
    plt.title('Error for Cubic Splines Interpolation with Chebychev Nodes')
    plt.legend()
    plt.show()
    
def create_natural_spline(yint,xint,N):

#    create the right  hand side for the linear system
    b = np.zeros(N+1)
#  vector values
    h = np.zeros(N+1)  
    for i in range(1,N):
       hi = xint[i]-xint[i-1]
       hip = xint[i+1] - xint[i]
       b[i] = (yint[i+1]-yint[i])/hip - (yint[i]-yint[i-1])/hi
       h[i-1] = hi
       h[i] = hip
#  create matrix so you can solve for the M values
# This is made by filling one row at a time 
    A = np.zeros((N+1,N+1))
    A[0][0] = 1.0
    for j in range(1,N):
       A[j][j-1] = h[j-1]/6
       A[j][j] = (h[j]+h[j-1])/3 
       A[j][j+1] = h[j]/6
    A[N][N] = 1

    Ainv = inv(A)
    
    M  = Ainv.dot(b)

#  Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
       C[j] = yint[j]/h[j]-h[j]*M[j]/6
       D[j] = yint[j+1]/h[j]-h[j]*M[j+1]/6
    return(M,C,D)
       
def eval_local_spline(xeval,xi,xip,Mi,Mip,C,D):
# Evaluates the local spline as defined in class
# xip = x_{i+1}; xi = x_i
# Mip = M_{i+1}; Mi = M_i

    hi = xip-xi
    yeval = (Mi*(xip-xeval)**3 +(xeval-xi)**3*Mip)/(6*hi) \
            + C*(xip-xeval) + D*(xeval-xi)
    return yeval 
    
    
def  eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D):
    
    yeval = np.zeros(Neval+1)
    
    for j in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        atmp = xint[j]
        btmp= xint[j+1]
    
#   find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval >= btmp))
        xloc = xeval[ind]
        
# evaluate the spline
        yloc = eval_local_spline(xloc,atmp,btmp,M[j],M[j+1],C[j],D[j])
#        print('yloc = ', yloc)
#   copy into yeval
        yeval[ind] = yloc
    return(yeval)
           
driver()               

