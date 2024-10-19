import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 
import matplotlib.pyplot as plt

def poly_eval(c_n, x):
    y = 0
    power = len(c) - 1

    for c in c_n:
        y += c * x**power
        power -= 1
 
    return y

def eval_bary(xeval,xint,yint,N):
    if xeval in xint:
        return yint[xint.index(xeval)]
    I=1
    for n in range(N):
        I = I*(xeval-xint[n])

    mid_sum = 0

    for j in range(N):
        w = 1

        for i in range(N):
            if i!=j:
                w=w/(xint[j]-xint[i])

        mid_sum += w*yint[j]/(xeval - xint[j])

    return I * mid_sum

def driver():

    N = 10
    h = 2/(N-1)

    f = lambda x: 1/(1+(10*x)**2)   
    #xvals = [-1 +(j-1)*h for j in range(1,N+1)]
    xvals = [(np.cos(2*j-1)*np.pi)/(2*N) for j in range(1,N+1)]
    yvals = [f(x) for x in xvals]

    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(-1,1,Neval+1)
    yeval_l = np.zeros(Neval+1)
    
    for kk in range(Neval+1):
        yeval_l[kk] = eval_bary(xeval[kk],xvals,yvals,N)

    ''' create vector with exact values'''
    fex = f(xeval)
    plt.figure()
    plt.plot(xeval,fex,'ro-', label = 'True Function')
    plt.plot(xeval,yeval_l,'bs--', label = 'Barycentric Approx.')
    plt.title('True Function versus using Barycentric Approximation')
    plt.show()
    #plt.plot(xvals,f(np.array(xvals)), 'o',label = 'Interpolation Nodes')

    #err = abs(yeval_l-fex)
    #plt.figure()
    #plt.plot(xeval,err,'ro-')
    #plt.title('Error using Approximation Barycentric')
    #plt.show()
driver()