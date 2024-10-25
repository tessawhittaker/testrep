import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import math

def driver():


    f = lambda x: 1./(1.+x**2)
    fp = lambda x: -2*x/(1.+x**2)**2

    N = [5,10,15,20]
    a = -5
    b = 5
    errH = [] 
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yevalH = []
    fex = f(xeval)

    for n in range(4):
        xint = np.linspace(a,b,N[n]+1)
        yint = np.zeros(N[n]+1)
        ypint = np.zeros(N[n]+1)

        for jj in range(N[n]+1):
            yint[jj] = f(xint[jj])
            ypint[jj] = fp(xint[jj])
        
        yeval = np.zeros(Neval+1)

        for kk in range(Neval+1):
            yeval[kk] = eval_hermite(xeval[kk],xint,yint,ypint,N[n])
        yevalH.append(yeval)

        ''' create vector with exact values'''
        fex = np.zeros(Neval+1)
        for kk in range(Neval+1):
            fex[kk] = f(xeval[kk])
        
        errH.append(abs(yeval-fex))
        
    
    plt.figure()
    plt.plot(xeval,fex,'ro-', label = 'Exact Function')
    plt.plot(xeval,yevalH[0],'c.--',label='N = 5')
    plt.plot(xeval,yevalH[1],'r.--',label='N = 10')
    plt.plot(xeval,yevalH[2],'g.--',label='N = 15') 
    plt.plot(xeval,yevalH[3],'k.--',label='N = 20')
    plt.legend() 
    plt.semilogy()
    plt.title('Hermite Interpolation')
    plt.show()
         
    plt.figure()
    plt.semilogy(xeval,errH[0],'c',label='N = 5')
    plt.semilogy(xeval,errH[1],'r',label='N = 10')
    plt.semilogy(xeval,errH[2],'g',label='N = 15')
    plt.semilogy(xeval,errH[3],'k',label='N = 20')
    plt.title('Error for Hermite Interpolation')
    plt.show()            


def eval_hermite(xeval,xint,yint,ypint,N):

    ''' Evaluate all Lagrange polynomials'''

    lj = np.ones(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    ''' Construct the l_j'(x_j)'''
    lpj = np.zeros(N+1)
#    lpj2 = np.ones(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
#              lpj2[count] = lpj2[count]*(xint[count] - xint[jj])
              lpj[count] = lpj[count]+ 1./(xint[count] - xint[jj])
              

    yeval = 0.
    
    for jj in range(N+1):
       Qj = (1.-2.*(xeval-xint[jj])*lpj[jj])*lj[jj]**2
       Rj = (xeval-xint[jj])*lj[jj]**2
#       if (jj == 0):
#         print(Qj)
         
#         print(Rj)
#         print(Qj)
#         print(xeval)
 #        return
       yeval = yeval + yint[jj]*Qj+ypint[jj]*Rj
       
    return(yeval)
       
driver() 