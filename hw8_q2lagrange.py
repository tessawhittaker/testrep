import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def driver():
    f = lambda x: 1/(1+x**2)
    a = -5
    b = 5
    N_list  = [5,10,15,25]
    Neval = 1000
    xeval = np.linspace(a,b,Neval)
    yeval_l = []
    fex = f(xeval)
    err_l = [] 

    for n in range(4): 
        xint = np.array([5*np.cos(((2*j-1)*np.pi)/(2*N_list[n])) for j in range(1,N_list[n]+1)])
        yint = f(xint)
        
        yeval = np.zeros(Neval)
      
        for kk in range(Neval):
         yeval[kk] = eval_lagrange(xeval[kk],xint,yint,N_list[n])
            
        yeval_l.append(yeval)
        err_l.append(abs(yeval-fex))
        

    plt.figure()    
    plt.plot(xeval,fex,'ro-',label = 'Exact Function')
    plt.plot(xeval,yeval_l[0],'paleturquoise', marker = '*',label = 'N=5') 
    plt.plot(xeval,yeval_l[1],'lightseagreen',marker = '*',label = 'N=10') 
    plt.plot(xeval,yeval_l[2],'aquamarine',marker = '*',label = 'N=15') 
    plt.plot(xeval,yeval_l[3],'seagreen',marker = '*',label = 'N=20') 
    plt.title('Lagrange Interpolation with Chebyshev for varied N')
    plt.legend() 
    plt.show() 
    
    plt.figure()
    plt.semilogy(xeval,err_l[0],'paleturquoise',marker = '*',label='N=5')
    plt.semilogy(xeval,err_l[1],'lightseagreen',marker = '*',label='N=10')
    plt.semilogy(xeval,err_l[2],'aquamarine',marker = '*',label='N=15')
    plt.semilogy(xeval,err_l[3],'seagreen',marker = '*',label='N=20')
    plt.title('Error for Lagrange Interpolation with Chebyshev Nodes')
    plt.legend()
    plt.show()
    
def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    
    for count in range(N):
       for jj in range(N):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)
driver()
