import numpy as np
import matplotlib.pyplot as plt


a = 0 
b = 5

x = np.linspace(a, b, 500)
f_x = np.sin(x)
 
mac = lambda x: x - (x**3)/6 + (x**5)/120
p1 = lambda x: (x+(-7*x**3)/60)/(1+(x**2/20))
p2 = lambda x: x/(1+(x**3/6)+(7*x**4)/360)
p3 = lambda x: (x+(-7*x**3)/60)/(1+(x**2/20))

error_p1 = np.abs(f_x - p1(x))
error_p2 = np.abs(f_x - p2(x))
error_p3 = np.abs(f_x - p3(x))
error_mac= np.abs(f_x - mac(x))

# Plot the errors
plt.figure()
plt.plot(x, error_p1, label='Error in $P_{3}^{3}$')
plt.plot(x, error_p2, label='Error in $P_{2}^{4}$')
plt.plot(x, error_p3, label='Error in $P_{4}^{2}$')
plt.plot(x, error_mac, label='Error in Maclaurin polynomial', linestyle='dotted')
plt.xlabel('x')
plt.ylabel('Absolute Error')
plt.title('Accuracy of Pade Approximations with Sixth-Order Maclaurin of sin(x)')
plt.legend()
plt.show()