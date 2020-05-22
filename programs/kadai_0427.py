import numpy as np
import math
import matplotlib.pyplot as plt

# equation
# \dot{x_1} = 0.5*{ -h(x_1) + x_2 }
# \dot{x_2} = 0.2*{ -x_1 -1.5x_2 + 1.2 }
# h(x_1) = 17.76x_1 -103.79x_1^2 +229.62x_1^3 -226.31x_1^4 +83.72x_1^5

ips = 3

def tunnel(x1_x2): #d equation : RHS

    return np.array([x1_x2[1] , -x1_x2[0] + ips * (1 - (x1_x2[0])**2) * x1_x2[1]])

x1_trj = np.array([2]) #trajectory of x1
x2_trj = np.array([0]) #trajectory of x2
array_t = np.array([0.0]) #time array

x1_x2 = np.array([0.0, 0.0]) #x1,x2 : each time
x1_x2[0] = x1_trj[0] #x2 : initial
x1_x2[1] = x2_trj[0] #x1 : initial
t = array_t[0] #initial time
h = 0.01 #notch width

#Runge-Kutta
for i in range(1,3000):
    k1 = h*tunnel(x1_x2)
    k2 = h*tunnel(x1_x2+0.5*k1)
    k3 = h*tunnel(x1_x2+0.5*k2)
    k4 = h*tunnel(x1_x2+k3)

    x1_x2 = x1_x2 + (k1+2*k2+2*k3+k4)/6
    t = t + h
    x1_trj = np.append(x1_trj,x1_x2[0])
    x2_trj = np.append(x2_trj,x1_x2[1])
    array_t = np.append(array_t,t)

#graph plot
plt.plot(x1_trj,x2_trj, color=(0,1,0), marker=".")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
