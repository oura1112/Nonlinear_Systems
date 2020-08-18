# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 18:20:11 2020

@author: Ryohei Oura
"""

import numpy as np
import math
import matplotlib.pyplot as plt

#Pendulum or mass spring without friction
class pendulum():
    
    def __init__(self, a, epsilon, init_x1, init_x2, T):
        self.a = a
        self.epsilon = epsilon
        self.init_x1 = init_x1
        self.init_x2 = init_x2
        self.x1 = init_x1
        self.x2 = init_x2
        self.T = T
        
        self.k = 3
        self.m = 20
        
    def pendulum(self,x1,x2):
        x1_dot = x2
        x2_dot = -self.a * math.sin(x1)
        
        return x1_dot, x2_dot
    
    def mass_spring(self,x1,x2):
        x1_dot = x2
        x2_dot = -x1
        
        return x1_dot, x2_dot
        
    def explicit_euler(self):
        
        H_x1 = []
        H_x2 = []
        
        time = []
        H = []
        
        H.append(self.x2**2/2 - self.a * math.cos(self.x1) + self.a)
        time.append(0)
        
        plt.scatter([self.x1], [self.x2], c="red", s=0.2)
        
        for t in range(self.T):
            x1_dot, x2_dot = self.pendulum(self.x1, self.x2)
            
            self.x1 = self.x1 + self.epsilon * x1_dot
            self.x2 = self.x2 + self.epsilon * x2_dot
        
            H_x1.append(self.x1)
            H_x2.append(self.x2)
            
            time.append(t+1)
            H.append(self.x2**2/2 - self.a * math.cos(self.x1) + self.a)
        
        plt.scatter([self.init_x1], [self.init_x2], c="red", s=20)
        plt.plot(H_x1, H_x2)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()
        
        plt.plot(time, H)
        plt.xlabel("time")
        plt.ylabel("Hamilton")
        plt.show()
        
        self.x1 = self.init_x1
        self.x2 = self.init_x2
        
    def runge_kutta_4th(self):
        H_x1 = []
        H_x2 = []
        
        time = []
        H = []
        
        H.append(self.x2**2/2 - self.a * math.cos(self.x1) + self.a)
        time.append(0)
        
        for t in range(self.T):
            x1_dot, x2_dot = self.pendulum(self.x1, self.x2)
            
            x1_x2_dot = np.array([x1_dot, x2_dot])
            
            k1 = self.epsilon * x1_x2_dot
            k2 = self.epsilon * np.array(self.pendulum(self.x1 + 0.5*k1[0], self.x2 + 0.5*k1[1]))
            k3 = self.epsilon * np.array(self.pendulum(self.x1 + 0.5*k2[0], self.x2 + 0.5*k2[1]))
            k4 = self.epsilon * np.array(self.pendulum(self.x1 + k3[0], self.x2 + k3[1]))
            
            x1_x2_dot = (k1+2*k2+2*k3+k4)/6
            self.x1 = self.x1 + x1_x2_dot[0]
            self.x2 = self.x2 + x1_x2_dot[1]
            
            H_x1.append(self.x1)
            H_x2.append(self.x2)
            
            time.append(t+1)
            H.append(self.x2**2/2 - self.a * math.cos(self.x1) + self.a)
        
        plt.scatter([self.init_x1], [self.init_x2], c="red", s=20)
        plt.plot(H_x1, H_x2)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()
        
        plt.plot(time, H)
        plt.xlabel("time")
        plt.ylabel("Hamiltonian")
        plt.show()
        
        self.x1 = self.init_x1
        self.x2 = self.init_x2
        
    def symplectic_euler(self):
        
        H_x1 = []
        H_x2 = []
        
        time = []
        H = []
        
        H.append(self.x2**2/2 - self.a * math.cos(self.x1) + self.a)
        time.append(0)
        
        for t in range(self.T):
            x1_dot, x2_dot = self.pendulum(self.x1, self.x2)           
            self.x1 = self.x1 + self.epsilon * x1_dot

            x1_dot, x2_dot = self.pendulum(self.x1, self.x2)
            self.x2 = self.x2 + self.epsilon * x2_dot
        
            H_x1.append(self.x1)
            H_x2.append(self.x2)
            
            time.append(t+1)
            H.append((self.x2**2)/2 - self.a * math.cos(self.x1) + self.a)
        
        plt.scatter([self.init_x1], [self.init_x2], c="red", s=20)
        plt.scatter([H_x1[0]], [H_x2[0]], c="red", s=20)
        plt.plot(H_x1, H_x2)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()
        
        plt.plot(time, H)
        plt.xlabel("time")
        plt.ylabel("Hamiltonian")
        plt.show()
        
        self.x1 = self.init_x1
        self.x2 = self.init_x2
        
    def discrete_gradient(self): #Gonzalez
        
        #harmonic oscillator
        
        H_x1 = []
        H_x2 = []
        
        time = []
        H = []
        
        H.append(self.x1**2/ (2*self.m) + self.k*self.x2**2/2)
        time.append(0)
        
        A = np.array([ [1, -self.epsilon*self.k/(2*self.m)], [self.epsilon*(1/2), 1] ])
        B = np.array([ [1, self.epsilon*self.k/(2*self.m)], [-self.epsilon*(1/2), 1] ])
        P = np.linalg.inv(A)@B
        print(P)
        x = np.array([self.x1, self.x2])
        
        for t in range(self.T):
            x = P @ x
            
            H_x1.append(x[0])
            H_x2.append(x[1])
            
            time.append(t+1)
            H.append((x[0]**2)/2 + (x[1]**2)/2)
            
        plt.scatter([self.init_x1], [self.init_x2], c="red", s=20)
        plt.plot(H_x1, H_x2)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()
        
        plt.plot(time, H)
        plt.xlabel("time")
        plt.ylabel("Hamiltonian")
        plt.show()
        
        self.x1 = self.init_x1
        self.x2 = self.init_x2
        
 
#Pendulum with friction       
class pendulum_wf():
    
    def __init__(self, a, b, epsilon, init_x1, init_x2, T):
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.init_x1 = init_x1
        self.init_x2 = init_x2
        self.x1 = init_x1
        self.x2 = init_x2
        self.T = T
        
    def pendulum(self,x1,x2):
        x1_dot = x2
        x2_dot = -self.a * math.sin(x1) - self.b * x2
        
        return x1_dot, x2_dot
        
    def explicit_euler(self):
        
        H_x1 = []
        H_x2 = []
        
        time = []
        H = []
        
        H.append(self.x2**2/2 - self.a * math.cos(self.x1) + self.a)
        time.append(0)
        
        plt.scatter([self.x1], [self.x2], c="red", s=0.2)
        
        for t in range(self.T):
            x1_dot, x2_dot = self.pendulum(self.x1, self.x2)
            
            self.x1 = self.x1 + self.epsilon * x1_dot
            self.x2 = self.x2 + self.epsilon * x2_dot
        
            H_x1.append(self.x1)
            H_x2.append(self.x2)
            
            time.append(t+1)
            H.append(self.x2**2/2 - self.a * math.cos(self.x1) + self.a)
        
        plt.scatter([self.init_x1], [self.init_x2], c="red", s=20)
        plt.plot(H_x1, H_x2)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()
        
        plt.plot(time, H)
        plt.xlabel("time")
        plt.ylabel("Hamiltonian")
        plt.show()
        
        self.x1 = self.init_x1
        self.x2 = self.init_x2
        
    def runge_kutta_4th(self):
        H_x1 = []
        H_x2 = []
        
        time = []
        H = []
        
        H.append(self.x2**2/2 - self.a * math.cos(self.x1) + self.a)
        time.append(0)
        
        for t in range(self.T):
            x1_dot, x2_dot = self.pendulum(self.x1, self.x2)
            
            x1_x2_dot = np.array([x1_dot, x2_dot])
            
            k1 = self.epsilon * x1_x2_dot
            k2 = self.epsilon * np.array(self.pendulum(self.x1 + 0.5*k1[0], self.x2 + 0.5*k1[1]))
            k3 = self.epsilon * np.array(self.pendulum(self.x1 + 0.5*k2[0], self.x2 + 0.5*k2[1]))
            k4 = self.epsilon * np.array(self.pendulum(self.x1 + k3[0], self.x2 + k3[1]))
            
            x1_x2_dot = (k1+2*k2+2*k3+k4)/6
            self.x1 = self.x1 + x1_x2_dot[0]
            self.x2 = self.x2 + x1_x2_dot[1]
            
            H_x1.append(self.x1)
            H_x2.append(self.x2)
            
            time.append(t+1)
            H.append(self.x2**2/2 - self.a * math.cos(self.x1) + self.a)
        
        plt.scatter([self.init_x1], [self.init_x2], c="red", s=20)
        plt.plot(H_x1, H_x2)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()
        
        plt.plot(time, H)
        plt.xlabel("time")
        plt.ylabel("Hamiltonian")
        plt.show()
        
        self.x1 = self.init_x1
        self.x2 = self.init_x2
        
    def symplectic_euler(self):
        
        H_x1 = []
        H_x2 = []
        
        time = []
        H = []
        
        H.append(self.x2**2/2 - self.a * math.cos(self.x1) + self.a)
        time.append(0)
        
        for t in range(self.T):
            x1_dot, x2_dot = self.pendulum(self.x1, self.x2)           
            self.x1 = self.x1 + self.epsilon * x1_dot

            x1_dot, x2_dot = self.pendulum(self.x1, self.x2)
            self.x2 = self.x2 + self.epsilon * x2_dot
        
            H_x1.append(self.x1)
            H_x2.append(self.x2)
            
            time.append(t+1)
            H.append((self.x2**2)/2 - self.a * math.cos(self.x1) + self.a)
        
        plt.scatter([self.init_x1], [self.init_x2], c="red", s=20)
        plt.plot(H_x1, H_x2)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()
        
        plt.plot(time, H)
        plt.xlabel("time")
        plt.ylabel("Hamiltonian")
        plt.show()
        
        self.x1 = self.init_x1
        self.x2 = self.init_x2
        
"""
a : param of potential
b : param of friction
epsilon : time step width
init_x2 : initial velue of momentum (mass = 1)
init_x1 : initial value of position (angle)
"""
pendulum = pendulum(a=1, epsilon=0.01, init_x1=1, init_x2=1, T=5000)
pendulum_wf = pendulum_wf(a=1, b=0.01, epsilon=0.01, init_x1=1, init_x2=1, T=5000)

pendulum.explicit_euler()
pendulum.runge_kutta_4th() 
pendulum.symplectic_euler() 
pendulum.discrete_gradient() 

pendulum_wf.explicit_euler()
pendulum_wf.runge_kutta_4th()
pendulum_wf.symplectic_euler() 