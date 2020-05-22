import numpy as np
import math
import matplotlib.pyplot as plt

g = 9.81 #重力加速度(m/s^2)
a = 5 #球の半径(m)
l = 10 #支点から球の中心までの距離(m)

omega_s2 = g/l * 1/(1 + (2*a*a)/(5*l*l)) #角振動数の2乗

print(omega_s2)

# 剛体振り子
def f(theta_omega): #関数の定義
    #d(theta)/dt = omega
    #d(omega)/dt = -omega_s2*theta

    return np.array([-omega_s2 * math.sin(theta_omega[1]), theta_omega[0]])


omega_trj = np.array([2]) #角速度配列
theta_trj = np.array([0]) #角度配列
array_t = np.array([0.0]) #時間発展配列

theta_omega = np.array([0, 0]) #各時刻での角速度・角度
theta_omega[0] = omega_trj[0] #初期角速度
theta_omega[1] = theta_trj[0] #初期角度
t = array_t[0] #初期時間
h = 0.01 #ルンゲクッタ法の刻み幅

#ルンゲクッタ法
for i in range(1,3000):
    k1 = h*f(theta_omega)
    k2 = h*f(theta_omega+0.5*k1)
    k3 = h*f(theta_omega+0.5*k2)
    k4 = h*f(theta_omega+k3)
    theta_omega = theta_omega + (k1+2*k2+2*k3+k4)/6
    t = t + h
    omega_trj = np.append(omega_trj,theta_omega[0])
    theta_trj = np.append(theta_trj,theta_omega[1])
    array_t = np.append(array_t,t)

#グラフ描写
plt.plot(array_t,omega_trj, color=(0,0,1), marker=".")
plt.show()
plt.plot(array_t,theta_trj, color=(0,1,0), marker=".")
plt.show()
