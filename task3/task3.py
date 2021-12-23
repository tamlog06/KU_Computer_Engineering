import math
import numpy as np
from math import cos, sin, tanh
from visualize import *

global N, dx, dt, c, Ts, TD, a, mu
N = 100
dx = 1.0/N
dt = 0.001
c = 5.0
Ts = 0.1
TD = 0.1
a = 0.01
mu = c**2 * dt**2 / dx**2

# boundary=True: 固定端 False: 自由端
def main2(boundary):
    u = np.zeros((2, N), dtype=float)
    u[0][0] = (tanh((-dt-Ts)/a) + 1)/2 - (tanh((-dt-Ts-TD)/a) + 1)/2
    u[1][0] = (tanh((0-Ts)/a) + 1)/2 - (tanh((0-Ts-TD)/a) + 1)/2
    x_list = [u[1]]
    T = 5*(Ts+TD)
    k = 1
    tk = dt
    while tk < T:
        u_k1 = np.zeros(N, dtype=float)
        u_k1[0] = (tanh((tk-Ts)/a) + 1)/2 - (tanh((tk-Ts-TD)/a) + 1)/2
        u_k1[1:-1] = -u[k-1][1:-1] + mu*u[k][0:-2] + 2*(1-mu)*u[k][1:-1] + mu*u[k][2:]
        # 固定端
        if boundary:
            u_k1[N-1] = -u[k-1][N-1] + mu*u[k][N-2] + 2*(1-mu)*u[k][N-1]
        # 自由端
        else:
            u_k1[N-1] = -u[k-1][N-1] + mu*u[k][N-2] + (2-mu)*u[k][N-1]
        
        x_list.append(u_k1)
        u = np.vstack((u, u_k1))
        k += 1
        tk += dt

    x_axis = [i for i in range(N)]
    if boundary:
        title = f'3.3.2 x_movie (Fixed end)'
    else:
        title = f'3.3.2 x_movie (Free end)'
    anime(x_axis, x_list, title, fps=10, save_movie=True)

def main3(boundary):
    u = np.zeros((2, N), dtype=float)
    x_list = [u[1]]
    T = 5*(Ts+TD)
    k = 1
    tk = dt
    while tk < T:
        u_k1 = np.zeros(N, dtype=float)
        if Ts <= k*dt <= Ts + TD:
            u_k1[0] = 1.0
        u_k1[1:-1] = -u[k-1][1:-1] + mu*u[k][0:-2] + 2*(1-mu)*u[k][1:-1] + mu*u[k][2:]
        # 固定端
        if boundary:
            u_k1[N-1] = -u[k-1][N-1] + mu*u[k][N-2] + 2*(1-mu)*u[k][N-1]
        # 自由端
        else:
            u_k1[N-1] = -u[k-1][N-1] + mu*u[k][N-2] + (2-mu)*u[k][N-1]
        
        x_list.append(u_k1)
        u = np.vstack((u, u_k1))
        k += 1
        tk += dt

    x_axis = [i for i in range(N)]
    if boundary:
        title = f'3.3.3 x_movie (Fixed end)'
    else:
        title = f'3.3.3 x_movie (Free end)'
    anime(x_axis, x_list, title, fps=10, save_movie=True)

if __name__ == '__main__':
    main2(True)
    main3(True)