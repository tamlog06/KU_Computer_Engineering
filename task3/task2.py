import math
from math import cos, sin
from visualize import *
import numpy as np

global N, k, m, w, Td
N = 100
k = 3
m = 4
w = math.sqrt(k/m)
Td = 50
# hを小さくしないとeulerやheunはうまくできない
# heun: 0.1
# euler: 0.01
# h = 0.5

def D(t):
    if 0 <= t <= Td:
        return 0.1
    else:
        return 0 

def diff_x(t, x, v):
    return v

# 境界条件：boundary=False -> 自由端、boundary=True: -> 固定端
def diff_v(t, x, v, boundary):
    differential_v = np.zeros(N, dtype=float)
    for i in range(N):
        if i == 0:
            differential_v[i] = w**2*(D(t)-2*x[0] + x[1])
        elif 1 <= i <= N-2:
            differential_v[i] = w**2*(x[i-1] - 2*x[i] + x[i+1])
        else:
            if boundary:
                differential_v[i] = w**2*(x[N-2] - 2*x[N-1])
            else:
                differential_v[i] = w**2*(x[N-2] - x[N-1])
    return differential_v

# オイラー法
def euler(x, v, t, boundary):
    result_x = x + h*diff_x(t, x, v)
    result_v = v + h*diff_v(t, x, v, boundary)

    return result_x, result_v

# ホイン法
def heun(x, v, t, boundary):
    k1_v = diff_v(t, x, v, boundary)
    k1_x = diff_x(t, x, v)

    k2_v = diff_v(t+h, x + h*k1_x, v + h*k1_v, boundary)
    k2_x = diff_x(t+h, x + h*k1_x, v + h*k1_v)

    next_v = v + h/2*(k1_v + k2_v)
    next_x = x + h/2*(k1_x + k2_x)

    return next_x, next_v

# ルンゲ・クッタ法
def Runge_Kutta(x, v, t, boundary):
    k1_v = diff_v(t, x, v, boundary)
    k1_x = diff_x(t, x, v)

    k2_v = diff_v(t+h/2, x + h/2*k1_x, v + h/2*k1_v, boundary)
    k2_x = diff_x(t+h/2, x + h/2*k1_x, v + h/2*k1_v)

    k3_v = diff_v(t+h/2, x + h/2*k2_x, v + h/2*k2_v, boundary)
    k3_x = diff_x(t+h/2, x + h/2*k2_x, v + h/2*k2_v)

    k4_v = diff_v(t+h, x + h*k3_x, v + h*k3_v, boundary)
    k4_x = diff_x(t+h, x + h*k3_x, v + h*k3_v)

    next_v = v + h/6*(k1_v + 2*k2_v + 2*k3_v + k4_v)
    next_x = x + h/6*(k1_x + 2*k2_x + 2*k3_x + k4_x)
    
    return next_x, next_v

# ルンゲ・クッタ・ジル法
def Gill(x, v, t, boundary):
    k1_v = diff_v(t, x, v, boundary)
    k1_x = diff_x(t, x, v)

    k2_v = diff_v(t+h/2, x + h/2*k1_x, v + h/2*k1_v, boundary)
    k2_x = diff_x(t+h/2, x + h/2*k1_x, v + h/2*k1_v)

    k3_v = diff_v(t+h/2, x + h*((1/math.sqrt(2)-1/2)*k1_x + (1-1/math.sqrt(2))*k2_x), v + h*((1/math.sqrt(2)-1/2)*k1_v + (1-1/math.sqrt(2))*k2_v), boundary)
    k3_x = diff_x(t+h/2, x + h*((1/math.sqrt(2)-1/2)*k1_x + (1-1/math.sqrt(2))*k2_x), v + h*((1/math.sqrt(2)-1/2)*k1_v + (1-1/math.sqrt(2))*k2_v))

    k4_v = diff_v(t+h, x + h*(-1/math.sqrt(2)*k2_x + (1+1/math.sqrt(2))*k3_x), v + h*(-1/math.sqrt(2)*k2_v + (1+1/math.sqrt(2))*k3_v), boundary)
    k4_x = diff_x(t+h, x + h*(-1/math.sqrt(2)*k2_x + (1+1/math.sqrt(2))*k3_x), v + h*(-1/math.sqrt(2)*k2_v + (1+1/math.sqrt(2))*k3_v))

    next_v = v + h/6*(k1_v + (2-math.sqrt(2))*k2_v + (2-math.sqrt(2))*k3_v + k4_v)
    next_x = x + h/6*(k1_x + (2-math.sqrt(2))*k2_x + (2-math.sqrt(2))*k3_x + k4_x)
    
    return next_x, next_v

def main(boundary, method, save_movie=False):
    global h
    if method == 'Euler':
        fps = 10
        h = 0.005
    elif method == 'Heun':
        fps = 10
        h = 0.1
    else:
        fps = 10
        h = 0.5
    t = 0
    x = np.zeros(N, dtype=float)
    v = np.zeros(N, dtype=float)
    x_list = [x.copy()]
    x_N = [x[-1]]
    T = N*math.sqrt(m/k)
    count = 0
    while t <= T*4:
        t += h
        if method == 'Euler':
            x, v = euler(x, v, t, boundary)
            if count % 100 == 0:
                x_list.append(x.copy())
                x_N.append(x[-1])
        elif method == 'Heun':
            x, v = heun(x, v, t, boundary)
            if count % 5 == 0:
                x_list.append(x.copy())
                x_N.append(x[-1])
        elif method == 'Runge Kutta':
            x, v = Runge_Kutta(x, v, t, boundary)
            x_list.append(x.copy())
            x_N.append(x[-1])
        elif method == 'Runge Kutta Gill':
            x, v = Gill(x, v, t, boundary)
            x_list.append(x.copy())
            x_N.append(x[-1])
        else:
            print('Error')
            exit()
        count += 1
    
    if boundary:
        title1 = f'3.2 {method} x_(N-1) (Fixed end)'
        title2 = f'3.2 {method} x_movie (Fixed end)'
    else:
        title1 = f'3.2 {method} x_(N-1) (Free end)'
        title2 = f'3.2 {method} x_movie (Free end)'
    x_axis = np.array([i for i in range(len(x_list))])
    visualize(x_axis, x_N, 'time', '$x_{{N-1}}$', title1, x_ticks=False)

    x_axis = [i for i in range(N)]
    anime(x_axis, x_list, title2, fps, save_movie)

if __name__ == '__main__':
    method = {0: 'Euler', 1: 'Heun', 2: 'Runge Kutta', 3: 'Runge Kutta Gill'}
    main(True, method[3], save_movie=True)