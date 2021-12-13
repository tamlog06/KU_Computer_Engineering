import math
from math import cos, sin
from visualize import *
import numpy as np

global N, k, m, w, Td, h
N = 100
k = 3
m = 4
w = math.sqrt(k/m)
Td = 50
# hを小さくしないとeulerやheunはうまくできない
# heun: 0.1
# euler: 0.01
h = 0.5

def D(t):
    if 0 <= t <= Td:
        return 0.1
    else:
        return 0 

# def differential_x(t, x, v, i):
#     return v[i]

def diff_x(t, x, v):
    return v

# 境界条件：boundary=False -> 自由端、boundary=True: -> 固定端
# def differential_v(t, x, v, i, boundary=False):
#     if i == 0:
#         return w**2*(D(t)-2*x[0] + x[1])
#     elif 1 <= i <= N-2:
#         return w**2*(x[i-1] - 2*x[i] + x[i+1])
#     else:
#         if boundary:
#             return w**2*(x[N-2] - 2*x[N-1])
#         else:
#             return w**2*(x[N-2] - x[N-1])

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

def euler(x, v, t, boundary):
    result_x = x + h*diff_x(t, x, v)
    result_v = v + h*diff_v(t, x, v, boundary)

    return result_x, result_v

def heun(x, v, t, boundary):
    k1_v = diff_v(t, x, v, boundary)
    k1_x = diff_x(t, x, v)

    k2_v = diff_v(t+h, x + h*k1_x, v + h*k1_v, boundary)
    k2_x = diff_x(t+h, x + h*k1_x, v + h*k1_v)

    next_v = v + h/2*(k1_v + k2_v)
    next_x = x + h/2*(k1_x + k2_x)

    return next_x, next_v

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

def Gill(x, v, t, boundary):
    k1_v = diff_v(t, x, v, boundary)
    k1_x = diff_x(t, x, v)

    k2_v = diff_v(t+h/2, x + h/2*k1_x, v + h/2*k1_v, boundary)
    k2_x = diff_x(t+h/2, x + h/2*k1_x, v + h/2*k1_v)

    k3_v = diff_v(t+h/2, x + h*((1/math.sqrt(2)-1/2)*k1_x + (1-1/math.sqrt(2))*k2_x), v + h*((1/math.sqrt(2)-1/2)*k1_v + (1-1/math.sqrt(2))*k2_v), boundary)
    k3_x = diff_x(t+h/2, x + h*((1/math.sqrt(2)-1/2)*k1_x + (1-1/math.sqrt(2))*k2_x), v + h*((1/math.sqrt(2)-1/2)*k1_v + (1-1/math.sqrt(2))*k2_v))

    k4_v = diff_v(t+h, x + h*(-1/math.sqrt(2)*k2_x+(1+1/math.sqrt(2))*k3_x), v + h*(-1/math.sqrt(2)*k2_v+(1+1/math.sqrt(2))*k3_v), boundary)
    k4_x = diff_x(t+h, x + h*(-1/math.sqrt(2)*k2_x+(1+1/math.sqrt(2))*k3_x), v + h*(-1/math.sqrt(2)*k2_v+(1+1/math.sqrt(2))*k3_v))

    next_v = v + h/6*(k1_v + (2-math.sqrt(2))*k2_v + (2-math.sqrt(2))*k3_v + k4_v)
    next_x = x + h/6*(k1_x + (2-math.sqrt(2))*k2_x + (2-math.sqrt(2))*k3_x + k4_x)
    
    return next_x, next_v

def main(boundary=True):
    t = 0
    x = np.zeros(N, dtype=float)
    v = np.zeros(N, dtype=float)
    x_list = [x.copy()]
    x_N = [x[-1]]
    T = N*math.sqrt(m/k)
    while t <= T*2:
        t += h
        # x, v = euler(x, v, t, boundary)
        # x, v = heun(x, v, t, boundary)
        # x, v = Runge_Kutta(x, v, t, boundary)
        x, v = Gill(x, v, t, boundary)
        x_list.append(x.copy())
        x_N.append(x[-1])
    x_axis = np.array([i for i in range(len(x_list))])
    visualize(x_axis, x_N, 'time', 'x_N-1', 'x_N-1', x_ticks=False)

    x_axis = [i for i in range(N)]
    anime(x_axis, x_list, 'Gill', True)


if __name__ == '__main__':
    main(True)