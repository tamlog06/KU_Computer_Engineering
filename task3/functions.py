import numpy as np
import math
from math import cos, sin

global k1, m1, w1
k1 = 3
m1 = 4
w1 = math.sqrt(k1/m1)

def correct_x1(t, w):
    return cos(w*t)

def correct_v1(t, w):
    return -w*sin(w*t)

def differential_v1(t, x, v):
    return -k1/m1*x

def differential_x1(t, x, v):
    return v

def euler(x0, v0, h):
    x = [x0]
    v = [v0]
    t = 0
    e = [abs(correct_x1(t, w1) - x[-1])]
    print( 2*math.pi/w1)
    while t < 2*math.pi/w1:
        t += h
        v_next = v[-1] - h*k1/m1*x[-1]
        x_next = x[-1] + h*v[-1]
        v.append(v_next)
        x.append(x_next)
        e.append(abs(correct_x1(t, w1) - x[-1]))
    return x, e

def heun(x0, v0, h):
    x = [x0]
    v = [v0]
    t = 0
    e = [abs(correct_x1(t, w) - x[-1])]
    while t < 2*math.pi/w1:
        t += h
        k1_v = differential_v(t, x[-1], v[-1])
        k1_x = differential_x(t, x[-1], v[-1])

        k2_v = differential_v(t+h, x[-1] + h*k1_x, v[-1] + h*k1_v)
        k2_x = differential_x(t+h, x[-1] + h*k1_x, v[-1] + h*k1_v)

        v_next = v[-1] + h/2*(k1_v + k2_v)
        x_next = x[-1] + h/2*(k1_x + k2_x)

        v.append(v_next)
        x.append(x_next)
        e.append(abs(correct_x1(t, w) - x[-1]))
    return x, e

def Runge_Kutta(x0, v0, h):
    x = [x0]
    v = [v0]
    t = 0
    e = [abs(correct_x1(t, w) - x[-1])]
    while t < 2*math.pi/w:
        t += h
        k1_v = differential_v(t, x[-1], v[-1])
        k1_x = differential_x(t, x[-1], v[-1])

        k2_v = differential_v(t+h/2, x[-1] + h/2*k1_x, v[-1] + h/2*k1_v)
        k2_x = differential_x(t+h/2, x[-1] + h/2*k1_x, v[-1] + h/2*k1_v)

        k3_v = differential_v(t+h/2, x[-1] + h/2*k2_x, v[-1] + h/2*k2_v)
        k3_x = differential_x(t+h/2, x[-1] + h/2*k2_x, v[-1] + h/2*k2_v)

        k4_v = differential_v(t+h, x[-1] + h*k3_x, v[-1] + h*k3_v)
        k4_x = differential_x(t+h, x[-1] + h*k3_x, v[-1] + h*k3_v)

        v_next = v[-1] + h/6*(k1_v + 2*k2_v + 2*k3_v + k4_v)
        x_next = x[-1] + h/6*(k1_x + 2*k2_x + 2*k3_x + k4_x)
        v.append(v_next)
        x.append(x_next)
        e.append(abs(correct_x1(t, w) - x[-1]))
    return x, e


####### task2 ########

global N2, k2, m2, Td2, h2
N2 = 100
k2 = 3
m2 = 4
Td2 = 50
h2 = 0.5

def D(t):
    if 0 <= t <= Td2:
        return 0.1
    else:
        return 0 

def differential_x2(t, x, v, i):
    return v[i]

# 境界条件：boundary=False -> 自由端、boundary=True: -> 固定端
def differential_v2(t, x, v, i, boundary=False):
    if i == 0:
        return k2/m2*(D(t)-2*x[0] + x[1])
    elif 1 <= i <= N2-2:
        return k2/m2*(x[i-1] - 2*x[i] + x[i+1])
    else:
        if boundary:
            return k2/m2*(x[N2-2] - 2*x[N2-1])
        else:
            return k2/m2*(x[N2-2] - x[N2-1])

def heun2(x, v, t, boundary):
    temp_x = x.copy()
    temp_v = v.copy()
    for i in range(N2):
        # k1_v = differential_v2(t, x, v, i, boundary)
        # k1_x = differential_x2(t, x, v, i)

        # k2_v = differential_v2(t+h2, x + h2*k1_x, v + h2*k1_v, i, boundary)
        # k2_x = differential_x2(t+h2, x + h2*k1_x, v + h2*k1_v, i)

        # v_next = v[i] + h2/2*(k1_v + k2_v)
        # x_next = x[i] + h2/2*(k1_x + k2_x)

        v_next = v[i] + h2*differential_v2(t, temp_x, temp_v, i, boundary)
        x_next = x[i] + h2*differential_x2(t, temp_x, temp_v, i)

        v[i] = v_next
        x[i] = x_next
    # print(x)
    # input()
    return x, v