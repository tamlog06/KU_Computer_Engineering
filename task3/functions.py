import numpy as np
import math
from math import cos, sin

global k, m, w
k = 3
m = 4
w = math.sqrt(k/m)

def correct_x(t, w):
    return cos(w*t)

def correct_v(t, w):
    return -w*sin(w*t)

def differential_v(t, x, v):
    return -k/m*x

def differential_x(t, x, v):
    return v

def euler(x0, v0, h):
    x = [x0]
    v = [v0]
    t = 0
    e = [abs(correct_x(t, w) - x[-1])]
    while t < 2*math.pi/w:
        t += h
        v_next = v[-1] - h*k/m*x[-1]
        x_next = x[-1] + h*v[-1]
        v.append(v_next)
        x.append(x_next)
        e.append(abs(correct_x(t, w) - x[-1]))
    return x, e

def heun(x0, v0, h):
    x = [x0]
    v = [v0]
    t = 0
    e = [abs(correct_x(t, w) - x[-1])]
    while t < 2*math.pi/w:
        t += h
        k1_v = differential_v(t, x[-1], v[-1])
        k1_x = differential_x(t, x[-1], v[-1])

        k2_v = differential_v(t+h, x[-1] + h*k1_x, v[-1] + h*k1_v)
        k2_x = differential_x(t+h, x[-1] + h*k1_x, v[-1] + h*k1_v)

        v_next = v[-1] + h/2*(k1_v + k2_v)
        x_next = x[-1] + h/2*(k1_x + k2_x)

        v.append(v_next)
        x.append(x_next)
        e.append(abs(correct_x(t, w) - x[-1]))
    return x, e

def Runge_Kutta(x0, v0, h):
    x = [x0]
    v = [v0]
    t = 0
    e = [abs(correct_x(t, w) - x[-1])]
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
        e.append(abs(correct_x(t, w) - x[-1]))
    return x, e

def Gill(x0, v0, h):
    x = [x0]
    v = [v0]
    t = 0
    e = [abs(correct_x(t, w) - x[-1])]
    while t < 2*math.pi/w:
        t += h
        k1_v = differential_v(t, x[-1], v[-1])
        k1_x = differential_x(t, x[-1], v[-1])

        k2_v = differential_v(t+h/2, x[-1] + h/2*k1_x, v[-1] + h/2*k1_v)
        k2_x = differential_x(t+h/2, x[-1] + h/2*k1_x, v[-1]+ h/2*k1_v)

        k3_v = differential_v(t+h/2, x[-1] + h*((1/math.sqrt(2)-1/2)*k1_x + (1-1/math.sqrt(2))*k2_x), v[-1] + h*((1/math.sqrt(2)-1/2)*k1_v + (1-1/math.sqrt(2))*k2_v))
        k3_x = differential_x(t+h/2, x[-1] + h*((1/math.sqrt(2)-1/2)*k1_x + (1-1/math.sqrt(2))*k2_x), v[-1] + h*((1/math.sqrt(2)-1/2)*k1_v + (1-1/math.sqrt(2))*k2_v))

        k4_v = differential_v(t+h, x[-1] + h*(-1/math.sqrt(2)*k2_x+(1+1/math.sqrt(2))*k3_x), v[-1] + h*(-1/math.sqrt(2)*k2_v+(1+1/math.sqrt(2))*k3_v))
        k4_x = differential_x(t+h, x[-1] + h*(-1/math.sqrt(2)*k2_x+(1+1/math.sqrt(2))*k3_x), v[-1] + h*(-1/math.sqrt(2)*k2_v+(1+1/math.sqrt(2))*k3_v))

        v_next = v[-1] + h*(1/6*k1_v + 1/3*(1-1/math.sqrt(2))*k2_v + 1/3*(1-1/math.sqrt(2))*k3_v + 1/6*k4_v)
        x_next = x[-1] + h*(1/6*k1_x + 1/3*(1-1/math.sqrt(2))*k2_x + 1/3*(1-1/math.sqrt(2))*k3_x + 1/6*k4_x)
        v.append(v_next)
        x.append(x_next)
        e.append(abs(correct_x(t, w) - x[-1]))
    
    return x, e