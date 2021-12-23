import math
from math import cos, sin
from visualize import *

global k
global m
global w

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

def main1():
    h = (2*math.pi)/(64*w)

    x, e = euler(1, 0, h)
    x_axis = [i for i in range(65)]
    xc_t = x[:-1]
    error = e[:-1]
    visualize(x_axis, xc_t, 'time', '$x_c(t)$', '3.1.1 Euler x_c(t)')
    visualize(x_axis, error, 'time', 'error', '3.1.1 Euler error')

def main2():
    h_list = [2*math.pi/w * 2**(-p) for p in range(2, 21)]
    error_list = []

    for h in h_list:
        x, e = euler(1, 0, h)
        e_max = max(e)
        error_list.append(e_max)
    
    x_axis = [p for p in range(2, 21)]
    y_axis = [math.log2(error) for error in error_list]
    title = '3.1.2 maximum error - p (Euler)'
    visualize(x_axis, y_axis, 'p', 'log2 Ex', title, label='log2 Ex' ,least_square_flag=True, lrange=[3, None])

def main3():
    h = (2*math.pi)/(64*w)

    x, e = heun(1, 0, h)
    x_axis = [i for i in range(65)]
    xc_t = x[:-1]
    error = e[:-1]
    visualize(x_axis, xc_t, 'time', '$x_c(t)$', '3.1.3 Heun x_c(t)')
    visualize(x_axis, error, 'time', 'error', '3.1.3 Heun error')

    h_list = [2*math.pi/w * 2**(-p) for p in range(2, 21)]
    error_list = []

    for h in h_list:
        x, e = heun(1, 0, h)
        e_max = max(e)
        error_list.append(e_max)
    
    x_axis = [p for p in range(2, 21)]
    logEx = [math.log2(error) for error in error_list]
    title = '3.1.3 maximum error - p (Heun)'
    visualize(x_axis, logEx, 'p', 'log2 Ex', title, label='log2 Ex', least_square_flag=True, lrange=[1, -3])

def main4():
    h = (2*math.pi)/(64*w)
    x, e = Runge_Kutta(1, 0, h)
    x_axis = [i for i in range(65)]
    xc_t = x[:-1]
    error = e[:-1]
    visualize(x_axis, xc_t, 'time', '$x_c(t)$', '3.1.4 Runge Kutta x_c(t)')
    visualize(x_axis, error, 'time', 'error', '3.1.4 Runge Kutta error')

    h_list = [2*math.pi/w * 2**(-p) for p in range(2, 21)]
    error_list = []

    for h in h_list:
        x, e = Runge_Kutta(1, 0, h)
        e_max = max(e)
        error_list.append(e_max)
    
    x_axis = [p for p in range(2, 21)]
    logEx = [math.log2(error) for error in error_list]
    title = '3.1.4 maximum error - p (Runge Kutta)'
    visualize(x_axis, logEx, 'p', 'log2 Ex', title, label='log2 Ex', least_square_flag=True, lrange=[0, 10])

# TODO
def extra():
    h = (2*math.pi)/(64*w)
    x, e = Gill(1, 0, h)
    x_axis = [i for i in range(65)]
    xc_t = x[:-1]
    error = e[:-1]
    visualize(x_axis, xc_t, 'time', '$x_c(t)$', '3.1.extra Runge Kutta Gill x_c(t)')
    visualize(x_axis, error, 'time', 'error', '3.1.extra Runge Kutta Gill error')

    h_list = [2*math.pi/w * 2**(-p) for p in range(2, 21)]
    error_list = []

    for h in h_list:
        x, e = Gill(1, 0, h)
        x, e = Gill(1, 0, h)
        e_max = max(e)
        error_list.append(e_max)
    
    x_axis = [p for p in range(2, 21)]
    logEx = [math.log2(error) for error in error_list]
    title = '3.1.extra maximum error - p (Runge Kutta Gill)'
    visualize(x_axis, logEx, 'p', 'log2 Ex', title, label='log2 Ex', least_square_flag=False, lrange=[0, 10])


if __name__ == '__main__':
    main1()
    main2()
    main3()
    main4()
    extra()