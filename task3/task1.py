import math
from math import cos, sin
from visualize import *
from functions import *

global k
global m
global w

k = 3
m = 4
w = math.sqrt(k/m)

def task1():
    h = (2*math.pi)/(64*w)

    x, e = euler(1, 0, h)
    x_axis = [i for i in range(65)]
    xc_t = x[:-1]
    error = e[:-1]
    visualize(x_axis, xc_t, 'time', 'x_c(t)', 'Euler Method')
    visualize(x_axis, error, 'time', 'error', 'Euler error')

def task2():
    h_list = [2*math.pi/w * 2**(-p) for p in range(2, 21)]
    error_list = []

    for h in h_list:
        x, e = euler(1, 0, h)
        e_max = max(e)
        error_list.append(e_max)
    
    x_axis = [p for p in range(2, 21)]
    y_axis = [math.log2(error) for error in error_list]
    title = 'The change in maximum error corresponding to the change h (Euler)'
    visualize(x_axis, y_axis, 'p', 'log2 Ex', title, label='log2 Ex' ,least_square_flag=True, lrange=[3, None])

def task3():
    h = (2*math.pi)/(64*w)

    x, e = heun(1, 0, h)
    x_axis = [i for i in range(65)]
    xc_t = x[:-1]
    error = e[:-1]
    visualize(x_axis, xc_t, 'time', 'x_c(t)', 'Heun method')
    visualize(x_axis, error, 'time', 'error', 'Heun error')

    h_list = [2*math.pi/w * 2**(-p) for p in range(2, 21)]
    error_list = []

    for h in h_list:
        x, e = heun(1, 0, h)
        e_max = max(e)
        error_list.append(e_max)
    
    x_axis = [p for p in range(2, 21)]
    logEx = [math.log2(error) for error in error_list]
    title = 'The change in maximum error corresponding to the change h (Heun)'
    visualize(x_axis, logEx, 'p', 'log2 Ex', title, label='log2 Ex', least_square_flag=True, lrange=[1, -3])

def task4():
    h = (2*math.pi)/(64*w)
    x, e = Runge_Kutta(1, 0, h)
    x_axis = [i for i in range(65)]
    xc_t = x[:-1]
    error = e[:-1]
    visualize(x_axis, xc_t, 'time', 'x_c(t)', 'Runge Kutta method')
    visualize(x_axis, error, 'time', 'error', 'Runge Kutta error')

    h_list = [2*math.pi/w * 2**(-p) for p in range(2, 21)]
    error_list = []

    for h in h_list:
        x, e = Runge_Kutta(1, 0, h)
        e_max = max(e)
        error_list.append(e_max)
    
    x_axis = [p for p in range(2, 21)]
    logEx = [math.log2(error) for error in error_list]
    title = 'The change in maximum error corresponding to the change h (Runge Kutta)'
    visualize(x_axis, logEx, 'p', 'log2 Ex', title, label='log2 Ex', least_square_flag=True, lrange=[0, 10])

def extra():
    h = (2*math.pi)/(64*w)
    x, e = Gill(1, 0, h)
    x_axis = [i for i in range(65)]
    xc_t = x[:-1]
    error = e[:-1]
    visualize(x_axis, xc_t, 'time', 'x_c(t)', 'Runge Kutta Gill method')
    visualize(x_axis, error, 'time', 'error', 'Runge Kutta Gill error')

    h_list = [2*math.pi/w * 2**(-p) for p in range(2, 21)]
    error_list = []

    for h in h_list:
        x, e = Gill(1, 0, h)
        e_max = max(e)
        error_list.append(e_max)
    
    x_axis = [p for p in range(2, 21)]
    logEx = [math.log2(error) for error in error_list]
    title = 'The change in maximum error corresponding to the change h (Runge Kutta Gill)'
    visualize(x_axis, logEx, 'p', 'log2 Ex', title, label='log2 Ex', least_square_flag=False, lrange=[0, 10])


if __name__ == '__main__':
    # task1()
    # task2()
    # task3()
    # task4()
    extra()