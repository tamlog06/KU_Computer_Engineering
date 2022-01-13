import math
from math import cos, sin
from visualize import *
import time
from tqdm import tqdm

global k
global m
global w

k = 3
m = 4
w = math.sqrt(k/m)

# 変位の解析解
def correct_x(t, w):
    return cos(w*t)

# 速度の解析解
def correct_v(t, w):
    return -w*sin(w*t)

# 変位の微分
def differential_x(t, x, v):
    return v

# 速度の微分
def differential_v(t, x, v):
    return -k/m*x

# オイラー法
def euler(x0, v0, h):
    x = [x0]
    v = [v0]
    t = 0
    # 解析解との絶対誤差
    e = [abs(correct_x(t, w) - x[-1])]
    while t < 2*math.pi/w:
        t += h
        # 次状態のvとxと解析解との絶対誤差を計算
        v_next = v[-1] - h*k/m*x[-1]
        x_next = x[-1] + h*v[-1]
        v.append(v_next)
        x.append(x_next)
        e.append(abs(correct_x(t, w) - x[-1]))
    return x, e

# ホイン法
def heun(x0, v0, h):
    x = [x0]
    v = [v0]
    t = 0
    #  解析解との絶対誤差
    e = [abs(correct_x(t, w) - x[-1])]
    while t < 2*math.pi/w:
        t += h
        k1_v = differential_v(t, x[-1], v[-1])
        k1_x = differential_x(t, x[-1], v[-1])

        k2_v = differential_v(t+h, x[-1] + h*k1_x, v[-1] + h*k1_v)
        k2_x = differential_x(t+h, x[-1] + h*k1_x, v[-1] + h*k1_v)

        # 次状態のvとxと解析解との絶対誤差を計算
        v_next = v[-1] + h/2*(k1_v + k2_v)
        x_next = x[-1] + h/2*(k1_x + k2_x)

        v.append(v_next)
        x.append(x_next)
        e.append(abs(correct_x(t, w) - x[-1]))
    return x, e

# ルンゲクッタ法
def Runge_Kutta(x0, v0, h):
    x = [x0]
    v = [v0]
    t = 0
    #  解析解との絶対誤差
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

        # 次状態のvとxと解析解との絶対誤差を計算
        v_next = v[-1] + h/6*(k1_v + 2*k2_v + 2*k3_v + k4_v)
        x_next = x[-1] + h/6*(k1_x + 2*k2_x + 2*k3_x + k4_x)
        v.append(v_next)
        x.append(x_next)
        e.append(abs(correct_x(t, w) - x[-1]))
    return x, e

# ルンゲクッタジル法
def Gill(x0, v0, h):
    x = [x0]
    v = [v0]
    t = 0
    #  解析解との絶対誤差
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

        # 次状態のvとxと解析解との絶対誤差を計算
        v_next = v[-1] + h*(1/6*k1_v + 1/3*(1-1/math.sqrt(2))*k2_v + 1/3*(1+1/math.sqrt(2))*k3_v + 1/6*k4_v)
        x_next = x[-1] + h*(1/6*k1_x + 1/3*(1-1/math.sqrt(2))*k2_x + 1/3*(1+1/math.sqrt(2))*k3_x + 1/6*k4_x)

        v.append(v_next)
        x.append(x_next)
        e.append(abs(correct_x(t, w) - x[-1]))
    return x, e

# 解析解を描画
def main0():
    h = (2*math.pi)/(64*w)

    x = [1]
    t = 0
    while t < 2*math.pi/w:
        t += h
        x.append(correct_x(t, w))

    x_axis = [i for i in range(len(x))]
    visualize(x_axis, x, 'time', '$x_a(t)$', '3.1.0 Analytical x_a(t)', '3.1', x_ticks=[i for i in range(0, len(x), 5)])

# オイラー法で解く
def main1():
    h = (2*math.pi)/(64*w)

    x, e = euler(1, 0, h)
    x_axis = [i for i in range(len(x))]
    # 変位の時間変化を描画
    visualize(x_axis, x, 'time', '$x_c(t)$', '3.1.1 Euler x_c(t)', '3.1', x_ticks=[i for i in range(0, len(x), 5)])
    # 解析解との絶対誤差を描画
    visualize(x_axis, e, 'time', 'error', '3.1.1 Euler error', '3.1', x_ticks=[i for i in range(0, len(x), 5)])

# オイラー法でpを変化させて解く
def main2():
    h_list = [2*math.pi/w * 2**(-p) for p in range(2, 21)]
    error_list = []

    for h in h_list:
        x, e = euler(1, 0, h)
        e_max = max(e)
        error_list.append(e_max)

    # pに対する解析解との最大誤差を描画
    x_axis = [p for p in range(2, 21)]
    y_axis = [math.log2(error) for error in error_list]
    title = '3.1.2 maximum error - p (Euler)'
    visualize(x_axis, y_axis, 'p', 'log2 Ex', title, '3.1', x_ticks=[i for i in range(2, 21)],label='log2 Ex', least_square_flag=True, lrange=[3, None])

# ホイン法で解く
def main3():
    h = (2*math.pi)/(64*w)

    # xと解析解との絶対誤差を描画
    x, e = heun(1, 0, h)
    x_axis = [i for i in range(len(x))]
    visualize(x_axis, x, 'time', '$x_c(t)$', '3.1.3 Heun x_c(t)', '3.1', x_ticks=[i for i in range(0, len(x), 5)])
    visualize(x_axis, e, 'time', 'error', '3.1.3 Heun error', '3.1', x_ticks=[i for i in range(0, len(x), 5)])

    h_list = [2*math.pi/w * 2**(-p) for p in range(2, 21)]
    error_list = []

    for h in h_list:
        x, e = heun(1, 0, h)
        e_max = max(e)
        error_list.append(e_max)
    
    # pに対する解析解との最大誤差を描画
    x_axis = [p for p in range(2, 21)]
    logEx = [math.log2(error) for error in error_list]
    title = '3.1.3 maximum error - p (Heun)'
    visualize(x_axis, logEx, 'p', 'log2 Ex', title, '3.1', x_ticks=[i for i in range(2, 21)], label='log2 Ex', least_square_flag=True, lrange=[1, -3])

# ルンゲクッタ法で解く
def main4():
    h = (2*math.pi)/(64*w)

    # xと解析解との絶対誤差を描画
    x, e = Runge_Kutta(1, 0, h)
    x_axis = [i for i in range(len(x))]
    visualize(x_axis, x, 'time', '$x_c(t)$', '3.1.4 Runge Kutta x_c(t)', '3.1', x_ticks=[i for i in range(0, len(x), 5)])
    visualize(x_axis, e, 'time', 'error', '3.1.4 Runge Kutta error', '3.1', x_ticks=[i for i in range(0, len(x), 5)])

    h_list = [2*math.pi/w * 2**(-p) for p in range(2, 21)]
    error_list = []

    for h in h_list:
        x, e = Runge_Kutta(1, 0, h)
        e_max = max(e)
        error_list.append(e_max)
    
    # pに対する解析解との最大誤差を描画
    x_axis = [p for p in range(2, 21)]
    logEx = [math.log2(error) for error in error_list]
    title = '3.1.4 maximum error - p (Runge Kutta)'
    visualize(x_axis, logEx, 'p', 'log2 Ex', title, '3.1', x_ticks=[i for i in range(2, 21)], label='log2 Ex', least_square_flag=True, lrange=[0, 10])

# ルンゲクッタジル法で解く
def extra():
    h = (2*math.pi)/(64*w)

    # xと解析解との絶対誤差を描画
    x, e = Gill(1, 0, h)
    x_axis = [i for i in range(len(x))]
    visualize(x_axis, x, 'time', '$x_c(t)$', '3.1.extra Runge Kutta Gill x_c(t)', '3.1', x_ticks=[i for i in range(0, len(x), 5)])
    visualize(x_axis, e, 'time', 'error', '3.1.extra Runge Kutta Gill error', '3.1', x_ticks=[i for i in range(0, len(x), 5)])

    h_list = [2*math.pi/w * 2**(-p) for p in range(2, 21)]
    error_list = []

    for h in h_list:
        x, e = Gill(1, 0, h)
        x, e = Gill(1, 0, h)
        e_max = max(e)
        error_list.append(e_max)
    
    # pに対する解析解との最大誤差を描画
    x_axis = [p for p in range(2, 21)]
    logEx = [math.log2(error) for error in error_list]
    title = '3.1.extra maximum error - p (Runge Kutta Gill)'
    visualize(x_axis, logEx, 'p', 'log2 Ex', title, '3.1', x_ticks=[i for i in range(2, 21)], label='log2 Ex', least_square_flag=True, lrange=[0, 10])

# 各手法の計算時間と誤差の関係を描画
def main5():
    t_euler = []
    e_euler = []
    t_heun = []
    e_heun = []
    t_Runge_Kutta = []
    e_Runge_Kutta = []
    t_Gill = []
    e_Gill = []

    h_list = [2*math.pi/w * 2**(-p) for p in range(2, 16)]

    # euler法
    for h in tqdm(h_list):
        time_sta = time.perf_counter()
        # 計算時間が短いので、10回同じ計算をしたときの平均値を取る
        for i in range(10):
            x, e = euler(1, 0, h)
            emax = max(e)
        time_end = time.perf_counter()
        t_euler.append((time_end - time_sta)/10)
        e_euler.append(emax)

    # heun法
    for h in tqdm(h_list):
        time_sta = time.perf_counter()
        # 計算時間が短いので、10回同じ計算をしたときの平均値を取る
        for i in range(10):
            x, e = heun(1, 0, h)
            emax = max(e)
        time_end = time.perf_counter()
        t_heun.append((time_end - time_sta)/10)
        e_heun.append(emax)

    # Runge_Kutta法
    for h in tqdm(h_list):
        time_sta = time.perf_counter()
        # 計算時間が短いので、10回同じ計算をしたときの平均値を取る
        for i in range(10):
            x, e = Runge_Kutta(1, 0, h)
            emax = max(e)
        time_end = time.perf_counter()
        t_Runge_Kutta.append((time_end - time_sta)/10)
        e_Runge_Kutta.append(emax)
    
    # Gill法
    for h in tqdm(h_list):
        time_sta = time.perf_counter()
        # 計算時間が短いので、10回同じ計算をしたときの平均値を取る
        for i in range(10):
            x, e = Gill(1, 0, h)
            emax = max(e)
        time_end = time.perf_counter()
        t_Gill.append((time_end - time_sta)/10)
        e_Gill.append(emax)

    t_list = [t_euler, t_heun, t_Runge_Kutta, t_Gill]
    e_list = [e_euler, e_heun, e_Runge_Kutta, e_Gill]
    title = '3.1.5 Calculation time and error in each scheme'
    folder = '3.1'
    label = ['Euler', 'Heun', 'Runge Kutta', 'Gill']
    visualize_difference(e_list, t_list, 'error', 'time', title, folder, label)

if __name__ == '__main__':
    # main0()
    # main1()
    # main2()
    # main3()
    # main4()
    # extra()
    main5()