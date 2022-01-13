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

# パルス
def D(t):
    if 0 <= t <= Td:
        return 0.1
    else:
        return 0 

# xの微分
def diff_x(t, x, v):
    return v

# 境界条件：boundary=False -> 自由端、boundary=True: -> 固定端
# vの微分
def diff_v(t, x, v, boundary):
    differential_v = np.zeros(N, dtype=float)
    differential_v[0] = w**2*(D(t)-2*x[0] + x[1])
    differential_v[1:-1] = w**2*(x[:-2] - 2*x[1:-1] + x[2:])
    if boundary:
        differential_v[-1] = w**2*(x[-2] - 2*x[-1])
    else:
        differential_v[-1] = w**2*(x[-2] - x[-1])
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

# 教科書通りのルンゲ・クッタ・ジル法
def Gill_1(x, v, t, boundary):
    k1_v = diff_v(t, x, v, boundary)
    k1_x = diff_x(t, x, v)

    k2_v = diff_v(t+h/2, x + h/2*k1_x, v + h/2*k1_v, boundary)
    k2_x = diff_x(t+h/2, x + h/2*k1_x, v + h/2*k1_v)

    k3_v = diff_v(t+h/2, x + h*((1/math.sqrt(2)-1/2)*k1_x + (1-1/math.sqrt(2))*k2_x), v + h*((1/math.sqrt(2)-1/2)*k1_v + (1-1/math.sqrt(2))*k2_v), boundary)
    k3_x = diff_x(t+h/2, x + h*((1/math.sqrt(2)-1/2)*k1_x + (1-1/math.sqrt(2))*k2_x), v + h*((1/math.sqrt(2)-1/2)*k1_v + (1-1/math.sqrt(2))*k2_v))

    k4_v = diff_v(t+h, x + h*(-1/math.sqrt(2)*k2_x + (1+1/math.sqrt(2))*k3_x), v + h*(-1/math.sqrt(2)*k2_v + (1+1/math.sqrt(2))*k3_v), boundary)
    k4_x = diff_x(t+h, x + h*(-1/math.sqrt(2)*k2_x + (1+1/math.sqrt(2))*k3_x), v + h*(-1/math.sqrt(2)*k2_v + (1+1/math.sqrt(2))*k3_v))

    next_v = v + h/6*(k1_v + (2-math.sqrt(2))*k2_v + (2+math.sqrt(2))*k3_v + k4_v)
    next_x = x + h/6*(k1_x + (2-math.sqrt(2))*k2_x + (2+math.sqrt(2))*k3_x + k4_x)
    
    return next_x, next_v

# スキームを少し変更したルンゲ・クッタ・ジル法
def Gill_2(x, v, t, boundary):
    k1_v = diff_v(t, x, v, boundary)
    k1_x = diff_x(t, x, v)

    k2_v = diff_v(t+h/2, x + h/2*k1_x, v + h/2*k1_v, boundary)
    k2_x = diff_x(t+h/2, x + h/2*k1_x, v + h/2*k1_v)

    k3_v = diff_v(t+h/2, x + h*((1/math.sqrt(2)-1/2)*k1_x + (1-1/math.sqrt(2))*k2_x), v + h*((1/math.sqrt(2)-1/2)*k1_v + (1-1/math.sqrt(2))*k2_v), boundary)
    k3_x = diff_x(t+h/2, x + h*((1/math.sqrt(2)-1/2)*k1_x + (1-1/math.sqrt(2))*k2_x), v + h*((1/math.sqrt(2)-1/2)*k1_v + (1-1/math.sqrt(2))*k2_v))

    k4_v = diff_v(t+h, x + h*(-1/math.sqrt(2)*k2_x + (1+1/math.sqrt(2))*k3_x), v + h*(-1/math.sqrt(2)*k2_v + (1+1/math.sqrt(2))*k3_v), boundary)
    k4_x = diff_x(t+h, x + h*(-1/math.sqrt(2)*k2_x + (1+1/math.sqrt(2))*k3_x), v + h*(-1/math.sqrt(2)*k2_v + (1+1/math.sqrt(2))*k3_v))

    # k3の項を足し算ではなく引き算で計算する
    next_v = v + h/6*(k1_v + (2-math.sqrt(2))*k2_v + (2-math.sqrt(2))*k3_v + k4_v)
    next_x = x + h/6*(k1_x + (2-math.sqrt(2))*k2_x + (2-math.sqrt(2))*k3_x + k4_x)
    
    return next_x, next_v

def main(boundary, method, save_movie=False):
    # 計算手法によっては時間刻み幅を0.5よりも小さくする必要がある
    global h
    if method == 'Euler':
        h = 0.005
    elif method == 'Heun':
        h = 0.01
    else:
        h = 0.5
    
    # 動画のfps
    fps = 10
    # 現在の時間
    t = 0
    # 変位と速度
    x = np.zeros(N, dtype=float)
    v = np.zeros(N, dtype=float)

    # 全ての質点の変位と、終点の変位を格納する配列
    x_list = [x.copy()]
    x_N = [x[-1]]
    # パルスが到達するまでの時間
    T = N*math.sqrt(m/k)
    # ループの回数
    count = 0

    # 2周期分計算する
    while t <= (2*T)*2:
        t += h
        if method == 'Euler':
            x, v = euler(x, v, t, boundary)
        elif method == 'Heun':
            x, v = heun(x, v, t, boundary)
        elif method == 'Runge Kutta':
            x, v = Runge_Kutta(x, v, t, boundary)
        elif method == 'Runge Kutta Gill 1':
            x, v = Gill_1(x, v, t, boundary)
        elif method == 'Runge Kutta Gill 2':
            x, v = Gill_2(x, v, t, boundary)
        else:
            print('Error')
            exit()
        # 全ての時間の計算結果を格納すると、描画時に重たくなってしまうので、1/hの倍数のループの回数の時だけ格納する
        if count % int(1/h) == 0:
            x_list.append(x.copy())
            x_N.append(x[-1])
        count += 1
    
    # 固定端
    if boundary:
        title1 = f'3.2 {method} x_(N-1) (Fixed end)'
        title2 = f'3.2 {method} x_movie (Fixed end)'
    # 自由端
    else:
        title1 = f'3.2 {method} x_(N-1) (Free end)'
        title2 = f'3.2 {method} x_movie (Free end)'
    
    # 終端の頂点を描画
    x_axis = np.array([i for i in range(len(x_list))])
    visualize(x_axis, x_N, 'time', '$x_{{N-1}}$', title1, '3.2')

    # 各頂点の変位の時間変位を動画として出力
    x_axis = [i for i in range(N)]
    movie(x_axis, x_list, title2, fps, '3.2', save_movie=save_movie)

if __name__ == '__main__':
    method = ['Euler', 'Heun', 'Runge Kutta', 'Runge Kutta Gill 1', 'Runge Kutta Gill 2']
    for met in method:
        # 固定端
        main(boundary=True, method = met, save_movie=True)
        # 自由端
        main(boundary=False, method = met, save_movie=True)