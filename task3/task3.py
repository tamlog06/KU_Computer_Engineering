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
def main2(boundary, save_movie):
    # u_k1: 一つ前の時刻のu（初期状態は時刻-dt）
    # u_k2: 二つ前の時刻のu（初期状態は時刻０）
    u_k1 = np.zeros(N, dtype=float)
    u_k2 = np.zeros(N, dtype=float)
    # 時刻 -dt
    u_k2[0] = (tanh((-dt-Ts)/a) + 1)/2 - (tanh((-dt-Ts-TD)/a) + 1)/2
    # 時刻 0
    u_k1[0] = (tanh((0-Ts)/a) + 1)/2 - (tanh((0-Ts-TD)/a) + 1)/2

    # 全ての質点の位置を含むリスト
    u_list = [u_k1]
    # 終端の質点の位置を含むリスト
    u_N = [u_k1[-1]]
    # 全体の長さが1だから、周期は1/cs。ここでは四周期
    T = Ts + 1/c*4
    k = 1
    # 次の時刻tk
    tk = dt
    while tk < T:
        # 次の時刻の位置のリスト
        u = np.zeros(N, dtype=float)
        u[0] = (tanh((tk-Ts)/a) + 1)/2 - (tanh((tk-Ts-TD)/a) + 1)/2
        u[1:-1] = -u_k2[1:-1] + mu*u_k1[0:-2] + 2*(1-mu)*u_k1[1:-1] + mu*u_k1[2:]
        # 固定端
        if boundary:
            u[-1] = -u_k2[-1] + mu*u_k1[-2] + 2*(1-mu)*u_k1[-1]
        # 自由端
        else:
            u[-1] = -u_k2[-1] + mu*u_k1[-2] + (2-mu)*u_k1[-1]
        
        if k == 250:
            x_palse = u

        u_N.append(u[-1])
        u_list.append(u)
        u_k2 = u_k1
        u_k1 = u
        k += 1
        tk += dt
    
    # 固定端
    if boundary:
        title1 = '3.3.2 u_N (Fixed end)'
        title2 = '3.3.2 u palse (Fixed end)'
        title3 = '3.3.2 u_movie (Fixed end)'
    # 自由端
    else:
        title1 = '3.3.2 u_N (Free end)'
        title2 = '3.3.2 u palse (Free end)'
        title3 = '3.3.2 u_movie (Free end)'
    
    # 終端の変位の時間変化を描画
    x_axis = np.array([i for i in range(len(u_N))])*dt
    visualize(x_axis, u_N, 'time', '$u_{{N-1}}$', title1, '3.3')

    x_axis = [i for i in range(N)]
    # パルスの空間幅を確認できる時間での各質点の変位の出力
    visualize(x_axis, x_palse, 'n', 'u', title2, '3.3')
    # 各質点の変位の時間変化を動画として出力
    movie(x_axis, u_list, title3, 10, '3.3', save_movie=save_movie)

# boundary=True: 固定端 False: 自由端
def main3(boundary, save_movie):
    # u_k1: 一つ前の時刻のu（初期状態は時刻-dt）
    # u_k2: 二つ前の時刻のu（初期状態は時刻０）
    u_k1 = np.zeros(N, dtype=float)
    u_k2 = np.zeros(N, dtype=float)
    # 時刻 -dt
    u_k2[0] = (tanh((-dt-Ts)/a) + 1)/2 - (tanh((-dt-Ts-TD)/a) + 1)/2
    # 時刻 0
    u_k1[0] = (tanh((0-Ts)/a) + 1)/2 - (tanh((0-Ts-TD)/a) + 1)/2

    # 全ての質点の位置を含むリスト
    u_list = [u_k1]
    # 終端の質点の位置を含むリスト
    u_N = [u_k1[-1]]
    # 全体の長さが1だから、周期は1/cs。ここでは四周期
    T = Ts + 1/c*4
    k = 1
    # 次の時刻tk
    tk = dt
    while tk < T:
        # 次の時刻の位置のリスト
        u = np.zeros(N, dtype=float)
        if Ts <= k*dt <= Ts + TD:
            u[0] = 1.0
        u[1:-1] = -u_k2[1:-1] + mu*u_k1[0:-2] + 2*(1-mu)*u_k1[1:-1] + mu*u_k1[2:]
        # 固定端
        if boundary:
            u[N-1] = -u_k2[N-1] + mu*u_k1[N-2] + 2*(1-mu)*u_k1[N-1]
        # 自由端
        else:
            u[N-1] = -u_k2[N-1] + mu*u_k1[N-2] + (2-mu)*u_k1[N-1]

        if k == 250:
            x_palse = u
        
        u_list.append(u)
        u_N.append(u[-1])
        u_k2 = u_k1
        u_k1 = u
        k += 1
        tk += dt
    
    # 固定端
    if boundary:
        title1 = '3.3.3 u_N (Fixed end)'
        title2 = '3.3.3 u palse (Fixed end)'
        title3 = '3.3.3 u_movie (Fixed end)'
    # 自由端
    else:
        title1 = '3.3.3 u_N (Free end)'
        title2 = '3.3.3 u palse (Free end)'
        title3 = '3.3.3 u_movie (Free end)'
    
    # 終端の変位の時間変化を描画
    x_axis = np.array([i for i in range(len(u_N))])*dt
    visualize(x_axis, u_N, 'time', '$u_{{N-1}}$', title1, '3.3')

    x_axis = [i for i in range(N)]
    # パルスの空間幅を確認できる時間での各質点の変位の出力
    visualize(x_axis, x_palse, 'n', 'u', title2, '3.3')
    # 各質点の変位の時間変化を動画として出力
    movie(x_axis, u_list, title3, 10, '3.3', save_movie=save_movie)

if __name__ == '__main__':
    # 固定端
    main2(boundary=True, save_movie=True)
    main3(boundary=True, save_movie=True)

    # 自由端
    main2(boundary=False, save_movie=True)
    main3(boundary=False, save_movie=True)