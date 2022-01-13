import numpy as np
import math
import matplotlib.pyplot as plt
import copy

def Euler(t, h):
    k = 3.
    m = 4.
    x0 = 1.
    v0 = 0.
    w = (k / m)**0.5
    
    x_c = []
    v_c = []

    for i in range(t.shape[0]):
        if i == 0:
            x1 = x0
            v1 = v0
        else:
            x = copy.copy(x1)
            v = copy.copy(v1)
            x1 = x + h * v
            v1 = v - h * x * (k / m)
        
        x_c.append(x1)
        v_c.append(v1)
    
    x_a = x0 * np.cos(w * t)
    # v_a = -x0 * w * .sin(w * t[0,:])

    e_x = abs(x_c - x_a)
    return x_c, e_x

def Euler_error(t, h):

    x_c, e_x = Euler(t, h)

    maxE = np.amax(e_x)

    return maxE

def hoin(t, h):
    k = 3.
    m = 4.
    x0 = 1.
    v0 = 0.
    w = (k / m)**0.5

    x_c = []
    v_c = []

    for i in range(t.shape[0]):
        if i == 0:
            x = x0
            v = v0

        else:
            #ひとつ前の値を保持
            xp = copy.copy(x)
            vp = copy.copy(v)

            #ひとつ前の傾き
            dxp = vp
            dvp = -(k * xp) / m

            #x, vの更新
            k1_x = dxp
            k1_v = dvp
            k2_x = vp + dvp * h
            k2_v = -(xp + dxp * h) * k / m
            x = xp + (h * (k1_x + k2_x)) / 2
            v = vp + (h * (k1_v + k2_v)) / 2

        x_c.append(x)
        v_c.append(v)

    x_a = x0 * np.cos(w * t)
    # v_a = -x0 * w * .sin(w * t[0,:])

    e_x = abs(x_c - x_a)
    return x_c, x_a, e_x,

def hoin_error(t, h):
    x_c, x_a, e_x = hoin(t, h)
    maxE = np.amax(e_x)
    return maxE

def kadai311():
    k = 3.
    m = 4.
    w = (k / m)**0.5
    tf = (2 * math.pi) / w #時間範囲

    h = (2 * math.pi) / (64 * w)
    t = np.arange(0, tf, h)
    x_ch, x_ah, e_xh =hoin(t, h)
    x_ce, e_xe = Euler(t, h)

    # プロット
    x1 = t #3.1.1時刻
    # x2 = p #3.1.2 p
    y1 = x_ch # 3.1.1ホイン法での数値解
    y2 = x_ah # 3.1.1解析解
    y3 = e_xh # 3.1.1ホイン法誤差
    y4 = x_ce # 3.1.1オイラー法数値解
    y5 = e_xe # 3.1.1オイラー法誤差

    # print(y2)
    # print(x_a)

    fig, ax = plt.subplots()

    ax.set_xlabel('t')  # x軸ラベル
    ax.set_ylabel('x')  # y軸ラベル
    # ax.set_title(r'$\sin(x)$ and $\cos(x)$') # グラフタイトル

    ax.grid()            # 罫線

    ax.plot(x1, y1, color="c", label="x_ch")
    ax.plot(x1, y2, color="y", label="x_ah")
    ax.plot(x1, y3, color="b", label="e_xh")
    ax.plot(x1, y4, color="r", label="x_ce")
    ax.plot(x1, y5, color="r", label="e_xe")
    # ax.plot(x2, y6, color="r", label="h_error")

    ax.legend(loc=0)    # 凡例
    fig.tight_layout()  # レイアウトの設定
    plt.show()

def kadai312():
    k = 3.
    m = 4.
    w = (k / m)**0.5
    tf = (2 * math.pi) / w #時間範囲

    #3.1.2用
    error_x = np.array([]) #誤差の最大値の配列
    p = np.arange(2,21,1) #時間刻み指定
    p_lsm = np.zeros(16) #最小2乗法用
    e_lsm = np.zeros(16) #最小2乗法用

    for i in p:
        h = (2 * math.pi) / ((2**i) * w)
        t = np.arange(0, tf, h)
        # print(t.shape)
        # maxE = Euler_error(t, h)
        maxE = hoin_error(t, h)
        # print(maxE)
        error_x = np.append(error_x, maxE)

    y6 = np.log2(error_x)

    for i in range(16):
        p_lsm[i] = p[i+3]
        e_lsm[i] = y6[i+3]

    lsm_co = np.polyfit(p_lsm, e_lsm, 1)
    print(f"{lsm_co[0]}x+{lsm_co[1]}")

    # プロット
    x2 = p #3.1.2 p

    # print(y2)
    # print(x_a)

    fig, ax = plt.subplots()

    ax.set_xlabel('t')  # x軸ラベル
    ax.set_ylabel('x')  # y軸ラベル
    # ax.set_title(r'$\sin(x)$ and $\cos(x)$') # グラフタイトル

    ax.grid()            # 罫線

    ax.plot(x2, y6, color="r", label="h_error")

    ax.legend(loc=0)    # 凡例
    fig.tight_layout()  # レイアウトの設定
    plt.show()

# kadai311()

kadai312()

