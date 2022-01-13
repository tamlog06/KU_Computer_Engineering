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

def o_rugen(t, h):
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
            k2_x = vp + k1_v * (h/2)
            k2_v = -(xp + k1_x * (h/2)) * k / m
            k3_x = vp + k2_v * (h/2)
            k3_v = -(xp + k2_x * (h/2)) * k / m
            k4_x = vp + k3_v * h
            k4_v = -(xp + k3_x * h) * k / m
            x = xp + (k1_x + 2*k2_x + 2*k3_x + k4_x) * (h / 6)
            v = vp + (k1_v + 2*k2_v + 2*k3_v + k4_v) * (h / 6)

        x_c.append(x)
        v_c.append(v)

    x_a = x0 * np.cos(w * t)
    # v_a = -x0 * w * .sin(w * t[0,:])

    e_x = abs(x_c - x_a)
    return x_c, x_a, e_x,

def o_rugen_error(t, h):
    out = o_rugen(t, h)
    maxE = np.amax(out[2])
    return maxE

def kadai311():
    k = 3.
    m = 4.
    w = (k / m)**0.5
    tf = (2 * math.pi) / w #時間範囲
    h = (2 * math.pi) / (64 * w)
    t = np.arange(0, tf, h)
    rugen_out = o_rugen(t, h)
    Euler_out = Euler(t, h)

    #プロット
    x1 = t
    y1 = rugen_out[0]
    y2 = rugen_out[1]
    y3 = rugen_out[2]
    y4 = Euler_out[0]
    y5 = Euler_out[1]

    fig, ax = plt.subplots()

    ax.set_xlabel('t')  # x軸ラベル
    ax.set_ylabel('x')  # y軸ラベル
    # ax.set_title(r'$\sin(x)$ and $\cos(x)$') # グラフタイトル
    ax.grid()            # 罫線

    # ax.plot(x1, y1, color="b", label="r_xc")
    # ax.plot(x1, y2, color="g", label="xa")
    ax.plot(x1, y3, color="r", label="r_ex")
    # ax.plot(x1, y4, color="c", label="E_xc")
    # ax.plot(x1, y5, color="y", label="E_ex")

    ax.legend(loc=0)    # 凡例
    fig.tight_layout()  # レイアウトの設定
    plt.show()

def kadai312():
    k = 3.
    m = 4.
    w = (k / m)**0.5
    tf = (2 * math.pi) / w #時間範囲

    error_x = np.array([]) #誤差の最大値の配列
    p = np.arange(2,21,1) #時間刻み指定
    # p_lsm = np.zeros(10) #最小2乗法用
    # e_lsm = np.zeros(10) #最小2乗法用

    for i in p:
        h = (2 * math.pi) / ((2**i) * w)
        t = np.arange(0, tf, h)
        # maxE = Euler_error(t, h)
        maxE = o_rugen_error(t, h)
        # print(maxE)
        error_x = np.append(error_x, maxE)

    y1 = np.log2(error_x)

    p_lsm = p[1:11]
    e_lsm = y1[1:11]
    # for i in range(10):
    #     p_lsm[i] = p[i+1]
    #     e_lsm[i] = y1[i+1]

    lsm_co = np.polyfit(p_lsm, e_lsm, 1) #最小2乗近似
    # print(f"{lsm_co[0]}x+{lsm_co[1]}") 

    # for i in range(10):
    #     y2 = np.zeros(10)
    #     y2[i] = p[i+1] * lsm_co[0] + lsm_co[1]
    


    #プロット
    x1 = p
    x2 = p[1:11]
    y2 = p[1:11]*lsm_co[0] + lsm_co[1]

    fig, ax = plt.subplots()

    ax.set_xlabel('t')  # x軸ラベル
    ax.set_ylabel('x')  # y軸ラベル
    # ax.set_title(r'$\sin(x)$ and $\cos(x)$') # グラフタイトル
    ax.grid()            # 罫線

    ax.plot(x1, y1, color="b", label="e_x")
    ax.plot(x2, y2, color="g", label=f"{lsm_co[0]}x+{lsm_co[1]}")

    ax.legend(loc=0)    # 凡例
    fig.tight_layout()  # レイアウトの設定
    plt.show()

# kadai311()

kadai312()