import numpy as np
import math
import matplotlib.pyplot as plt
import copy

def Euler_error(t, h):
    k = 3.
    m = 4.
    x0 = 1.
    v0 = 0.
    w = (k / m)**0.5

    # print("t", t.shape)
    x_c = []
    v_c = []
    maxE = 0.

    for i in range(t.shape[0]):
        if i == 0:
            x1 = x0
            v1 = v0
        else:
            x = copy.copy(x1)
            v = copy.copy(v1)
            x1 = x + h * v
            v1 = v - h * x * (k / m)
        
        # x_c = np.append(x_c, x1):遅い
        # v_c = np.append(v_c, v1)
        x_c.append(x1)
        v_c.append(v1)
    
    x_a = x0 * np.cos(w * t)
    # v_a = -x0 * w * .sin(w * t[0,:])

    e_x = abs(x_c - x_a)

    # for i in range(e_x.shape[0]):
    #     if maxE < abs(e_x[i]):
    #         maxE = abs(e_x[i])

    maxE = np.amax(e_x)

    return maxE


k = 3.
m = 4.
w = (k / m)**0.5
tf = (2 * math.pi) / w #時間範囲
error_x = np.array([]) #誤差の最大値の配列
p = np.arange(2,21,1) #時間刻み指定
p_lsm = np.zeros(16) #最小2乗法用
e_lsm = np.zeros(16) #最小2乗法用

for i in p:
    h = (2 * math.pi) / ((2**i) * w)
    t = np.arange(0, tf, h)
    print(t.shape)
    maxE = Euler_error(t, h)
    print(maxE)
    error_x = np.append(error_x, maxE)

y1 = np.log2(error_x)

for i in range(16):
    p_lsm[i] = p[i+3]
    e_lsm[i] = y1[i+3]

lsm_co = np.polyfit(p_lsm, e_lsm, 1)
print(f"{lsm_co[0]}x+{lsm_co[1]}")

# プロット
x1 = p
x2 = p_lsm
# y1 = np.log2(error_x)
y2 = (lsm_co[0]*p_lsm + lsm_co[1])


fig, ax = plt.subplots()

# c1,c2,c3 = "blue","green","red"     #グラフ色
# l = ["Jacobi_r","Jacobi_e","Gauss_r","Gauss_e","sor_r","sor_e"]   # 各ラベル

ax.set_xlabel('p')  # x軸ラベル
ax.set_ylabel('max_Ex')  # y軸ラベル
# ax.set_title(r'$\sin(x)$ and $\cos(x)$') # グラフタイトル
ax.grid()            # 罫線
# ax.set_yscale("log") #yを対数に
# ax.set_xlim([0, 7000]) # x方向の描画範囲を指定
# ax.set_ylim([0, 1])    # y方向の描画範囲を指定

ax.plot(x1, y1, color="c", label="max_error")
ax.plot(x2, y2, color="y", label="lsm")
# ax.plot(x1, y3, color="y", label="e_x")

ax.legend(loc=0)    # 凡例
fig.tight_layout()  # レイアウトの設定
plt.show()