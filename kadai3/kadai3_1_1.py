import numpy as np
import math
import matplotlib.pyplot as plt
import copy

def Euler():
    k = 3.
    m = 4.
    x0 = 1.
    v0 = 0.
    w = (k / m)**0.5
    tf = (2 * math.pi) / w
    h = (2 * math.pi) / (64 * w)
    t = np.arange(0, tf, h)
    print("t", t.shape)
    x_c = np.array([])
    v_c = np.array([])

    for i in range(t.shape[0]):
        if i == 0:
            x1 = x0
            v1 = v0
        else:
            x = copy.copy(x1)
            v = copy.copy(v1)
            x1 = x + h * v
            v1 = v - h * x * (k / m)
        
        x_c = np.append(x_c, x1)
        v_c = np.append(v_c, v1)
    
    x_a = x0 * np.cos(w * t)
    # v_a = -x0 * w * .sin(w * t[0,:])

    e_x = abs(x_c - x_a)
    return x_c, x_a, e_x, t

x_c, x_a, e_x, t = Euler()

# プロット
x1 = t
y1 = x_c
y2 = x_a
y3 = e_x
# print(y2)
# print(x_a)

fig, ax = plt.subplots()

# c1,c2,c3 = "blue","green","red"     #グラフ色
# l = ["Jacobi_r","Jacobi_e","Gauss_r","Gauss_e","sor_r","sor_e"]   # 各ラベル

ax.set_xlabel('t')  # x軸ラベル
ax.set_ylabel('x')  # y軸ラベル
# ax.set_title(r'$\sin(x)$ and $\cos(x)$') # グラフタイトル
ax.grid()            # 罫線
# ax.set_yscale("log") #yを対数に
# ax.set_xlim([0, 7000]) # x方向の描画範囲を指定
# ax.set_ylim([0, 1])    # y方向の描画範囲を指定

ax.plot(x1, y1, color="c", label="x_c")
ax.plot(x1, y2, color="y", label="x_a")
ax.plot(x1, y3, color="y", label="e_x")

ax.legend(loc=0)    # 凡例
fig.tight_layout()  # レイアウトの設定
plt.show()


        
        
        

