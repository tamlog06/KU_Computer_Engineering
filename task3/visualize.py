import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os

def reg1dim(x, y):
    n = len(x)
    a = ((np.dot(x, y)- y.sum() * x.sum()/n)/
        ((x ** 2).sum() - x.sum()**2 / n))
    b = (y.sum() - a * x.sum())/n
    return a, b

def visualize(x, y, xlabel, ylabel, title, x_ticks=True, ylim=None, label=None, log_flag=False, least_square_flag=False, lrange=[0, -1]):
    fig = plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axhline(0, color='k')
    plt.title(title)

    x = np.array(x)
    y = np.array(y)

    ax = plt.gca()
    if not label == None:
        ax.plot(x, y, c='b', linestyle='solid', label=label)
    else:
        ax.plot(x, y, c='b', linestyle='solid')
    ax.spines['top'].set_color('none')

    if log_flag:
        ax.set_yscale('log')
    
    if least_square_flag:
        if lrange[1] == None:
            a, b = reg1dim(x[lrange[0]:], y[lrange[0]:])
            ax.plot([x[lrange[0]], x[-1]], [a*x[lrange[0]]+b, a*x[-1]+b], c='r', label=f'y={a:.2e}x+{b:.2e}')
        else:
            a, b = reg1dim(x[lrange[0]:lrange[1]+1], y[lrange[0]:lrange[1]+1])
            ax.plot([x[lrange[0]], x[lrange[1]]], [a*x[lrange[0]]+b, a*x[lrange[1]]+b], c='r', label=f'y={a:.2e}x+{b:.2e}')

    ax.grid(which='both')
    if label or least_square_flag:
        ax.legend(loc='upper right')
    if x_ticks:
        plt.xticks(x)
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.show()
    os.makedirs('task3/img', exist_ok=True)
    fig.savefig(f'task3/img/{title}.png')

def anime(x, y):
    # fig, axオブジェクトを作成
    fig, ax = plt.subplots()

    # グラフのリスト作成
    ims=[]
    for i in range(len(y)): 
        # y = np.sin(x + 2*np.pi*(i/10))
        im = ax.plot(x, y[i], color='b')
        # グラフをリストに加える
        ims.append(im)

    # 各軸のラベル
    ax.set_xlabel(r"$x$", fontsize=15)
    ax.set_ylabel(r"$y$", fontsize=15)
    # グラフの範囲を設定
    # ax.set_xlim([0, 100])
    ax.set_ylim([-100, 100]) 

    # ArtistAnimationにfigオブジェクトとimsを代入してアニメーションを作成
    anim = animation.ArtistAnimation(fig, ims, interval=10)
    plt.show()

# # 反復計算法のそれぞれについて計算回数と誤差の関係を一つのグラフに重ねて可視化
# def visualization_convergence(jacobi_err, gauss_err, sor_err, title, ylim=None):
#     fig = plt.figure()
#     plt.xlabel("loop")
#     plt.ylabel("error")
#     plt.axhline(0, color='k')
#     plt.title(title)

#     x_jacobi = [i for i in range(len(jacobi_err))]
#     x_gauss = [i for i in range(len(gauss_err))]
#     x_sor = [i for i in range(len(sor_err))]

#     ax = plt.gca()

#     ax.plot(x_jacobi, jacobi_err, c='b', linestyle='solid', label='Jacobi')
#     ax.plot(x_gauss, gauss_err, c='g', linestyle='solid', label='Gauss Seidel')
#     ax.plot(x_sor, sor_err, c='r', linestyle='solid', label='SOR')

#     ax.spines['top'].set_color('none')
#     ax.set_yscale('log')  # メイン: y軸をlogスケールで描く
#     if ylim:
#         plt.ylim(0, ylim)
    
#     ax.grid(which="both") # グリッド表示。"both"はxy軸両方にグリッドを描く。
#     ax.legend()
#     plt.show()
#     fig.savefig(f'img/{title}.png')

# # 残差ノルムと真の値との差のノルムを比較
# def visualization_norm_difference(error, true_error, title, ylim=None):
#     fig = plt.figure()
#     plt.xlabel("$loop$")
#     plt.ylabel("abs(True error norm - Residual norm)")
#     plt.axhline(0, color='k')
#     plt.title(title)

#     x = [i for i in range(len(error))]
#     norm = np.abs(np.array(true_error) - np.array(error))

#     ax = plt.gca()
#     ax.spines['top'].set_color('none')
#     if ylim:
#         plt.ylim(0, ylim)

#     ax.plot(x, norm, c='b', linestyle='solid')
    
#     ax.grid(which="both") # グリッド表示。"both"はxy軸両方にグリッドを描く。
#     plt.show()
#     fig.savefig(f'img/{title}.png')

# # 反復回数に対するノルムの変化を表示
# def visualization_norm(norm, title, ylim=None, log=False):
#     fig = plt.figure()
#     plt.xlabel("$loop$")
#     plt.ylabel("norm")
#     plt.axhline(0, color='k')
#     plt.title(title)

#     x = [i for i in range(len(norm))]

#     ax = plt.gca()
#     ax.spines['top'].set_color('none')
#     if ylim:
#         plt.ylim(0, ylim)

#     ax.plot(x, norm, c='b', linestyle='solid')
#     if log:
#         ax.set_yscale('log')  # y軸をlogスケールで描く
    
#     ax.grid(which="both") # グリッド表示。"both"はxy軸両方にグリッドを描く。
#     plt.show()
#     fig.savefig(f'img/{title}.png')