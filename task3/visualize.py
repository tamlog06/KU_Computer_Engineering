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

def anime(x, y, title, fps, save_gif=False):
    # fig, axオブジェクトを作成
    fig, ax = plt.subplots()
    plt.title(title)

    # グラフのリスト作成
    ims=[]
    for i in range(0, len(y)): 
        im = ax.plot(x, y[i], color='b')
        ims.append(im)

    # 各軸のラベル
    ax.set_xlabel(r"$x$", fontsize=15)
    ax.set_ylabel(r"$y$", fontsize=15)
    # グラフの範囲を設定
    # ax.set_xlim([0, 100])
    # ax.set_ylim([-0.1, 0.1]) 

    # ArtistAnimationにfigオブジェクトとimsを代入してアニメーションを作成
    anim = animation.ArtistAnimation(fig, ims, interval=fps)
    plt.show()
    if save_gif:
        anim.save(f'task3/img/{title}.mp4', writer='ffmpeg', fps=100) # fpsはデフォルトの5