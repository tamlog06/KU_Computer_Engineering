import numpy as np
import matplotlib.pyplot as plt

# 反復計算法のそれぞれについて計算回数と誤差の関係を一つのグラフに重ねて可視化
def visualization_convergence(jacobi_err, gauss_err, sor_err, title, ylim=None):
    fig = plt.figure()
    plt.xlabel("loop")
    plt.ylabel("error")
    plt.axhline(0, color='k')
    plt.title(title)

    x_jacobi = [i for i in range(len(jacobi_err))]
    x_gauss = [i for i in range(len(gauss_err))]
    x_sor = [i for i in range(len(sor_err))]

    ax = plt.gca()

    ax.plot(x_jacobi, jacobi_err, c='b', linestyle='solid', label='Jacobi')
    ax.plot(x_gauss, gauss_err, c='g', linestyle='solid', label='Gauss Seidel')
    ax.plot(x_sor, sor_err, c='r', linestyle='solid', label='SOR')

    ax.spines['top'].set_color('none')
    ax.set_yscale('log')  # メイン: y軸をlogスケールで描く
    if ylim:
        plt.ylim(0, ylim)
    
    ax.grid(which="both") # グリッド表示。"both"はxy軸両方にグリッドを描く。
    ax.legend()
    plt.show()
    fig.savefig(f'img/{title}.png')

# 残差ノルムと真の値との差のノルムを比較
def visualization_norm_difference(error, true_error, title, ylim=None):
    fig = plt.figure()
    plt.xlabel("$loop$")
    plt.ylabel("abs(True error norm - Residual norm)")
    plt.axhline(0, color='k')
    plt.title(title)

    x = [i for i in range(len(error))]
    norm = np.abs(np.array(true_error) - np.array(error))

    ax = plt.gca()
    ax.spines['top'].set_color('none')
    if ylim:
        plt.ylim(0, ylim)

    ax.plot(x, norm, c='b', linestyle='solid')
    
    ax.grid(which="both") # グリッド表示。"both"はxy軸両方にグリッドを描く。
    plt.show()
    fig.savefig(f'img/{title}.png')

# 反復回数に対するノルムの変化を表示
def visualization_norm(norm, title, ylim=None, log=False):
    fig = plt.figure()
    plt.xlabel("$loop$")
    plt.ylabel("norm")
    plt.axhline(0, color='k')
    plt.title(title)

    x = [i for i in range(len(norm))]

    ax = plt.gca()
    ax.spines['top'].set_color('none')
    if ylim:
        plt.ylim(0, ylim)

    ax.plot(x, norm, c='b', linestyle='solid')
    if log:
        ax.set_yscale('log')  # y軸をlogスケールで描く
    
    ax.grid(which="both") # グリッド表示。"both"はxy軸両方にグリッドを描く。
    plt.show()
    fig.savefig(f'img/{title}.png')