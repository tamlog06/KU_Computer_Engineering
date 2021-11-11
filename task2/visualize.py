import numpy as np
import matplotlib.pyplot as plt

# 反復計算法のそれぞれについて計算回数と誤差の関係を一つのグラフに重ねて可視化
def visualization_convergence(jacobi_err, gauss_err, sor_err):
    plt.xlabel("$loop$")
    plt.ylabel("$error$")
    plt.grid()
    plt.axhline(0, color='k')
    plt.title('Residual norm')

    plt.scatter(0, jacobi_err[0], c="b", label="Jacobi")
    plt.scatter(0, gauss_err[0], c="g", label="Gauss_Seidel")
    plt.scatter(0, sor_err[0], c="r", label="SOR")

    for loop in range(1, len(jacobi_err), 100):
        err = jacobi_err[loop]
        plt.scatter(loop, err, c="b")

    for loop in range(1, len(gauss_err), 50):
        err = gauss_err[loop]
        plt.scatter(loop, err, c="g")

    for loop in range(1, len(sor_err)):
        err = sor_err[loop]
        plt.scatter(loop, err, c="r")

    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.set_yscale('log')  # メイン: y軸をlogスケールで描く
    
    plt.grid(which="both") # グリッド表示。"both"はxy軸両方にグリッドを描く。
    plt.legend()
    plt.show()

# 残差ノルムと真の値との差のノルムを比較
def visualization_norm_difference(error, true_error, step, title):
    plt.xlabel("$loop$")
    plt.ylabel("True error norm - Residual norm")
    plt.grid()
    plt.axhline(0, color='k')
    plt.title(title)

    for loop in range(10, len(error), step):
        err = error[loop]
        true_err = true_error[loop]
        plt.scatter(loop, true_err - err, c="b")

    ax = plt.gca()
    ax.spines['top'].set_color('none')
    
    plt.grid(which="both") # グリッド表示。"both"はxy軸両方にグリッドを描く。
    plt.show()