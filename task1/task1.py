import matplotlib.pyplot as plt
import numpy as np
import math

# 二項係数
def comb(n: float, k: int):
    if k < 0:
        print('error: invalid number. k < 0')
        return 0
    if (n == k or k == 0):
        return 1
    
    result = 1
    for i in range(k):
        result *= n-i
    for i in range(1, k+1):
        result /= i
    return result

# ルジャンドル多項式のaの値
def Legendre_a(n: int):
    a = []
    for k in range(n+1):
        a.append(2**n * comb(n, k) * comb((n+k-1)/2, n))
    return a

# ルジャンドル多項式のPの値
def Legendre_P(n: int, x: float):
    P = 0
    a = Legendre_a(n)
    for k in range(n+1):
        P += a[k] * x**k
    return P


#二分法（方程式の関数項、探索区間の左端、探索区間の右端、誤差範囲、最大反復回数）
def bisection(n, x_min, x_max, error=1e-9, max_loop=100, option=False):
    # 計算途中の値と反復数を含むリスト
    middle_value = []
    #初期値を表示
    num_calc = 0  #計算回数

    #中間値の定理の条件を満たすか調べる
    if(0.0 < Legendre_P(n, x_min)*Legendre_P(n, x_max)):
        print("error: Section definition is invalid (0.0 < P(n, x_min)*P(n, x_max)).")
        return False

    while(True):
        #新たな中間値の計算
        x_mid = (x_max + x_min)/2.0

        #探索区間を更新
        if (0.0 < Legendre_P(n, x_mid)*Legendre_P(n, x_max)):
            x_max = x_mid
        else:
            x_min = x_mid

        num_calc += 1
        middle_value.append([x_mid, num_calc])

        #「誤差範囲が一定値以下」または「計算回数が一定値以上」ならば終了
        if x_max-x_min <= error:
            break
        elif  max_loop <= num_calc:
            print(f'error: too complex to caluculate within {max_loop} times')
            return False

    #最終的に得られた解
    print("x = {:.9f}".format(x_mid))
    print(num_calc)

    # 課題1.3で計算回数を出すために、計算途中の値を出力する場合はmiddle_valueも出力
    if option:
        return x_mid, middle_value
    else:
        return x_mid

# 二分法で全ての解を出す
# option が True なら自動で初期区間を出して計算する（課題1.4）
def bisection_all(n: int, x=[], error=1e-9, option=False):
    if option:
        # 1 <= n <= 5の場合は既知とする
        if 1 <= n <= 5:
            return newton_all(n, option=True)
        else:
            # nの解は、[-1, 1]のうちのn-1の解の区間に１つづつ存在するので、その区間を再帰的に計算
            previous_ans = bisection_all(n-1, option=True)
            previous_ans.append(-1)
            previous_ans.append(1)
            previous_ans.sort()
            x = []
            for i in range(len(previous_ans)-1):
                x.append([previous_ans[i], previous_ans[i+1]])
            return bisection_all(n, x, error=error)
    else:
        # 与えられた区間の中の解を全て出す
        result = []
        for x_min, x_max in x:
            ans = bisection(n, x_min, x_max, error=error)
            if ans != False:
                result.append(ans)
        return result

# newton法で解を一つ出す
def newton(x0, n: int, error=1e-9, max_loop=100, option=False):
    # 計算途中の値
    middle_value = []

    for i in range(max_loop):
        x1 = x0 - Legendre_P(n, x0) / diff_Legendre(n, x0)
        # 計算した値が誤差の範囲内であれば終了
        if abs(x1 - x0) < error:
            break
        x0 = x1

        middle_value.append([x1, i+1])

    if i >= max_loop:
        f'error: too complex to caluculate within {max_loop} times'

    # 課題1.3で計算回数を出すために、計算途中の値を出力する場合はmiddle_valueも出力
    if option:
        return x1, middle_value
    else:
        return x1

#ルジャンドル多項式の微分
def diff_Legendre(n: int, x):
    # オーバーフロー対策
    if (x == 1) or (x == -1):
            x += 0.001
    return n*(Legendre_P(n-1, x) - x*Legendre_P(n,x))/(1-x**2)

# newton法で全ての解を出す
# option==Trueなら、初期近似解を自動で、Falseなら決め打ち
def newton_all(n: int, x=[], option=False):
    result = []
    if option:
        for i in range(math.ceil(n)):
            x = np.cos(((i+0.75)/(n+0.5))*np.pi)
            point = newton(x, n)
            if abs(point) < 1e-9:
                point = 0
            result.append(point)
    else:
        if not x:
            print("list is not added")
        for x0 in x:
            result.append(newton(x0, n))
    return result

#可視化（方程式の関数項、グラフ左端、グラフ右端、方程式の解）
def visualization(n: int, x_min, x_max, x_solved, title):
    plt.xlabel("$x$")  #x軸の名前
    plt.ylabel("$f(x)$")  #y軸の名前
    plt.grid()  #点線の目盛りを表示
    plt.axhline(0, color='k')  #f(x)=0の線
    plt.title(title)

    #関数
    exact_x = np.arange(x_min,x_max, (x_max-x_min)/500.0)
    exact_y = Legendre_P(n, exact_x)

    plt.plot(exact_x,exact_y, label="$f(x)$", color='r')  #関数を折線グラフで表示

    for x in x_solved:
        plt.scatter(x,0.0, c='b')  #数値解を点グラフで表示
        plt.text(x,0.0, "$x$ = {:.9f}".format(x), va='bottom', color='b')
    plt.show()  #グラフを表示

# 計算回数と誤差の関係を可視化
def visualization_convergence(n: int, middle_value, true_value):
    plt.xlabel("$loop$")
    plt.ylabel("$error$")
    plt.grid()
    plt.axhline(0, color='k')

    for middle, loop in middle_value:
        plt.scatter(loop, abs(true_value - middle), c="b")

    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.set_yscale('log')  # メイン: y軸をlogスケールで描く
    
    plt.grid(which="both") # グリッド表示。"both"はxy軸両方にグリッドを描く。
    plt.show()

# 課題1.1
def main1(n, k):
    print(comb(n, k))
    print(Legendre_a(n=5))
    print(Legendre_a(n=10))

# 課題1.2
# x_legendreは初期区間のリスト
def main2(x_legendre):
    n = 5
    result_bisection = bisection_all(n, x_legendre)
    title = f'Legendre {n}'
    visualization(n, -1, 1, result_bisection, title)
    print(result_bisection)

# 課題1.3
# x_newtonは初期近似解のリスト
def main3(x_newton):
    n = 5
    result_newton = newton_all(n, x_newton)
    title = f'Legendre {n}'
    visualization(n, -1, 1, result_newton, title)
    print(result_newton)

    # 反復の速さをグラフに描画
    # [-1, -0.75]の区間の解を出す時の反復の速さ
    result, middle_value = bisection(n, -1, -0.75, option=True)
    # 修正量の絶対値が10^-15以下として、真の値とする
    true_value = bisection(n, -1, -0.75, error=1e-15, max_loop=1000)
    visualization_convergence(n, middle_value, true_value)
    
    # 二分法と同じ解を出す
    result, middle_value = newton(-1, n, option=True)
    # 修正量の絶対値が10^-15以下として、真の値とする
    true_value = newton(-1, n, error=1e-15, max_loop=1000)
    visualization_convergence(n, middle_value, true_value)

# 課題1.4
def main4(n):
    print(newton_all(n, option=True))
    print(bisection_all(n, option=True))

    # 一応 n=6~10 で可視化
    for i in range(6, 11):
        ans_bisection = bisection_all(i, option=True)
        title_bisection = f'Legendre {i} bisection method'
        visualization(i, -1, 1, ans_bisection, title_bisection)

        ans_newton = newton_all(i, option=True)
        title_newton = f'Legendre {i} newton method'
        visualization(i, -1, 1, ans_newton, title_newton)

if __name__ == "__main__":
    n = 5
    k = 3
    x_legendre = [[-1, -0.75], [-0.75, -0.3], [-0.3, 0.25], [0.25, 0.75], [0.75, 1.0]]
    x_newton = [-1, -0.5, 0.1, 0.5, 1]

    main1(n, k)

    main2(x_legendre)

    main3(x_newton)

    main4(6)