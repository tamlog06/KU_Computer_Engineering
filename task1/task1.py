import matplotlib.pyplot as plt
import numpy as np
import math

# 二項係数
def comb(n: float, k: int):
    if k < 0:
        print('error: invalid number. k < 0')
        return False
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


#二分法（n、探索区間の左端、探索区間の右端、誤差範囲、最大反復回数、計算過程も返した場合はoptionをTrueにする）
def bisection(n, x_min, x_max, error=1e-9, max_loop=100, option=False):
    # 計算途中の値と反復数を含むリスト
    middle_value = []
    #計算回数
    num_calc = 0

    #中間値の定理の条件を満たすか調べる
    if(0 < Legendre_P(n, x_min)*Legendre_P(n, x_max)):
        print("error: Section definition is invalid (0.0 < P(n, x_min)*P(n, x_max)).")
        return False

    while True:
        #新たな中間値の計算
        x_mid = (x_max + x_min) / 2

        #探索区間を更新
        if (0 < Legendre_P(n, x_mid)*Legendre_P(n, x_max)):
            x_max = x_mid
        else:
            x_min = x_mid

        # 計算回数と、計算途中の値を更新
        num_calc += 1
        middle_value.append([x_mid, num_calc])

        x_mid = (x_max + x_min) / 2

        #「絶対誤差が一定値以下」または「計算回数が一定値以上」ならば終了
        if abs(Legendre_P(n, x_mid)) <= error:
            break
        elif  num_calc >= max_loop:
            print(f'error: too complex to caluculate within {max_loop} times')
            return False
    

    if abs(x_mid) < 1e-9:
        x_mid = 0.0

    # 課題1.3で計算回数を出すために、計算途中の値を出力する場合はmiddle_valueも出力
    if option:
        return x_mid, middle_value
    else:
        return x_mid

# 二分法で全ての解を出す
# option が True なら自動で初期区間を出して計算する（課題1.4）
def bisection_all(n: int, x=[], error=1e-9, option=False):
    if option:
        # nが1の場合の値は既知とする
        if n == 1:
            return [0]
        else:
            # nの解は、[-1, 1]のうちのn-1の解の区間に１つづつ存在するので、その区間を再帰的に計算
            previous_ans = bisection_all(n-1, option=True)
            previous_ans.append(-1)
            previous_ans.append(1)
            previous_ans.sort()
            x = []
            for i in range(len(previous_ans)-1):
                x.append([previous_ans[i], previous_ans[i+1]])
            return bisection_all(n, x, error)
    else:
        # 与えられた区間の中の解を全て出す
        result = []
        if not x:
            print('list is not added')
            return False
        for x_min, x_max in x:
            ans = bisection(n, x_min, x_max)
            if ans is not False:
                result.append(ans)
        return result

#ルジャンドル多項式の微分
def diff_Legendre(n: int, x):
    # オーバーフロー対策
    if (x == 1) or (x == -1):
            x += 0.001
    return n*(Legendre_P(n-1, x) - x*Legendre_P(n,x))/(1-x**2)

# newton法で解を一つ出す
def newton(x0, n: int, error=1e-9, max_loop=100, option=False):
    # 計算途中の値
    middle_value = []
    num_calc = 0

    while True:
        x1 = x0 - Legendre_P(n, x0) / diff_Legendre(n, x0)
        # 近似解の修正量が誤差の範囲内であれば終了
        if abs(x1 - x0) < error:
            break
        x0 = x1

        num_calc += 1
        middle_value.append([x1, num_calc])

        if num_calc >= max_loop:
            print(f'error: too complex to caluculate within {max_loop} times')
            return False
    

    if abs(x1) < error:
        x1 = 0

    # 課題1.3で計算回数を出すために、計算途中の値を出力する場合はmiddle_valueも出力
    if option:
        return x1, middle_value
    else:
        return x1


# newton法で全ての解を出す
# option==Trueなら、初期近似解を自動で、Falseなら決め打ち
def newton_all(n: int, x=[], option=False, max_loop=100):
    result = []
    if option:
        for i in range(n):
            x = np.cos(((i+0.75)/(n+0.5))*np.pi)
            point = newton(x, n, max_loop=max_loop)
            result.append(point)
    else:
        if not x:
            print("list is not added")
            return False
        for x0 in x:
            ans = newton(x0, n, max_loop=max_loop)
            if ans is not False:
                result.append(ans)
    return result

#可視化（方程式の関数項、グラフ左端、グラフ右端、方程式の解）
def visualization(n: int, x_min, x_max, x_solved, title):
    plt.xlabel("$x$")  #x軸の名前
    plt.ylabel("$P(x)$")  #y軸の名前
    plt.grid()  #点線の目盛りを表示
    plt.axhline(0, color='k')  #f(x)=0の線
    plt.title(title)

    #関数
    exact_x = np.arange(x_min,x_max, (x_max-x_min)/500.0)
    exact_y = Legendre_P(n, exact_x)

    plt.plot(exact_x,exact_y, label="$P(x)$", color='r')  #関数を折線グラフで表示

    for x in x_solved:
        plt.scatter(x,0.0, c='b')  #数値解を点グラフで表示
        plt.text(x,0.0, "$x$ = {:.9f}".format(x), va='bottom', color='b')
    plt.show()  #グラフを表示

# 計算回数と誤差の関係を可視化
def visualization_convergence(n: int, middle_value, true_value, title):
    plt.xlabel("$loop$")
    plt.ylabel("$error$")
    plt.grid()
    plt.axhline(0, color='k')
    plt.title(title)

    for middle, loop in middle_value:
        plt.scatter(loop, abs(true_value - middle), c="b")

    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.set_yscale('log')  # メイン: y軸をlogスケールで描く
    
    plt.grid(which="both") # グリッド表示。"both"はxy軸両方にグリッドを描く。
    plt.show()

# 課題1.1
def main1():
    print(comb(5, 3))
    print(Legendre_a(5))
    print(Legendre_a(10))

# 課題1.2
# x_legendreは初期区間のリスト
def main2():
    x_legendre = [[-1, -0.75], [-0.75, -0.25], [-0.25, 0.25], [0.25, 0.75], [0.75, 1.0]]
    n = 5
    result_bisection = bisection_all(n, x_legendre)
    title = f'Legendre {n} bisection method'
    visualization(n, -1, 1, result_bisection, title)
    print(result_bisection)

# 課題1.3
# x_newtonは初期近似解のリスト
def main3():
    x_newton = [-1, -0.5, 0, 0.5, 1]
    n = 5
    result_newton = newton_all(n, x_newton)
    title = f'Legendre {n} newton method'
    visualization(n, -1, 1, result_newton, title)
    print(result_newton)

    # 反復の速さをグラフに描画
    # [-0.75, -0.25]の区間の解を出す時の反復の速さ
    result, middle_value = bisection(n, -0.75, -0.25, option=True)
    # 修正量の絶対値が10^-15以下として、真の値とする
    true_value = bisection(n, -0.75, -0.25, error=1e-15, max_loop=1000)
    title = 'bisection [-0.75, -0.25]'
    visualization_convergence(n, middle_value, true_value, title)
    
    # 二分法と同じ解を出す
    result, middle_value = newton(-0.7, n, option=True)
    # 修正量の絶対値が10^-15以下として、真の値とする
    true_value = newton(-0.7, n, error=1e-15, max_loop=1000)
    title = 'newton [-0.7]'
    visualization_convergence(n, middle_value, true_value, title)

# 課題1.4
def main4():
    # n=6~10 で可視化
    for i in range(6, 11):
        ans_bisection = bisection_all(i, option=True)
        title_bisection = f'Legendre {i} bisection method'
        visualization(i, -1, 1, ans_bisection, title_bisection)

        ans_newton = newton_all(i, option=True)
        title_newton = f'Legendre {i} newton method'
        visualization(i, -1, 1, ans_newton, title_newton)

def main5():
    ans_30 = newton(1, 30, max_loop=10000)
    ans_40 = newton(1, 40, max_loop=10000)
    print(ans_30)
    print(ans_40)

if __name__ == "__main__":
    # main1()

    main2()

    # main3()

    # main4()

    # main5()