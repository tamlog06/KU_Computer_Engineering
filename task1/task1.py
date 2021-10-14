import matplotlib.pyplot as plt
import numpy as np

# 二項係数
def comb(n: float, k: int):
    if k < 0:
        return 0
    if (n == k or k == 0):
        return 1
    
    result = 1
    for i in range(k):
        result *= n-i
    for i in range(1, k+1):
        result /= i
    return result


# return a
def Legendre_a(n: float):
    a = []
    for k in range(n+1):
        a.append(2**n * comb(n, k) * comb((n+k-1)/2, n))
    return a

# return P
def Legendre_P(n: float, x: float):
    P = 0
    a = Legendre_a(n)
    for k in range(n+1):
        P += a[k] * x**k
    return P


#二分法（方程式の関数項、探索区間の左端、探索区間の右端、誤差範囲、最大反復回数）
def bisection(n, x_min, x_max, error=1e-9, max_loop=100):
    #初期値を表示
    num_calc = 0  #計算回数
    print("{:3d}:  {:.15f} <= x <= {:.15f}".format(num_calc, x_min, x_max))

    #中間値の定理の条件を満たすか調べる
    if(0.0 < Legendre_P(n, x_min)*Legendre_P(n, x_max)):
        print("error: Section definition is invalid (0.0 < P(n, x_min)*P(n, x_max)).")
        quit()

    #ずっと繰り返す
    while(True):
        #新たな中間値の計算
        x_mid = (x_max +x_min)/2.0

        #探索区間を更新
        if (0.0 < Legendre_P(n, x_mid)*Legendre_P(n, x_max)):  #中間と右端の値が同じの時
            x_max = x_mid  #右端を更新
        else:  #中間と左端の値が同じの時
            x_min = x_mid  #左端を更新

        #結果を表示
        num_calc += 1  #計算回数を数える
        print("{:3d}:  {:.15f} <= x <= {:.15f}".format(num_calc, x_min, x_max))

        #「誤差範囲が一定値以下」または「計算回数が一定値以上」ならば終了
        if x_max-x_min <= error:
            break
        elif  max_loop <= num_calc:
            return False

    #最終的に得られた解
    print("x = {:.15f}".format(x_mid))
    print(num_calc)

    return x_mid

# 二分法で全ての解を出す
def bisection_all(n, x: list):
    result = []
    for x_min, x_max in x:
        result.append(bisection(n, x_min, x_max))
    return result

#Newton法（方程式の関数項、探索の開始点、微小量、誤差範囲、最大反復回数）
def newton(n, x0, eps=1e-9, error=1e-9, max_loop=100):
    num_calc = 0  #計算回数
    print("{:3d}:  x = {:.15f}".format(num_calc, x0))

    #ずっと繰り返す
    while True:
        #中心差分による微分値
        P_df = (Legendre_P(n, x0+eps) - Legendre_P(n, x0-eps)) / (2*eps)
        # func_df = (func_f(x0 +eps) -func_f(x0 -eps))/(2*eps)
        # if(abs(func_df) <= eps):  #傾きが0に近ければ止める
        if abs(P_df) <= eps:
            print("error: abs(P_df) is too small (<=", eps, ").")
            quit()

        #次の解を計算
        x1 = x0 - Legendre_P(n, x0) / P_df

        num_calc += 1  #計算回数を数える
        print("{:3d}:  x = {:.15f}".format(num_calc, x0))

        #「誤差範囲が一定値以下」または「計算回数が一定値以上」ならば終了
        if(abs(x1-x0)<=error or max_loop<=num_calc):
            break

        #解を更新
        x0 = x1

    #最終的に得られた解
    print("x = {:.15f}".format(x0))

    return x0

def newton_all(n, x: list):
    result = []
    for x0 in x:
        result.append(newton(n, x0))
    return set(result)

#可視化（方程式の関数項、グラフ左端、グラフ右端、方程式の解）
def visualization(n, x_min, x_max, x_solved):
    plt.xlabel("$x$")  #x軸の名前
    plt.ylabel("$f(x)$")  #y軸の名前
    plt.grid()  #点線の目盛りを表示
    plt.axhline(0, color='#000000')  #f(x)=0の線

    #関数
    exact_x = np.arange(x_min,x_max, (x_max-x_min)/500.0)
    exact_y = Legendre_P(n, exact_x)

    plt.plot(exact_x,exact_y, label="$f(x)$", color='#ff0000')  #関数を折線グラフで表示

    for x in x_solved:
        plt.scatter(x,0.0)  #数値解を点グラフで表示
        plt.text(x,0.0, "$x$ = {:.9f}".format(x), va='bottom', color='#0000ff')
    plt.show()  #グラフを表示

if __name__ == "__main__":
    n = 5
    k = 3
    x_legendre = [[-1, -0.75], [-0.75, -0.3], [-0.3, 0.25], [0.25, 0.75], [0.75, 1.0]]
    x_newton = [-1, -0.5, 0, 0.5, 0.9]

    print(comb(n, k))
    print(Legendre_a(5))
    print(Legendre_a(10))
    # print(Legendre_P(n, -0.99))

    result_bisection = bisection_all(n, x_legendre)
    visualization(n, -1, 1, result_bisection)
    print(result_bisection)

    result_newton = newton_all(n, x_newton)
    visualization(n, -1, 1, result_newton)
    print(result_newton)