import numpy as np

''' 数学演算の関数 '''

# 行列arrとベクトルxの積
def dot(arr, x):
    n = x.shape[0]
    result = np.copy(x)
    for i in range(n):
        result[i] = sum(arr[i][j]*x[j] for j in range(n))
    return result

# 行列AとBの積: A@B
def multiply(A, B):
    r = A.shape[0]
    c = B.shape[1]
    C = np.zeros((r, c))
    for i in range(r):
        for j in range(c):
            C[i][j] = inner_product(A[i], B.T[j])
    return C

# arrのノルム
def norm(arr):
    result = 0
    n = arr.shape[0]
    for i in range(n):
        result += arr[i]**2
    result = result ** (1/2)
    return result 

# 内積
def inner_product(a, b):
    return np.sum(np.multiply(a,b))

# 下三角行列の逆行列
def linalg_lower(arr):
    n = arr.shape[0]
    L = np.zeros_like(arr)
    for i in range(n):
        L[i][i] = 1/arr[i][i]
        for j in range(i+1, n):
            L[j][i] = - np.sum(arr[j][k]*L[k][i] for k in range(i-1, j)) / arr[j][j]
    return L

''' 連立方程式を解く関数 '''

# 直接法での計算をするためのLU分解
def LU(arr):
    n = arr.shape[0]
    # LとUをゼロ行列で初期化
    L = np.zeros((n, n), dtype=float)
    U = np.zeros((n, n), dtype=float)
    # Uの対角成分を１にする
    for i in range(n):
        U[i][i] = 1

    # LとUを計算
    for j in range(n):
        for i in range(j-1, n):
            L[i][j] = arr[i][j] - sum(L[i][k]*U[k][j] for k in range(j))
        for i in range(j-1, n):
            U[j][i] = (arr[j][i] - sum(L[j][k]*U[k][i] for k in range(j))) / L[j][j]
    return L, U

# メモリ抑える
# def LU(arr):
#     n = arr.shape[0]

#     for i in range(n):
#         for j in range(i + 1, n):
#             arr[j][i] /= arr[i][i]
#             for k in range(i + 1, n):
#                 arr[j][k] -= arr[j][i] * arr[i][k]
#     return arr, 0

# 直接法
def direct(arr, b, L, U):
    n = arr.shape[0]
    # Ly=b で前進代入してyを求める
    y = np.zeros(n, dtype=float)
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j]*y[j] for j in range(i))) / L[i][i]
    
    # Ux=y で後進代入してxを求める
    x = np.zeros(n, dtype=float)
    for i in range(1, n+1):
        x[n-i] = y[n-i] - sum(U[n-i][j]*x[j] for j in range(n-i, n))
    return x

# 反復法の計算に必要なD, L, Uの分解した値を返す
def devide_DLU(arr):
    n = arr.shape[0]
    D = np.zeros((n, n), dtype=float)
    L = np.zeros((n, n), dtype=float)
    U = np.zeros((n, n), dtype=float)

    for i in range(n):
        D[i][i] = arr[i][i]
    
    for i in range(1, n):
        for j in range(i):
            L[i][j] = arr[i][j]
            U[j][i] = arr[j][i]
    return D, L, U

# Jacobi法
def Jacobi(arr, b, option=False):
    n = arr.shape[0]
    x = np.zeros(n, dtype=float)
    r = b - dot(arr, x)
    count = 0
    error = [norm(r)]

    if option:
        true_error = [norm(1-x)]
    while error[-1] > 1e-10 and count < 1e4:
        temp = np.copy(x)
        for i in range(n):
            temp[i] = (b[i] - sum(arr[i][j]*x[j] for j in range(n)) + arr[i][i]*x[i]) / (arr[i][i]+ 1e-20)
        x = temp
        r = b - dot(arr, x)
        count += 1
        error.append(norm(r))
        if option:
            true_error.append(norm(1-x))
    if option:
        return x, error, true_error
    else:
        return x, error

# Gauss_Seidel法
def Gauss_Seidel(arr, b, option=False):
    n = arr.shape[0]
    x = np.zeros(n, dtype=float)
    r = b - dot(arr, x)
    count = 0
    error = [norm(r)]

    if option:
        true_error = [norm(1-x)]
    while error[-1] > 1e-10 and count < 1e4:
        for i in range(n):
            x[i] = (b[i] - np.sum(arr[i][j]*x[j] for j in range(n)) + arr[i][i]*x[i]) / (arr[i][i]+ 1e-20)
        r = b - dot(arr, x)
        count += 1
        error.append(norm(r))
        if option:
            true_error.append(norm(1-x))
    if option:
        return x, error, true_error
    else:
        return x, error

# SOR法
def SOR(arr, b, w, option=False):
    n = arr.shape[0]
    x = np.zeros(n, dtype=float)
    r = b - dot(arr, x)
    count = 0
    error = [norm(r)]

    if option:
        true_error = [norm(1-x)]
    while error[-1] > 1e-10 and count < 1e4:
        for i in range(n):
            x[i] = w*(b[i] - np.sum(arr[i][j]*x[j] for j in range(n)) + arr[i][i]*x[i]) / (arr[i][i]+ 1e-20) + (1-w)*x[i]
        r = b - dot(arr, x)
        count += 1
        error.append(norm(r))
        if option:
            true_error.append(norm(1-x))
    if option:
        return x, error, true_error
    else:
        return x, error

# 共役勾配法(Conjugate_Gradient -> CG)
def CG(arr, b):
    n = arr.shape[0]
    x = np.zeros(n, dtype=float)
    r = [b - dot(arr, x)]
    p = r[0]
    error = [norm(r[-1])]
    true_error = [norm(1-x)]
    k = 0
    while error[-1] > 1e-10 and k < 1e4:
        if k == 0:
            beta = 0
        else:
            beta = inner_product(r[k], r[k]) / inner_product(r[k-1], r[k-1])
        p = r[k] + beta*p
        alpha = inner_product(r[k], r[k]) / inner_product(p, dot(arr, p))
        x = x + alpha * p
        r.append(r[k] - alpha * dot(arr, p))
        k += 1
        error.append(norm(r[-1]))
        true_error.append(norm(1-x))
    return x, error, true_error

##### 連立方程式を解く関数 #####

# レイリー商
def R(x, y):
    upper = 0
    n = y.shape[0]
    for i in range(n):
        upper += x[i]*y[i]
    return upper / norm(y)**2

# 絶対値最大の固有値
def max_eigenval(arr):
    n = arr.shape[0]
    
    # 初期値をノルムが１の適当なものにする
    x = np.zeros(n, dtype=float)
    x[0] = 1.0
    k = 0
    delta = 1e4
    eig_before = 1e4
    while abs(delta) >= 1e-15 and k < 1e4 :
        x = dot(arr, x)
        x = x / norm(x)

        eig = R(dot(arr, x), x)
        delta = (eig - eig_before) / (eig_before + 1e-20)
        k += 1
        eig_before = eig
    return eig, x

# 絶対値最小の固有値
def min_eigenval(arr):
    n = arr.shape[0]

    # 初期値をノルムが１の適当なものにする
    x = np.zeros(n, dtype=float)
    x[0] = 1.0
    k = 0
    delta = 1e4
    eig_before = 1e4
    L, U = LU(arr)
    while abs(delta) >= 1e-15 and k < 1e4:
        x = direct(arr, x, L, U)
        x = x / norm(x)

        temp = R(direct(arr, x, L, U), x)
        eig = 1 / temp

        delta = (eig - eig_before) / (eig_before + 1e-9)
        k += 1
        eig_before = eig
    return eig, x

# グラム・シュミット直交化関数
def schmidt(arr):
    # 渡した配列の列数(基底に含まれるベクトル数)
    k = arr.shape[1]

    # 1列目のベクトルを選択
    u = arr.T[0]

    # uを正規化
    q = u / norm(u)
    q = np.reshape(q, [-1, 1])

    # シュミットの直交化
    for j in range(1, k):
        u = arr.T[j] - sum(inner_product(q.T[i], arr.T[j])*q.T[i] for i in range(j))
        qi = u / norm(u)
        qi = qi.reshape(-1, 1)
        q = np.hstack([q, qi])
    return q

# QR法
def QR(arr):
    n = arr.shape[0]
    Q = schmidt(arr)
    R = multiply(Q.T, arr)
    k = 0
    error = 1e9
    while abs(error) >= 1e-10 and k <= 1e4:
        arr = multiply(R, Q)
        Q = schmidt(arr)
        R = multiply(Q.T, arr)
        arr = multiply(Q, R)
        D, L, U = devide_DLU(arr)
        err = L + U
        error = np.average(err)
        k += 1
    result = np.array([D[i][i] for i in range(n)])
    print(k, error)
    return result
