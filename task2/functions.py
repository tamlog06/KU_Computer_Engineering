import numpy as np

def norm(arr):
    return (arr @ arr)**(1/2)

def inner_product(a, b):
    return np.sum(np.multiply(a,b))

# クラウト法での計算
def LU(A):
    n = A.shape[0]
    np.set_printoptions(precision=3, suppress=True)
    # LとUをゼロ行列で初期化
    L = np.zeros((n, n), dtype=float)
    U = np.zeros((n, n), dtype=float)
    # Uの対角成分を１にする
    for i in range(n):
        U[i][i] = 1
    
    for k in range(n):
        for i in range(k-1, n):
            L[i][k] = A[i][k] - sum(L[i][j]*U[j][k] for j in range(k))
        for j in range(k-1, n):
            U[k][j] = (A[k][j] - sum(L[k][i]*U[i][j] for i in range(k))) / L[k][k]
    return L, U

# 直接法
def direct(A, L, U):
    n = A.shape[0]
    alpha = np.ones(n)
    b = A@alpha
    # Ly=b からbを求める
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j]*y[j] for j in range(i))) / L[i][i]
    
    # Ux=y からxを求める
    x = np.zeros(n)
    for i in range(1, n+1):
        x[n-i] = y[n-i] - sum(U[n-i][j]*x[j] for j in range(n-i, n))
    return x

# 反復法の計算に必要なD, L, Uの分解した値を返す
def devide_DLU(A):
    n = A.shape[0]
    D = np.zeros((n, n))
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        D[i][i] = A[i][i]
    
    for i in range(1, n):
        for j in range(i):
            L[i][j] = A[i][j]
            U[j][i] = A[j][i]
    return D, L, U

# Jacobi法
def Jacobi(A, option=False):
    D, L, U = devide_DLU(A)
    M = D
    N = -(L+U)
    return iterative(A, M, N, option)

# Gauss_Seidel法
def Gauss_Seidel(A, option=False):
    D, L, U = devide_DLU(A)
    M = D+L
    N = -U
    return iterative(A, M, N, option)

# SOR法
def SOR(A, w, option=False):
    D, L, U = devide_DLU(A)
    M = (D+w*L)/w
    N = ((1-w)*D - w*U) / w
    return iterative(A, M, N, option)

# 反復法
def iterative(A, M, N, option=False):
    n = A.shape[0]
    alpha = np.ones(n)
    b = A@alpha

    B = np.linalg.inv(M)@N
    c = np.linalg.inv(M)@b
    x = np.zeros(n)
    r = b - A@x
    count = 0
    error = [norm(r)]
    if option:
        true_error = [norm(1-x)]
    while error[-1] > 1e-10 and count < 1e4:
        x = B@x + c
        r = b - A@x
        count += 1
        error.append(norm(r))
        if option:
            true_error.append(norm(1-x))
    if option:
        return x, error, true_error
    else:
        return x, error

# 共役勾配法(Conjugate_Gradient -> CG)
def CG(A):
    n = A.shape[0]
    alpha = np.ones(n)
    b = A@alpha
    x = np.zeros(n)
    r = [b - A@x]
    I = np.eye(n)
    p = r[0]
    error = norm(r[-1])
    k = 0
    while error > 1e-10 and k < 1e4:
        if k == 0:
            beta = 0
        else:
            beta = inner_product(r[k], r[k]) / inner_product(r[k-1], r[k-1])
        p = r[k] + beta*p
        alpha = inner_product(r[k], r[k]) / inner_product(p, A@p)
        x = x + alpha * p
        r.append(r[k] - alpha * (A@p))
        k += 1
        error = norm(r[-1])
        print(x)
    return x, k