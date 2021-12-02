import numpy as np
import matplotlib.pyplot as plt
from functions import *
from visualize import *

A = np.array([
            [12, 1, 5, 1, 1, 2,-4, 1, 2],
            [ 1,16,-1,-4,-5,-2, 1, 2, 3],
            [ 5,-1,15,-5, 3, 1,-2, 1,-4],
            [ 1,-4,-5,10, 3,-3,-1, 4, 1],
            [ 1,-5, 3, 3,11,-1, 4, 1, 1],
            [ 2,-2, 1,-3,-1,15,-5, 2, 5],
            [-4, 1,-2,-1, 4,-5,15, 4,-4],
            [ 1, 2, 1, 4, 1, 2, 4,11,-1],
            [ 2, 3,-4, 1, 1, 5,-4,-1,15]
            ])

def main1():
    n = A.shape[0]
    alpha = np.ones(n, dtype=float)
    b = dot(A, alpha)
    print(b)

def main2():
    n = A.shape[0]
    alpha = np.ones(n, dtype=float)
    b = dot(A, alpha)

    L, U = LU(A.copy())
    print(f' L: \n{L}\n')
    print(f' U: \n{U}\n')
    print(f'LU: \n{multiply(L, U)}\n')
    x = direct(np.copy(A), b, L, U)
    print(f'direct x: \n{x}\n')
    error = np.abs(np.array([1-x]))
    average_error = np.average(error)
    print(f'average direct error: {average_error}')

def main3():
    n = A.shape[0]
    alpha = np.ones(n, dtype=float)
    b = dot(A, alpha)

    x, jacobi_err = Jacobi(np.copy(A), b)
    print(f'Jacovi count: {len(jacobi_err)-1}')
    print(f'Jacovi x: \n{x}')
    e = 1.0 - x
    e = e[1:]
    print(f'Jacovi error: \n{norm(e)}')
    print()

    x, gauss_err = Gauss_Seidel(np.copy(A), b)
    print(f'Gauss_Seidel count: {len(gauss_err)-1}')
    print(f'Gauss_Seidel x: \n{x}')
    e = 1.0 - x
    print(f'Gauss_Seidel error: \n{norm(e)}')

    x, sor_err = SOR(np.copy(A), b, 1.8512999999999944)
    print(f'SOR count: {len(sor_err)-1}')
    print(f'SOR x: \n{x}')
    e = 1.0 - x
    print(f'SOR error: \n{norm(e)}')

    visualization_convergence(jacobi_err, gauss_err, sor_err, 'Residual norm(2.1.3)')

def main4():
    n = A.shape[0]
    alpha = np.ones(n, dtype=float)
    b = dot(A, alpha)

    x, jacobi_error, jacobi_true_error = Jacobi(np.copy(A), b, option=True)
    visualization_norm(jacobi_error, 'Jacobi Residual norm(2.1.4)', 1)
    visualization_norm(jacobi_true_error, 'Jacobi true error norm(2.1.4)', 1)

    x, gauss_error, gauss_true_error = Gauss_Seidel(np.copy(A), b, option=True)
    visualization_norm(gauss_error, 'Gauss Seidel Residual norm(2.1.4)', 1)
    visualization_norm(gauss_true_error, 'Gauss Seidel true error norm(2.1.4)', 1)

    x, sor_error, sor_true_error = SOR(np.copy(A), b, 1.85, option=True)
    visualization_norm(sor_error, 'SOR Residual norm(2.1.4)', 10)
    visualization_norm(sor_true_error, 'SOR true error norm(2.1.4)', 10)


    print(f'last Residual norm of  Jacobi: {jacobi_error[-1]}')
    print(f'last true error norm of  Jacobi: {jacobi_true_error[-1]}')
    print(f'last Residual norm of  Gauss Seidel: {gauss_error[-1]}')
    print(f'last true error norm of  Gauss Seidel: {gauss_true_error[-1]}')
    print(f'last Residual norm of  SOR: {sor_error[-1]}')
    print(f'last true error norm of  SOR: {sor_true_error[-1]}')

def main5():
    n = A.shape[0]
    alpha = np.ones(n, dtype=float)
    b = dot(A, alpha)

    x, error, true_error = CG(np.copy(A), b)
    print(f'CG count: {len(error)-1}')
    print(f'CG x: \n{x}')
    print(f'CG Residual norm: \n{error[1:]}')
    print(f'CG true error norm: \n{true_error[1:]}')

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf,linewidth=np.inf)
    np.set_printoptions(formatter={'float': '{:.4e}'.format})
    main1()
    main2()
    main3()
    main4()
    main5()
