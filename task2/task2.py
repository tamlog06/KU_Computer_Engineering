import numpy as np
import matplotlib.pyplot as plt
from functions import *
from visualize import *

A = np.zeros((9, 9))
for i in range(9):
    for j in range(9):
        A[i][j] = 1/(i+j+1)

def main1():
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

def main2():
    n = A.shape[0]
    alpha = np.ones(n, dtype=float)
    b = dot(A, alpha)

    x, jacobi_error, jacobi_true_error = Jacobi(np.copy(A), b, option=True)
    print(f'Jacovi count: {len(jacobi_error)-1}')
    print(f'Jacovi x: \n{x}')
    visualization_norm(jacobi_error, 'Jacobi Residual norm(2.2.2)', log=True)
    visualization_norm(jacobi_true_error, 'Jacobi true error norm(2.2.2)', log=True)

    x, gauss_error, gauss_true_error = Gauss_Seidel(np.copy(A), b, option=True)
    print(f'Gauss_Seidel count: {len(gauss_error)-1}')
    print(f'Gauss_Seidel x: \n{x}')
    visualization_norm(gauss_error, 'Gauss Seidel Residual norm(2.2.2)', 1)
    visualization_norm(gauss_true_error, 'Gauss Seidel true error norm(2.2.2)', 1)

    x, sor_error, sor_true_error = SOR(np.copy(A), b, 1.85, option=True)
    print(f'SOR count: {len(sor_error)-1}')
    print(f'SOR x: \n{x}')
    visualization_norm(sor_error, 'SOR Residual norm(2.2.2)', 1)
    visualization_norm(sor_true_error, 'SOR true error norm(2.2.2)', 1)


    print(f'last Residual norm of  Jacobi: {jacobi_error[-1]}')
    print(f'last true error norm of  Jacobi: {jacobi_true_error[-1]}')
    print(f'last Residual norm of  Gauss Seidel: {gauss_error[-1]}')
    print(f'last true error norm of  Gauss Seidel: {gauss_true_error[-1]}')
    print(f'last Residual norm of  SOR: {sor_error[-1]}')
    print(f'last true error norm of  SOR: {sor_true_error[-1]}')

def main3():
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
