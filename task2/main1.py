import numpy as np
import matplotlib.pyplot as plt
from functions import *
from visualize import *

A = np.array([[12, 1, 5, 1, 1, 2,-4, 1, 2],
              [ 1,16,-1,-4,-5,-2, 1, 2, 3],
              [ 5,-1,15,-5, 3, 1,-2, 1,-4],
              [ 1,-4,-5,10, 3,-3,-1, 4, 1],
              [ 1,-5, 3, 3,11,-1, 4, 1, 1],
              [ 2,-2, 1,-3,-1,15,-5, 2, 5],
              [-4, 1,-2,-1, 4,-5,15, 4,-4],
              [ 1, 2, 1, 4, 1, 2, 4,11,-1],
              [ 2, 3,-4, 1, 1, 5,-4,-1,15]])

def main1():
    n = A.shape[0]
    alpha = np.ones(n)
    b = A@alpha
    print(b)

def main2():
    L, U = LU(A.copy())
    print(f' L: \n{L}\n')
    print(f' U: \n{U}\n')
    print(f'LU: \n{L@U}\n')
    x = direct(A.copy(), L, U)
    print(f'direct x: \n{x}\n')
    for i in x:
        print(f'direct error: {float(1-i)}')

def main3():
    x, jacobi_err = Jacobi(A.copy())
    print(f'Jacovi count: {len(jacobi_err)}')
    print(f'Jacovi x: \n{x}')
    e = 1.0 - x
    print(f'Jacovi error: \n{norm(e)}')
    print()

    x, gauss_err = Gauss_Seidel(A.copy())
    print(f'Gauss_Seidel count: {len(gauss_err)}')
    print(f'Gauss_Seidel x: \n{x}')
    e = 1.0 - x
    print(f'Gauss_Seidel error: \n{norm(e)}')

    x, sor_err = SOR(A.copy(), 1.8549999999999998)
    print(f'SOR count: {len(sor_err)}')
    print(f'SOR x: \n{x}')
    e = 1.0 - x
    print(f'SOR error: \n{norm(e)}')

    visualization_convergence(jacobi_err, gauss_err, sor_err)

def main4():
    x, jacobi_error, jacobi_true_error = Jacobi(A.copy(), option=True)
    x, gauss_error, gauss_true_error = Gauss_Seidel(A.copy(), option=True)
    x, sor_error, sor_true_error = SOR(A.copy(), 1.8549999999999998, option=True)

    visualization_norm_difference(jacobi_error, jacobi_true_error, 100, 'Jacobi')
    visualization_norm_difference(gauss_error, gauss_true_error, 50, 'Gauss_Seidel')
    visualization_norm_difference(sor_error, sor_true_error, 1, 'SOR')

    print(f'last difference of Jacobi: {jacobi_true_error[-1] - jacobi_error[-1]}')
    print(f'last difference of Gauss_Seidel: {gauss_true_error[-1] - gauss_error[-1]}')
    print(f'last difference of SOR: {sor_true_error[-1] - sor_error[-1]}')

def main5():
    x, k = CG(A.copy())
    print(x)
    print(k)

if __name__ == '__main__':
    # main1()
    main2()
    # main3()
    # main4()
    # main5()

    # count_max = 1e4
    # w_max = 2
    # for w in np.arange(0.01, 2.0, 0.005):
    #     x, count = SOR(A.copy(), w)
    #     if count < count_max:
    #         count_max = count
    #         w_max = w
    #     print(count)
    #     print(w)
    # print(count_max, w_max)
