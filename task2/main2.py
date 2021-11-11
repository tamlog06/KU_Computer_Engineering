import numpy as np
import matplotlib.pyplot as plt
from functions import *
from visualize import *

A = np.zeros((9, 9))
for i in range(9):
    for j in range(9):
        A[i][j] = 1/(i+j+1)

def main1():
    L, U = LU(A.copy())
    print(f' L: \n{L}\n')
    print(f' U: \n{U}\n')
    print(f'LU: \n{L@U}\n')
    x = direct(A.copy(), L, U)
    print(f'direct x: \n{x}\n')
    for i in x:
        print(f'direct error: {float(1-i)}')

def main2():
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

def main3():
    x, k = CG(A.copy())
    print(x)
    print(k)

if __name__ == '__main__':
    # main1()
    # main2()
    main3()