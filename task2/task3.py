import numpy as np
import matplotlib.pyplot as plt
from functions import *
from visualize import *

A1 = np.array([
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

A2 = np.zeros((9, 9))
for i in range(9):
    for j in range(9):
        A2[i][j] = 1/(i+j+1)

def main1():
    max_eig_1, max_eig1_vector = max_eigenval(A1)
    min_eig_1, min_eig1_vector = min_eigenval(A1)
    max_eig_2, max_eig2_vector = max_eigenval(A2)
    min_eig_2, min_eig2_vector = min_eigenval(A2)
    
    print('A1')
    print(f'max_eigen_value: {max_eig_1}\neigen_vector: {max_eig1_vector}')
    print(f'min_eigen_value: {min_eig_1}\neigen_vector: {min_eig1_vector}')
    print()

    print('A2')
    print(f'max_eigen_value: {max_eig_2}\neigen_vector: {max_eig2_vector}')
    print(f'min_eigen_value: {min_eig_2}\neigen_vector: {min_eig2_vector}')

def main2():
    max_eig_1, max_eig1_vector = max_eigenval(A1)
    min_eig_1, min_eig1_vector = min_eigenval(A1)
    max_eig_2, max_eig2_vector = max_eigenval(A2)
    min_eig_2, min_eig2_vector = min_eigenval(A2)

    print('A1')
    print(f'Condition number: {abs(max_eig_1 / min_eig_1):.4e}')
    print()
    print('A2')
    print(f'Condition number: {abs(max_eig_2 / min_eig_2):.4e}')

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf,linewidth=np.inf)
    np.set_printoptions(formatter={'float': '{:.4e}'.format})
    main1()
    main2()
