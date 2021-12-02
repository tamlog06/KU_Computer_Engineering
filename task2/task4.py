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

def main():
    eig1 = QR(A1)
    eig1 = np.sort(eig1)
    print(f'eigenvalue A1: \n{eig1}')
    print()

    eig2 = QR(A2)
    eig2 = np.sort(eig2)
    print(f'eigenvalue A2: \n{eig2}')


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf,linewidth=np.inf)
    np.set_printoptions(formatter={'float': '{:.4e}'.format})
    main()