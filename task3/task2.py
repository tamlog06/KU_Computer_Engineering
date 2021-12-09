import math
from math import cos, sin
from visualize import *
from functions import *

global N, k, m, Td, h
N = 100
k = 3
m = 4
Td = 50
h = 0.5

def main(boundary=True):
    t = 0
    x = np.zeros(N, dtype='float')
    v = np.zeros(N, dtype='float')
    x[0] = 0.1
    x_N = [x.copy()]
    T = N*math.sqrt(m/k)
    while t <= T*3/2:
        t += h
        x, v = heun2(x, v, t, boundary)
        x_N.append(x.copy())
    # x_axis = np.array([i for i in range((int(T)+1)*2+10)])
    # visualize(x_axis, x_N, 'time', 'x_N-1', 'x_N-1', x_ticks=False, ylim=[-100, 100])

    x_axis = [i for i in range(N)]
    anime(x_axis, x_N)


if __name__ == '__main__':
    main(False)