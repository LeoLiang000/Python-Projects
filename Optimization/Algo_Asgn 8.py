import time
import random
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


# cost functions (f1, f2) from week 4
def f1(x, y):
    return 10 * (x - 9) ** 4 + 7 * (y - 0) ** 2


def f2(x, y):
    return sp.Max(x - 9, 0) + 7 * sp.Abs(y)


def df1(x, y):
    grad_x = 40 * (x - 9) ** 3
    grad_y = 14 * y
    return grad_x, grad_y


def df2(x, y):
    grad_x = 1 if x > 9 else 0

    if y > 0:
        grad_y = 7
    elif y < 0:
        grad_y = -7
    else:
        grad_y = 0

    return grad_x, grad_y


# gradDescent from week 4
def gradDescent(f, df, x0, y0, alpha=0.01, iterations=100):
    x, y = x0, y0
    X = np.zeros((iterations, 2))
    F = np.zeros(iterations)
    X[0], F[0] = (x, y), f(x, y)

    for i in range(iterations):
        if i != 0:
            grad_x, grad_y = df(x, y)
            x -= grad_x * alpha
            y -= grad_y * alpha
            X[i], F[i] = (x, y), f(x, y)

    return X, F


def globalRandomSearch(n, N, limits_ps, f):
    lowest_cost = float('inf')
    cost_values = list()
    running_time = list()
    t0 = time.time()

    for i in range(N):
        tmp_params = [random.uniform(li, ui) for li, ui in limits_ps]
        tmp_cost = f(*tmp_params)

        running_time.append(time.time() - t0)
        cost_values.append(tmp_cost)

        if tmp_cost < lowest_cost:
            lowest_cost = tmp_cost
            desired_params = tmp_params

    return lowest_cost, desired_params, cost_values, running_time


def plot_(costs1, label1='', costs2='', label2='', xlabel='Iteration', ylabel='Cost Function Values', title='', multiPlot=False,
          iterations=None, isF1=True):
    if multiPlot:
        plt.plot(iterations, costs1, label=label1)
        plt.plot(iterations, costs2, label=label2)
    else:
        plt.plot(costs1, label='Cost Function Values')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale('log')
    plt.legend()
    plt.show()


def main(isF1=True):
    n = 2  # number of parameters

    if isF1:  # f1: 10*(x-9)^4+7*(y-0)^2
        title = f'Values of Cost Function: 10*(x-9)^4+7*(y-0)^2 vs Running Time'
        N = 10000  # number of samples
        limit_p1 = (-1, 10)  # boundary of parameter 1
        limit_p2 = (-10, 10)  # boundary of parameter 2
        cost1, params1, costs_grs, times1 = globalRandomSearch(n, N, [limit_p1, limit_p2], f1)
        x, costs_gd = gradDescent(f1, df1, x0=2, y0=1, alpha=0.0001, iterations=N)

    else:  # f2: Max(x-9,0)+7*|y-0|
        title = f'Values of Cost Function: Max(x-9,0)+7*|y-0| vs Running Time'
        N = 10000  # number of samples
        limit_p1 = (-1, 1)  # boundary of parameter 1
        limit_p2 = (-5, 10)  # boundary of parameter 2
        cost2, params2, costs_grs, times2 = globalRandomSearch(n, N, [limit_p1, limit_p2], f2)
        x2, costs_gd = gradDescent(f2, df2, x0=0.5, y0=1, alpha=0.0001, iterations=N)

    # plot comparison between global random search and gradient descent
    plot_(np.minimum.accumulate(costs_grs), label1='Global Random Search',
          costs2=np.minimum.accumulate(costs_gd), label2='Gradient Descent',
          multiPlot=True, iterations=np.arange(N), title=title)


if __name__ == '__main__':
    main(isF1=False)
