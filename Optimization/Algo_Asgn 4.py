import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

x, y = sp.symbols('x y', real=True)
f1 = 10 * (x - 9) ** 4 + 7 * (y - 0) ** 2  # function: 10*(x-9)^4+7*(y-0)^2
f2 = sp.Max(x - 9, 0) + 7 * sp.Abs(y)  # function: Max(x-9,0)+7*|y-0|


def a_i_Polyak(f, iteration=100, epsilon=1e-3, x_val=10, y_val=5):
    """
    find the minimum of function value with ployak step size
    @param f: sympy representation of function f(x)
    @param iteration: iteration times
    @param epsilon: avoid division by zero
    @param x_val: x value
    @param y_val: y value
    @return:
    """
    df_dx = sp.diff(f, x)
    df_dy = sp.diff(f, y)

    f_star = 0  # minimum value of f(x)
    f_cur = f.subs({x: x_val, y: y_val})  # current value of f(x)

    for i in range(iteration):
        # calculate polyak step size alpha
        grad_x = df_dx.subs({x: x_val, y: y_val}).evalf()
        grad_y = df_dy.subs({x: x_val, y: y_val}).evalf()
        grad_sum = grad_x * grad_x + grad_y * grad_y  # sum all feature gradients at current point
        alpha = (f_cur - f_star) / (grad_sum + epsilon)

        # update x and y with polyak step size and move current f(x) to new point
        x_val -= alpha * grad_x
        y_val -= alpha * grad_y
        f_cur = f.subs({x: x_val, y: y_val}).evalf()
        print(f"Iteration {i + 1}: x={x_val}, y={y_val}, f(x, y) = {f_cur}")


def a_ii_RmsProp(f, iteration=100, alpha0=0.1, beta=0.9, epsilon=1e-3, x_val=10, y_val=5):
    """
    find the minimum of function value with RmsProp step size
    @param f: sympy representation of function f(x)
    @param iteration: iteration times
    @param alpha0: initial alpha value (learning rate)
    @param beta: decay factor for past values
    @param epsilon: avoid division by zero
    @param x_val: x value
    @param y_val: y value
    @return:
    """
    df_dx = sp.diff(f, x)
    df_dy = sp.diff(f, y)

    grad_sum_x, grad_sum_y = 0, 0

    for i in range(iteration):
        grad_x = df_dx.subs({x: x_val, y: y_val}).evalf()
        grad_y = df_dy.subs({x: x_val, y: y_val}).evalf()

        grad_sum_x = beta * grad_sum_x + (1 - beta) * grad_x * grad_x
        grad_sum_y = beta * grad_sum_y + (1 - beta) * grad_y * grad_y

        alphaX = alpha0 / sp.sqrt(grad_sum_x + epsilon)
        alphaY = alpha0 / sp.sqrt(grad_sum_y + epsilon)

        x_val -= alphaX * grad_x
        y_val -= alphaY * grad_y

        f_cur = f.subs({x: x_val, y: y_val}).evalf()
        print(f"Iteration {i + 1}: x = {x_val}, y = {y_val}, f(x, y) = {f_cur}")


def a_iii_HeavyBall():
    pass


def a_iv_Adam(f, iteration=100, alpha0=0.1, epsilon=1e-3, x_val=10, y_val=5):
    """
    find the minimum of function value with adam step size
    @param f: sympy representation of function f(x)
    @param iteration: iteration times
    @param alpha0: initial alpha value (learning rate)
    @param epsilon: avoid division by zero
    @param x_val: x value
    @param y_val: y value
    @return:
    """
    df_dx = sp.diff(f, x)
    df_dy = sp.diff(f, y)

    grad_sum_x, grad_sum_y = epsilon, epsilon

    for i in range(iteration):
        grad_x = df_dx.subs({x: x_val, y: y_val}).evalf()
        grad_y = df_dy.subs({x: x_val, y: y_val}).evalf()

        # accumulate square of gradient x and y
        grad_sum_x += grad_x * grad_x
        grad_sum_y += grad_y * grad_y

        alphaX = alpha0 / sp.sqrt(grad_sum_x)
        alphaY = alpha0 / sp.sqrt(grad_sum_y)

        x_val -= alphaX * grad_x
        y_val -= alphaY * grad_y

        f_cur = f.subs({x: x_val, y: y_val}).evalf()
        print(f"Iteration {i + 1}: x = {x_val}, y = {y_val}, f(x, y) = {f_cur}")


def main():
    # a_i_Polyak(f2)
    # a_iv_Adam(f2, alpha0=0.65)
    a_ii_RmsProp(f2, beta=0.7)


if __name__ == '__main__':
    main()
