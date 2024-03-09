import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

x, y = sp.symbols('x y', real=True)
f1 = 10 * (x - 9) ** 4 + 7 * (y - 0) ** 2  # function: 10*(x-9)^4+7*(y-0)^2
f2 = sp.Max(x - 9, 0) + 7 * sp.Abs(y)  # function: Max(x-9,0)+7*|y-0|


def a_i():
    def polyak(f, iteration=100, epsilon=0.001):
        """
        update function value with ployak step size
        @param f: sympy representation of function f(x)
        @param iteration: iteration times
        @return:
        """
        df_dx = sp.diff(f, x)
        df_dy = sp.diff(f, y)

        x_val = 10  # initial x value
        y_val = 5  # initial y value
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

    polyak(f2)


def main():
    a_i()


if __name__ == '__main__':
    main()
