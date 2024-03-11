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
    f_ret = []  # function result

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
        f_ret.append(f_cur)

        print(f"Iteration {i + 1}: x = {x_val}, y = {y_val}, f(x, y) = {f_cur}")

    return f_ret


def a_iii_HeavyBall(f, iteration=100, alpha=0.1, beta=0.9, x_val=10, y_val=5):
    """
    find the minimum of function value with Polyak Momentum
    @param f: sympy representation of function f(x)
    @param iteration: iteration times
    @param alpha: learning rate
    @param beta: coefficient determining how much previous momentum would affect current point
    @param x_val: initial x value
    @param y_val: initial y value
    @return:
    """

    z_x, z_y = 0, 0  # momentum variable for x and y

    df_dx = sp.diff(f, x)
    df_dy = sp.diff(f, y)

    for i in range(iteration):
        grad_x = df_dx.subs({x: x_val, y: y_val}).evalf()
        grad_y = df_dy.subs({x: x_val, y: y_val}).evalf()

        z_x = beta * z_x + alpha * grad_x  # accumulate previous momentum for x
        z_y = beta * z_y + alpha * grad_y  # accumulate previous momentum for y

        x_val -= z_x
        y_val -= z_y

        f_cur = f.subs({x: x_val, y: y_val}).evalf()
        print(f"Iteration {i + 1}: x = {x_val}, y = {y_val}, f(x, y) = {f_cur}")


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


def b_i_plot_RMSProp(f, beta=0.7):
    f_ret = a_ii_RmsProp(f, beta=beta)
    plt.plot(f_ret, label=f'alpha={0.1}, beta={beta}')
    plt.title('Function Value vs. Iteration for RMSprop')
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.legend()
    plt.yscale('log')  # &#8203;``【oaicite:0】``&#8203;
    plt.show()


def b_ii_plot_HeaveyBall():
    pass


def b_iii_plot_Adam():
    pass


def main():
    # a_i_Polyak(f2)
    # a_iv_Adam(f2, alpha0=0.65)
    # a_ii_RmsProp(f1, beta=0.7)
    # a_iii_HeavyBall(f1, alpha=0.01, beta=0.8)
    # a_iii_HeavyBall(f2, alpha=0.008, beta=0.9)

    b_i_plot_RMSProp(f1)

if __name__ == '__main__':
    main()
