import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import warnings

x, y = sp.symbols('x y', real=True)
f1 = 10 * (x - 9) ** 4 + 7 * (y - 0) ** 2  # function 1
f2 = sp.Max(x - 9, 0) + 7 * sp.Abs(y)  # function 2


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


def a_ii_RmsProp(f, iteration=100, alpha=0.1, beta=0.9, epsilon=1e-3, x_val=10, y_val=5, path=False):
    """
    find the minimum of function value with RmsProp step size
    @param f: sympy representation of function f(x)
    @param iteration: iteration times
    @param alpha: initial alpha value (learning rate)
    @param beta: decay factor for past values
    @param epsilon: avoid division by zero
    @param x_val: x value
    @param y_val: y value
    @return:
    """
    f_ret = []  # function result
    step_sizes = []  # step sizes for x and y
    path_trajectory = []

    df_dx = sp.diff(f, x)
    df_dy = sp.diff(f, y)

    grad_sum_x, grad_sum_y = 0, 0

    for i in range(iteration):
        grad_x = df_dx.subs({x: x_val, y: y_val}).evalf()
        grad_y = df_dy.subs({x: x_val, y: y_val}).evalf()

        grad_sum_x = beta * grad_sum_x + (1 - beta) * grad_x * grad_x
        grad_sum_y = beta * grad_sum_y + (1 - beta) * grad_y * grad_y

        alphaX = alpha / sp.sqrt(grad_sum_x + epsilon)
        alphaY = alpha / sp.sqrt(grad_sum_y + epsilon)

        x_val -= alphaX * grad_x
        y_val -= alphaY * grad_y

        f_cur = f.subs({x: x_val, y: y_val}).evalf()

        f_ret.append(f_cur)
        step_sizes.append(sp.sqrt(alphaX * alphaX + alphaY * alphaY))
        if path:
            path_trajectory.append((x_val, y_val))

        print(f"Iteration {i + 1}: x = {x_val}, y = {y_val}, f(x, y) = {f_cur}")

    if not path:
        return f_ret, step_sizes
    else:
        return f_ret, step_sizes, path_trajectory


def a_iii_HeavyBall(f, iteration=100, alpha=0.01, beta=0.9, x_val=10, y_val=5, path=False):
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
    f_ret = []  # function result
    step_sizes = []  # step sizes for x and y
    path_trajectory = []

    m_x, m_y = 0, 0  # momentum variable for x and y directions

    df_dx = sp.diff(f, x)
    df_dy = sp.diff(f, y)

    for i in range(iteration):
        grad_x = df_dx.subs({x: x_val, y: y_val}).evalf()
        grad_y = df_dy.subs({x: x_val, y: y_val}).evalf()

        m_x = beta * m_x + alpha * grad_x  # accumulate previous momentum for x
        m_y = beta * m_y + alpha * grad_y  # accumulate previous momentum for y

        x_val -= m_x
        y_val -= m_y

        f_cur = f.subs({x: x_val, y: y_val}).evalf()

        f_ret.append(f_cur)
        step_sizes.append(sp.sqrt(m_x * m_x + m_y * m_y))
        if path:
            path_trajectory.append((x_val, y_val))

        print(f"Iteration {i + 1}: x = {x_val}, y = {y_val}, f(x, y) = {f_cur}")

    if not path:
        return f_ret, step_sizes
    else:
        return f_ret, step_sizes, path_trajectory


def a_iv_Adam(f, iteration=100, alpha=0.1, beta1=0.9, beta2=0.999, epsilon=1e-3, x_val=1, y_val=5, path=False):
    f_ret = []  # function result
    path_trajectory = []

    df_dx = sp.diff(f, x)
    df_dy = sp.diff(f, y)
    m_x, m_y, v_x, v_y = 0, 0, 0, 0

    for i in range(1, iteration + 1):
        grad_x = df_dx.subs({x: x_val, y: y_val}).evalf()
        grad_y = df_dy.subs({x: x_val, y: y_val}).evalf()

        m_x = beta1 * m_x + (1 - beta1) * grad_x
        m_y = beta1 * m_y + (1 - beta1) * grad_y
        m_hat_x = m_x / (1 - beta1 ** i)
        m_hat_y = m_y / (1 - beta1 ** i)

        v_x = beta2 * v_x + (1 - beta2) * grad_x * grad_x
        v_y = beta2 * v_y + (1 - beta2) * grad_y * grad_y
        v_hat_x = v_x / (1 - beta2 ** i)
        v_hat_y = v_y / (1 - beta2 ** i)

        x_val -= alpha * m_hat_x / (sp.sqrt(v_hat_x) + epsilon)
        y_val -= alpha * m_hat_y / (sp.sqrt(v_hat_y) + epsilon)

        f_cur = f.subs({x: x_val, y: y_val}).evalf()
        print(f"Iteration {i + 1}: x = {x_val}, y = {y_val}, f(x, y) = {f_cur}")

        f_ret.append(f_cur)
        if path:
            path_trajectory.append((x_val, y_val))

    if not path:
        return f_ret
    else:
        return f_ret, path_trajectory


def b_i_plot_RMSProp(f, iter_num=100):
    betas = [0.25, 0.9]
    alphas = [0.01, 0.05, 0.1]
    plt.figure(figsize=(10, 10))

    for a in alphas:
        for b in betas:
            f_ret, step_size = a_ii_RmsProp(f, iteration=iter_num, alpha=a, beta=b, x_val=3, y_val=1)

            plt.subplot(2, 1, 1)
            plot_(f_ret, f'alpha={a}, beta={b}', 'Iteration', 'Function Value', f'RMSProp: {f}')

            plt.subplot(2, 1, 2)
            plot_(step_size, f'alpha={a}, beta={b}', 'Iteration', 'Step Size')

    plt.show()


def b_ii_plot_HeaveyBall(f, iter_num=100):
    betas = [0.25, 0.8]
    alphas = [0.01, 0.05]
    plt.figure(figsize=(10, 10))

    for a in alphas:
        for b in betas:
            f_ret, step_size = a_iii_HeavyBall(f, iteration=iter_num, alpha=a, beta=b)

            plt.subplot(2, 1, 1)
            plot_(f_ret, f'alpha={a}, beta={b}', 'Iteration', 'Function Value', f'RMSProp: {f}')

            plt.subplot(2, 1, 2)
            plot_(step_size, f'alpha={a}, beta={b}', 'Iteration', 'Step Size')

    plt.show()


def b_iii_plot_Adam(f, iter_num=100):
    beta1 = [0.25, 0.9]
    beta2 = 0.999
    alphas = [0.25, 0.5, 0.8]
    plt.figure(figsize=(10, 10))

    for a in alphas:
        for b1 in beta1:
            f_ret = a_iv_Adam(f, iteration=iter_num, alpha=a, beta1=b1, beta2=beta2)
            plot_(f_ret, f'alpha={a}, beta1={b1}, beta2={beta2}', 'Iteration', 'Function Value', f'RMSProp: {f}')

    plt.show()


def b_iv_plot_contour(f, iter_num=100):
    x_range = np.linspace(8, 12, 400)
    y_range = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x_range, y_range)
    F = sp.lambdify((x, y), f, 'numpy')
    Z = F(X, Y)

    init_x = 10
    init_y = 5

    fx_rms, step_rms, path_rms = a_ii_RmsProp(f, iteration=iter_num, alpha=0.05, beta=0.9, x_val=init_x, y_val=init_y,
                                              path=True)
    fx_hb, step_hb, path_hb = a_iii_HeavyBall(f, iteration=iter_num, alpha=0.01, beta=0.8, x_val=init_x, y_val=init_y,
                                              path=True)
    fx_adam, path_adam = a_iv_Adam(f, iteration=iter_num, alpha=0.5, beta1=0.9, beta2=0.999, x_val=init_x, y_val=init_y,
                                   path=True)
    rms_x, rms_y = zip(*path_rms)
    hb_x, hb_y = zip(*path_hb)
    adam_x, adam_y = zip(*path_adam)

    # Plot the contour plot
    plt.figure(figsize=(10, 10))
    contours = plt.contour(X, Y, Z, levels=np.logspace(0, 5, 35), cmap='viridis')
    plt.clabel(contours, inline=True, fontsize=8)

    # Plot the optimization paths
    plt.plot(rms_x, rms_y, 'y.-', label='RMSProp Path')
    plt.plot(hb_x, hb_y, 'c.-', label='Heavy Ball Path')
    plt.plot(adam_x, adam_y, 'g.-', label='Adam Path')

    # Adding labels and legend
    plt.title('Optimization paths on contour plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.show()


def plot_(variables, label, xlabel, ylabel, title=None):
    plt.plot(variables, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.yscale('log')


def plot_contour(f, x, y):
    f_lambda = sp.lambdify((x, y), f, 'numpy')

    X = np.linspace(0, 10, 400)
    Y = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(X, Y)
    Z = f_lambda(X, Y)

    plt.figure(figsize=(8, 6))
    contour = plt.contour(X, Y, Z, levels=np.logspace(0, 5, 35), cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=8)
    plt.title(f'Contour plot of {f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(contour)
    # plt.show()


def main():
    # a_i_Polyak(f2)
    # a_iv_Adam(f1, alpha=0.5, beta1=0.9, beta2=0.999)
    # a_ii_RmsProp(f1, beta=0.7)
    # a_iii_HeavyBall(f1, alpha=0.01, beta=0.8)
    # a_iii_HeavyBall(f2, alpha=0.008, beta=0.9)

    # b_i_plot_RMSProp(f1, iter_num=500)
    # b_i_plot_RMSProp(f2)
    # b_ii_plot_HeaveyBall(f1, iter_num=100)
    # b_ii_plot_HeaveyBall(f2, iter_num=50)
    # b_iii_plot_Adam(f2, iter_num=30)
    # b_iii_plot_Adam(f1)
    b_iv_plot_contour(f1, iter_num=50)


if __name__ == '__main__':
    main()
