import sys

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x, y = sp.symbols('x y')


def generate_trainingdata(m=25):
    return np.array([0, 0]) + 0.25 * np.random.randn(m, 2)


def f1(x_val, y_val, minibatch):
    # loss function sum_{w in training data} f(x,w)
    y_sum = sp.sympify(0)
    for w in minibatch:
        z_x = x_val - w[0] - 1
        z_y = y_val - w[1] - 1
        y_sum += sp.Min(21 * (z_x ** 2 + z_y ** 2), (z_x + 8) ** 2 + (z_y + 4) ** 2)
    return y_sum / len(minibatch)


def f2(x, minibatch):
    # loss function sum_{w in training data} f(x,w)
    y = 0
    count = 0
    for w in minibatch:
        z = x - w - 1
        y = y + min(34 * (z[0] ** 2 + z[1] ** 2), (z[0] + 2) ** 2 + (z[1] + 2) ** 2)
        count = count + 1
    return y / count


def SGD(x_initial, epochs, batch_size, update_param, step_size=0.01, beta=0.9, beta2=0.999):
    updated_vals = []  # store all updated vals: x, y, f

    x_val, y_val = x_initial
    grad_sum = np.zeros(2)
    m_x, m_y = np.zeros_like(x_val), np.zeros_like(y_val)
    v_x, v_y = np.zeros_like(x_val), np.zeros_like(y_val)

    for i in range(epochs):
        data = generate_trainingdata()
        np.random.shuffle(data)

        for j in range(len(data) // batch_size):
            minibatch = data[j * batch_size: (j + 1) * batch_size]

            grad = get_grad(f1(x, y, minibatch), x_val, y_val)
            f_cur = f1_lambdified(x_val, y_val)

            if update_param.__name__ == "const_step":
                x_val, y_val = update_param(np.array([x_val, y_val]), grad, alpha=step_size)
            if update_param.__name__ == "polyak":
                x_val, y_val = update_param(np.array([x_val, y_val]), grad, f_cur)
            elif update_param.__name__ == "rmsprop":
                [x_val, y_val], grad_sum = update_param(np.array([x_val, y_val]), grad, grad_sum, alpha=step_size)
            elif update_param.__name__ == "heavyball":
                x_val, y_val = update_param(np.array([x_val, y_val]), grad, np.array([m_x, m_y]), alpha=step_size,
                                            beta=beta)
            elif update_param.__name__ == "adam":
                x_val, y_val = update_param([x_val, y_val], grad, np.array([m_x, m_y]), np.array([v_x, v_y]),
                                            alpha=step_size, beta1=beta, beta2=beta2)
            updated_vals.append((x_val, y_val, f_cur))
            print(f"Iteration{i + 1}.Batch{j + 1}: (x, y) ={x_val, y_val}, f(x)={f_cur}, grad={grad}")
    return updated_vals


def get_grad(f, x_val, y_val):
    df_dx = sp.diff(f, x)
    df_dy = sp.diff(f, y)
    grad_x = df_dx.subs({x: x_val, y: y_val}).evalf()
    grad_y = df_dy.subs({x: x_val, y: y_val}).evalf()
    return np.array([grad_x, grad_y], dtype=float)


def const_step(x, gradient, alpha=0.2):
    return x - alpha * gradient


def polyak(x_val, grad, f_cur, f_star=0, epsilon=1e-3):
    grad_sum = np.sum(grad * grad)
    alpha = (f_cur - f_star) / (grad_sum + epsilon)
    return x_val - alpha * grad


def rmsprop(x_val, grad, grad_sum, beta=0.9, alpha=0.1, epsilon=1e-3):
    grad_sum = beta * grad_sum + (1 - beta) * np.square(grad)
    adjusted_grad = grad / (np.sqrt(grad_sum) + epsilon)
    x_val -= alpha * adjusted_grad
    return x_val, grad_sum


def heavyball(x_val, grad, m_prev, beta=0.9, alpha=0.1):
    m = beta * m_prev + alpha * grad
    x_val -= m
    return x_val


def adam(x_val, grad, m_prev, v_prev, beta1=0.9, beta2=0.999, alpha=0.001, epsilon=1e-8, iteration=1):
    m = beta1 * m_prev + (1 - beta1) * grad
    v = beta2 * v_prev + (1 - beta2) * np.square(grad)
    m_hat = m / (1 - beta1 ** iteration)
    v_hat = v / (1 - beta2 ** iteration)
    x_val -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)
    return x_val


# a(ii)
def plot_SGD():
    fig = plt.figure(figsize=(14, 6))

    # Wireframe plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_wireframe(x_grid, y_grid, f1_values, color='green')
    ax1.set_title('Wireframe of f1')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f1(x, y)')

    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(x_grid, y_grid, f1_values, 50, cmap='RdGy')
    fig.colorbar(contour, ax=ax2)
    ax2.set_title('Contour Plot of f1')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    plt.show()


# Plot setting
x_range = np.linspace(-8, 6, 400)
y_range = np.linspace(-8, 6, 400)
# x_range = np.linspace(-1, 3, 400)
# y_range = np.linspace(-1, 3, 400)
x_grid, y_grid = np.meshgrid(x_range, y_range)

T = generate_trainingdata(m=20)  # generate training data
f1_lambdified = sp.lambdify((x, y), f1(x, y, T), 'numpy')  # speed up computation
f1_values = f1_lambdified(x_grid, y_grid)

plt.figure(figsize=(8, 6))
contour = plt.contour(x_grid, y_grid, f1_values, levels=50, cmap='RdGy')
plt.colorbar(contour)
plt.xlabel('x')
plt.ylabel('y')


# b(i)
def grad_descent_const_step(alpha=0.2, iterations=20):
    x_initial = np.array([3, 3])

    # Gradient descent
    vals = x_initial.astype(np.float64)  # vals: x and y values
    for _ in range(iterations):
        grads = get_grad(f1(x, y, T), vals[0], vals[1])  # grads: gx and gy
        vals -= alpha * grads  # Update step
        f_cur = f1_lambdified(vals[0], vals[1])
        print(f"Iteration {_}: x={vals}, f(x)={f_cur}, grad={grads}")
        plt.plot(vals[0], vals[1], '-o', color='red')

    plt.title('Contour of Gradient Descent (Constant Step-Size)')
    plt.legend()
    plt.show()


# b(ii)
def SGD_const_run_several_times(x_initial, num_epochs, batch_size, const_step):
    alpha = 0.2  # same as b(i)
    for run in range(3):
        updated_vals = SGD(x_initial, num_epochs, batch_size, const_step, step_size=alpha)
        x, y, f = zip(*updated_vals)
        plt.plot(x, y, '-o', label=f'Run {run + 1}')

    plt.title('Contour of SGD (Constant Step-Size)')
    plt.legend()
    plt.show()


# b(iii)
def SGD_varying_batch(x_initial, num_epochs, const_step):
    alpha = 0.2  # same as b(i) & b(ii)
    for batch_size in [5, 10, 20]:
        updated_vals = SGD(x_initial, num_epochs, batch_size, const_step, step_size=alpha)
        x, y, f = zip(*updated_vals)
        plt.plot(x, y, '-o', label=f'Batch size {batch_size}')
    plt.title('Contour of SGD (Varying Batch)')
    plt.legend()
    plt.show()


# b(iv)
def SGD_varying_step_size(x_initial, num_epochs, batch_size, const_step):
    for alpha in [0.001, 0.01, 0.1, 0.2]:
        updated_vals = SGD(x_initial, num_epochs, batch_size, const_step, step_size=alpha)
        x, y, f = zip(*updated_vals)
        plt.plot(x, y, '-o', label=f'Step size {alpha}')
    plt.title('Contour of SGD (Varying Step-Size)')
    plt.legend()
    plt.show()


# c(i) polyak
def SGD_polyak_plot(x_initial, num_epochs):
    for batch_size in [5, 10, 20]:
        updated_vals = SGD(x_initial, num_epochs, batch_size, polyak)
        x, y, f = zip(*updated_vals)
        plt.plot(x, y, '-o', label=f'Batch size {batch_size}')

    plt.title('Contour of Mini-Batch SGD: Polyak')
    plt.legend()
    plt.show()


# c(ii) rmsprop
def SGD_rmsprop_plot(x_initial, num_epochs):
    step_size = 1.0
    beta = 0.9

    for batch_size in [5, 10, 20]:
        updated_vals = SGD(x_initial, num_epochs, batch_size, rmsprop, step_size, beta)
        x, y, f = zip(*updated_vals)
        plt.plot(x, y, '-o', label=f'Batch size {batch_size}')

    plt.title('Contour of Mini-Batch SGD: RMSProp')
    plt.legend()
    plt.show()


# c(iii) heavyball
def SGD_heavyball_plot(x_initial, num_epochs):
    alpha = 0.085
    #alpha = 0.05
    beta = 0.9
    for batch_size in [5, 10, 20]:
        updated_vals = SGD(x_initial, num_epochs, batch_size, heavyball, step_size=alpha, beta=beta)
        x, y, f = zip(*updated_vals)
        plt.plot(x, y, '-o', label=f'Batch size {batch_size}')

    plt.title('Contour of Mini-Batch SGD: Heavyball')
    plt.legend()
    plt.show()


# c(iv) adam
def SGD_adam_plot(x_initial, num_epochs, iteration=20):
    alpha = 1.65
    # alpha = 0.5
    beta1 = 0.9
    beta2 = 0.999
    for batch_size in [5, 10, 20]:
        updated_vals = SGD(x_initial, num_epochs, batch_size, adam, step_size=alpha, beta=beta1, beta2=beta2)
        x, y, f = zip(*updated_vals)
        plt.plot(x, y, '-o', label=f'Batch size {batch_size}')

    plt.title('Contour of Mini-Batch SGD: Adam')
    plt.legend()
    plt.show()


def main():
    x_initial = np.array([3.0, 3.0])  # Initial vector x
    num_epochs = 15
    batch_size = 5

    # =============#
    # Question a
    # =============#
    # updated_vals = SGD(x_initial, num_epochs, batch_size, polyak)
    # updated_vals = SGD(x_initial, num_epochs, batch_size, rmsprop)
    # updated_vals = SGD(x_initial, num_epochs, batch_size, heavyball)
    # updated_vals = SGD(x_initial, num_epochs, batch_size, adam)
    # updated_vals = SGD(x_initial, num_epochs, batch_size, const_step)
    # plot_SGD()

    # =============#
    # Question b
    # =============#
    # grad_descent_const_step(alpha=0.2)  # pass
    # grad_descent_const_step(alpha=0.01)  # stuck in local minima
    # SGD_const_run_several_times(x_initial, num_epochs, batch_size, const_step)
    # SGD_varying_batch(x_initial, num_epochs, const_step)
    # SGD_varying_step_size(x_initial, num_epochs, batch_size, const_step)

    # =============#
    # Question c
    # =============#

    x_initial = np.array([3.0, 3.0])
    # SGD_polyak_plot(x_initial, num_epochs)  # pass
    # SGD_rmsprop_plot(x_initial, num_epochs)  # pass
    # SGD_heavyball_plot(x_initial, num_epochs)  # pass
    SGD_adam_plot(x_initial, num_epochs)


if __name__ == '__main__':
    main()
