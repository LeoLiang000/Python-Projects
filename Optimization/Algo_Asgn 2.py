import sympy
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# formula expression
x = sympy.symbols('x', real=True)
f = x ** 4
df_dx = sympy.diff(f, x)
# print(f'Expression for dy/dx is: {df_dx}')

# calculate derivative using expression
f = sympy.lambdify(x, f)
df_dx = sympy.lambdify(x, df_dx)

x = np.linspace(-5, 5, 400)


def a_i():
    x = sympy.symbols('x', real=True)
    f = x ** 4
    df_dx = sympy.diff(f, x)
    print(f'Expression for dy/dx is: {df_dx}')


def a_ii():
    # finite difference
    delta = 0.01
    df_dx_finite = ((x + delta) ** 4 - x ** 4) / delta

    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, df_dx(x), label='Derivative ($4x^3$)', color='blue')
    plt.plot(x, df_dx_finite, label=f'Finite difference estimate (delta={delta})', linestyle='--', color='red')
    plt.title('Comparison of derivative and finite difference estimate')
    plt.xlabel('x')
    plt.ylabel('Derivative Value')
    plt.legend()
    plt.grid(True)
    plt.show()


def a_iii():
    # plot accurate derivative
    plt.figure(figsize=(10, 6))
    plt.plot(x, df_dx(x), label='Exact Derivative ($4x^3$)', color='blue')

    # finite difference
    color = ['red', 'orange', 'green', 'purple']
    i = 0
    for delta in [0.001, 0.01, 0.1, 1]:
        df_dx_finite = ((x + delta) ** 4 - x ** 4) / delta
        plt.plot(x, df_dx_finite, label=f'Finite Difference Estimate (delta={delta})', linestyle='--', color=color[i])
        i += 1

    plt.title('Comparison of derivative and finite difference estimate')
    plt.xlabel('x')
    plt.ylabel('Derivative Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


class QuadraticFn:
    # function expression
    def f(self, x):
        return x ** 2

    # derivative of f(x)
    def df(self, x):
        return 2 * x


class QuarticFn:
    # function expression
    def f(self, x):
        return x ** 4

    # derivative of f(x)
    def df(self, x):
        return 4 * x ** 3


class QuadraticFnGama:
    def __init__(self, gamma):
        self.gamma = gamma

    def f(self, x):
        return self.gamma * x ** 2

    def df(self, x):
        return 2 * self.gamma * x


class AbsFn:
    def __init__(self, gamma):
        self.gamma = gamma

    def f(self, x):
        return self.gamma * np.abs(x)

    def df(self, x):
        return self.gamma * np.sign(x)


def gradDescent(fn, x0=1, alpha=0.15, num_iters=50):
    x = x0
    X = np.array([x])
    F = np.array(fn.f(x))

    for i in range(num_iters):
        step = alpha * fn.df(x)
        x -= step
        X = np.append(X, [x], axis=0)
        F = np.append(F, fn.f(x))

    return X, F


# gradient descent
def b_i():
    X, F = gradDescent(QuadraticFn())
    print(F)


def b_ii():
    X, F = gradDescent(QuarticFn(), alpha=0.1)

    # plot x and y(x) against iteration number
    iterations = np.arange(X.shape[0])

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(iterations, X, marker='+')
    plt.xlabel('Iteration')
    plt.ylabel('x')
    plt.title('Variation of x against iteration')

    plt.subplot(1, 2, 2)
    plt.plot(iterations, F, marker='+')
    plt.xlabel('Iteration')
    plt.ylabel('y(x)')
    plt.title('Variation of y(x) against iteration')

    plt.tight_layout()
    plt.show()


def b_iii():
    init_x = np.linspace(-2, 2, 5)
    alphas = np.linspace(0.05, 0.5, 5)

    results = dict()
    for x0 in init_x:
        for alpha in alphas:
            results[(x0, alpha)] = gradDescent(QuarticFn(), x0=x0, alpha=alpha)

    # plot
    plt.figure(figsize=(15, 10))
    for i, (x0, alpha) in enumerate(results.keys(), start=1):
        X, F = results[(x0, alpha)]
        iterations = np.arange(X.shape[0])
        plt.subplot(len(init_x), len(alphas), i)
        plt.plot(iterations, F, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('y(x)')
        plt.title(f'x0={x0}, alpha={alpha}')

    plt.tight_layout()
    plt.show()


def c_i():
    alpha = 0.15
    gamma_values = [0.5, 1, 2, 5]
    results = {}

    # Perform gradient descent for each gamma
    for gamma in gamma_values:
        qua_fn_gama = QuadraticFnGama(gamma)
        X, F = gradDescent(qua_fn_gama, alpha=alpha)
        results[gamma] = (X, F)

    # Plot convergence behavior for each gamma
    plt.figure(figsize=(10, 6))
    for gamma, (X, F) in results.items():
        iterations = np.arange(X.shape[0])
        plt.plot(iterations, F, marker='o', label=f'gamma={gamma}')

    plt.xlabel('Iteration')
    plt.ylabel('y(x)')
    plt.title('Convergence Behavior for Different Gamma Values')
    plt.legend()
    plt.grid(True)
    plt.show()


def c_ii():
    alpha = 0.15
    gamma_values = [0.5, 1, 2, 5]
    results = {}

    for gamma in gamma_values:
        abs_fn = AbsFn(gamma)
        X, F = gradDescent(abs_fn, alpha=alpha)
        results[gamma] = (X, F)

    # Plot convergence behavior for each gamma
    plt.figure(figsize=(10, 6))
    for gamma, (X, F) in results.items():
        iterations = np.arange(X.shape[0])
        plt.plot(iterations, F, marker='o', label=f'gamma={gamma}')

    plt.xlabel('Iteration')
    plt.ylabel('y(x)')
    plt.title('Convergence Behavior for Different Gamma Values')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # a_iii()
    # b_iii()
    c_ii()
    # a_iii()


if __name__ == '__main__':
    main()
