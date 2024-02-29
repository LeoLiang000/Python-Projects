import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

df = pd.read_csv('# id8-8-8.csv')  # training data (ID: # id8-8-8)
x1 = df.iloc[:, 0]  # feature 1
x2 = df.iloc[:, 1]  # feature 2
X = np.column_stack((x1, x2))  # concatenate features
Y = df.iloc[:, 2]  # target value
pf = PolynomialFeatures(degree=5, include_bias=False)  # add up features to power 5
poly_X = pf.fit_transform(X)  # transform original features into polynomial features
ci_range = [1, 10, 100, 1000, 10000]  # range of parameter ci
ci_range_ext = [0.0001, 1, 10, 100, 1000]  # extended range of parameter ci
ci_range_cross_lasso = [8, 10, 15, 30, 100, 1000, 15000]  # range of ci for Lasso cross validation
ci_range_cross_ridge = [0.1, 0.4, 1, 10, 1000, 10000]  # range of ci for Ridge cross validation


# plot data
def i_a():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, Y, label='raw data')
    ax.set_xlabel('Feature X1')
    ax.set_ylabel('Feature X2')
    ax.set_zlabel('Target Value Y')
    plt.title('3D Plot of Rawdata')
    plt.legend()
    plt.show()


# add features, train Lasso regression model
def i_b(mode='Lasso'):
    ret = dict()  # results

    local_ci_range = None
    if mode == 'Ridge':
        local_ci_range = ci_range_ext
    elif mode == 'Lasso':
        local_ci_range = ci_range

    for ci in local_ci_range:
        tmp_ret = list()  # temp results
        tmp_int = list()  # temp intercept
        tmp_coe = list()  # temp coefficient

        # apply Lasso regression
        if mode == 'Lasso':
            model = Lasso(alpha=1 / (2 * ci))
        elif mode == 'Ridge':
            model = Ridge(alpha=1 / (2 * ci))
        scores = cross_val_score(model, poly_X, Y, cv=5, scoring="neg_mean_squared_error")

        # apply k-fold cross validation, and train model with training dataset
        for train_data, test_data in KFold(n_splits=5).split(poly_X):
            model.fit(poly_X[train_data], Y[train_data])
            tmp_coe.append(model.coef_)
            tmp_int.append(model.intercept_)

        # fill model parameters into temp result list, with average score, mean intercept, mean coef respectively
        if tmp_coe and tmp_int:
            df_coe = pd.DataFrame(tmp_coe)
            tmp_ret.append(scores.mean())
            tmp_ret.append(np.array(tmp_int).mean())
            tmp_ret += [df_coe[i].mean() for i in range(df_coe.shape[1])]
            ret[ci] = tmp_ret

    if ret:
        print(pd.DataFrame(ret, index=['Score', 'Intercept'] + list(pf.get_feature_names_out())))


# generate predictions using models from i.(b) above and plot
def i_c(mode='Lasso'):
    local_ci_range = None
    if mode == 'Ridge':
        local_ci_range = ci_range_ext
    elif mode == 'Lasso':
        local_ci_range = ci_range

    for ci in local_ci_range:
        # train Lasso regression model
        if mode == 'Lasso':
            model = Lasso(alpha=1 / (2 * ci))
        elif mode == 'Ridge':
            model = Ridge(alpha=1 / (2 * ci))
        model.fit(poly_X, Y)  # do not need split raw data as we generate new test features

        # generate predictions on the grid of features
        new_x1, new_x2 = np.meshgrid(np.linspace(-2, 2), np.linspace(-2, 2))
        pred_Y = np.empty_like(new_x1)
        for i, new_x in enumerate(zip(new_x1, new_x2)):
            new_poly_X = pf.fit_transform(np.column_stack((new_x[0], new_x[1])))  # transform to new polynomial features
            pred_Y[i, :] = model.predict(new_poly_X)  # predicted Y target values

        # draw raw data and surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surface = ax.plot_surface(new_x1, new_x2, pred_Y, cmap='spring', alpha=0.7,
                                  label=f'prediction plane when C={ci}')  # new data
        fig.colorbar(surface, shrink=0.5, aspect=10)
        ax.scatter(x1, x2, Y, label='raw data')  # rawdata
        ax.set_xlabel('Feature X1')
        ax.set_ylabel('Feature X2')
        ax.set_zlabel('Target Value Y')
        plt.title(f'{mode} Regression (C={ci})')
        plt.legend()
        plt.tight_layout()
        plt.show()


# repeat i(b) by Ridge Regression
def i_e_1():
    i_b(mode='Ridge')


# repeat i(c) by Ridge Regression
def i_e_2():
    i_c(mode='Ridge')


# choose C by cross validation
def ii_a(mode='Lasso'):
    err_std = list()
    err_mean = list()

    local_ci_range = None
    if mode == 'Lasso':
        local_ci_range = ci_range_cross_lasso
    elif mode == 'Ridge':
        local_ci_range = ci_range_cross_ridge

    for ci in local_ci_range:
        if mode == 'Lasso':
            model = Lasso(alpha=1 / (2 * ci))
        elif mode == 'Ridge':
            model = Ridge(alpha=1 / (2 * ci))
        err_tmp = list()
        for train_data, test_data in KFold(n_splits=5).split(poly_X):
            model.fit(poly_X[train_data], Y[train_data])
            pred_Y = model.predict(poly_X[test_data])
            err_tmp.append(mean_squared_error(Y[test_data], pred_Y))

        err_tmp = np.array(err_tmp)
        err_std.append(err_tmp.std())
        err_mean.append(err_tmp.mean())

    plt.xscale('log')
    if mode == 'Lasso':
        plt.xlim(5, 18000)
    elif mode == 'Ridge':
        plt.xlim(0.07, 12000)
    plt.errorbar(local_ci_range, err_mean, yerr=err_std)
    plt.xlabel('C Value')
    plt.ylabel('Mean Square Error')
    plt.title('Lasso Regression')
    plt.grid()
    plt.show()


def ii_c():
    ii_a(mode='Ridge')


def main():
    ii_c()


if __name__ == '__main__':
    main()
