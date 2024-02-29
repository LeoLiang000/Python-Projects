import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve

warnings.filterwarnings("ignore")


def update_dataset(dataset):
    global df, x1, x2, X, Y, pf, poly_X, x_train, x_test, y_train, y_test, x_train_poly, x_test_poly, y_train_poly, y_test_poly, model_knn, model_log

    ds1 = '1# id6--6-6-0.csv'  # dataset 1
    ds2 = '2# id6-12-6-0.csv'  # dataset 2

    # the first dataset
    if dataset == 1:
        df = pd.read_csv(ds1)
        pf = PolynomialFeatures(degree=4, include_bias=False)  # choose polynomial degree=4 for dataset 1
        model_knn = KNeighborsClassifier(n_neighbors=11, weights='uniform')  # choose K=11 for dataset 1
        model_log = LogisticRegression(penalty='l2', C=100)  # choose C=100 for dataset 1

    # the second dataset
    elif dataset == 2:
        df = pd.read_csv(ds2)
        pf = PolynomialFeatures(degree=3, include_bias=False)  # choose polynomial degree=3 for dataset 2
        model_knn = KNeighborsClassifier(n_neighbors=29, weights='uniform')  # choose K=21 for dataset 2
        model_log = LogisticRegression(penalty='l2', C=10)  # choose C=10 for dataset 2

    # input data
    x1 = df.iloc[:, 0]
    x2 = df.iloc[:, 1]
    X = np.column_stack((x1, x2))
    Y = df.iloc[:, 2]
    poly_X = pf.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    x_train_poly, x_test_poly, y_train_poly, y_test_poly = train_test_split(poly_X, Y, test_size=0.2)


# select the maximum order of polynomial to use
def question_a_i(dataset=1):
    update_dataset(dataset)  # choose dataset

    if dataset == 1:
        ci_range = [1e-3, 1e-2, 0.1, 1, 10, 1e2, 1e3, 1e4, 'Max']
    elif dataset == 2:
        ci_range = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 1e2, 1e3, 1e4, 1e5, 'Max']

    ret = dict()  # store result
    for i in range(1, 7):  # add polynomial features power to 6
        tmp_pf = PolynomialFeatures(degree=i, include_bias=False)  # temp polynomial features
        temp_poly_X = tmp_pf.fit_transform(X)

        tmp_scores = list()
        for ci in ci_range[:-1]:
            model = LogisticRegression(C=ci, penalty='l2', solver='lbfgs')
            tmp_f1 = cross_val_score(model, temp_poly_X, Y, cv=5, scoring='f1')
            tmp_scores.append(tmp_f1.mean())
        tmp_scores += [max(tmp_scores)]
        ret[i] = tmp_scores

    with pd.option_context('display.expand_frame_repr', False):
        df_ret = pd.DataFrame(ret, index=ci_range)
        print(df_ret)


# select the weight C given to penalty in the cost function
def question_a_ii(dataset=1):
    update_dataset(dataset)  # choose dataset

    if dataset == 1:
        ci_range = [1e-2, 0.1, 1, 10, 100, 1e3, 1e4, 1e5]
    elif dataset == 2:
        ci_range = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 1e2, 1e3, 1e4, 1e5]

    f1_mean = list()
    f1_std = list()

    for ci in ci_range:
        model = LogisticRegression(penalty='l2', C=ci)
        tmp_scores = cross_val_score(model, poly_X, Y, cv=5, scoring='f1')
        f1_mean.append(tmp_scores.mean())
        f1_std.append(tmp_scores.std())

    plt.xscale('log')
    plt.errorbar(ci_range, f1_mean, yerr=f1_std, color='gray', ecolor='r', capsize=5)
    plt.xlabel('C Value')
    plt.ylabel('Mean F1 Score')
    if dataset == 1:
        plt.title('Logistic Regression with polynomial features up to 4th degree')
    elif dataset == 2:
        plt.title('Logistic Regression with polynomial features up to 3rd degree')
    plt.show()


# plot data
def question_a_iii(dataset=1):
    update_dataset(dataset)  # choose dataset

    # train logistic regression model using training dataset
    model_log.fit(x_train_poly, y_train_poly)

    # use trained model to predict Y
    Y_pred = model_log.predict(poly_X)

    # set plot attributes
    ax = plt.subplot(111, projection='3d')
    ax.scatter(x1, x2, Y, c='red', label='Raw Data', alpha=0.8)
    ax.scatter(x1, x2, Y_pred, c='blue', label='Predicted Y', alpha=0.2)
    plt.xlabel('Feature X1')
    plt.ylabel('Feature X2')
    ax.set_zlabel('Target Y')
    plt.title("Logistic Regression Prediction")
    plt.legend()
    plt.tight_layout()
    plt.show()


# train KNN classifier and choose K=11 based on result
def question_b(dataset=1):
    update_dataset(dataset)  # choose dataset

    if dataset == 1:
        k_range = range(1, 31)
    elif dataset == 2:
        k_range = range(1, 51)

    f1_mean = list()
    f1_std = list()

    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k, weights='uniform')
        tmp_f1 = cross_val_score(model, X, Y, cv=5, scoring='f1')
        f1_mean.append(tmp_f1.mean())
        f1_std.append(tmp_f1.std())

    plt.errorbar(k_range, f1_mean, yerr=f1_std, color='gray', ecolor='r', capsize=5)
    plt.xlabel('K Value')
    plt.ylabel('Mean F1 Score')
    plt.title('kNN Classifier with differing K values')
    plt.show()


# calculate confusion matrices
def question_c(dataset=1):
    update_dataset(dataset)  # choose dataset

    # apply dummy classifier to predict target values
    dummy_type = ['most_frequent', 'uniform']
    for d_type in dummy_type:
        d_c = DummyClassifier(strategy=d_type)
        d_c.fit(X, Y)
        Y_pred_dummy = d_c.predict(X)
        print(f'{d_type} confusion matrix\n{confusion_matrix(Y, Y_pred_dummy)}')
        print(f'{d_type} classification report\n{classification_report(Y, Y_pred_dummy)}')

    # apply knn model to predict target values
    model_knn.fit(x_train, y_train)
    Y_pred_knn = model_knn.predict(x_test)
    print(f'KNN confusion matrix\n{confusion_matrix(y_test, Y_pred_knn)}')
    print(f'KNN classification report\n{classification_report(y_test, Y_pred_knn)}')

    # apply logistic model to predict target values
    model_log.fit(x_train_poly, y_train_poly)
    Y_pred_log = model_log.predict(x_test_poly)
    print(f'Logistic Regression confusion matrix\n{confusion_matrix(y_test_poly, Y_pred_log)}')
    print(f'Logistic Regression classification report\n{classification_report(y_test_poly, Y_pred_log)}')


# plot ROC curves for Logistic Regression and kNN classifiers
def question_d(dataset=1):
    update_dataset(dataset)  # choose dataset

    plt.rc('font', size=18)
    plt.rcParams['figure.constrained_layout.use'] = True

    # create baseline predictor(random)
    dummy = DummyClassifier(strategy="uniform").fit(x_train, y_train)
    y_pred_dummy = dummy.predict_proba(x_test)[:, 1]
    fal_pos_rate, true_pos_rate, _ = roc_curve(y_test, y_pred_dummy)
    plt.plot(fal_pos_rate, true_pos_rate, label='Baseline Predictor(Random)')

    # create baseline predictor(most frequent)
    dummy = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    y_pred_dummy = dummy.predict_proba(x_test)[:, 1]
    fal_pos_rate, true_pos_rate, _ = roc_curve(y_test, y_pred_dummy)
    plt.plot(fal_pos_rate, true_pos_rate, linestyle='dotted', label='Baseline Predictor(Most Frequent)')

    # plot knn prediction
    model_knn.fit(x_train, y_train)
    y_pred_knn = model_knn.predict_proba(x_test)[:, 1]
    fal_pos_rate, true_pos_rate, _ = roc_curve(y_test, y_pred_knn)
    if dataset == 1:
        plt.plot(fal_pos_rate, true_pos_rate, label='kNN(K=11)')
    elif dataset == 2:
        plt.plot(fal_pos_rate, true_pos_rate, label='kNN(K=29)')

    # plot logistic regression prediction
    model_log.fit(x_train, y_train)
    y_pred_log = model_log.predict_proba(x_test)[:, -1]
    fal_pos_rate, true_pos_rate, _ = roc_curve(y_test, y_pred_log)
    if dataset == 1:
        plt.plot(fal_pos_rate, true_pos_rate, label='Logistic Regression(Poly=4, C=100)')
    elif dataset == 2:
        plt.plot(fal_pos_rate, true_pos_rate, label='Logistic Regression(Poly=3, C=10)')

    # set plot attribute
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()


# use the second dataset for question i(a).1
def question_a_i_2():
    question_a_i(dataset=2)


# use the second dataset for question i(a).2
def question_a_ii_2():
    question_a_ii(dataset=2)


# use the second dataset for question i(a).3
def question_a_iii_2():
    question_a_iii(dataset=2)


# use the second dataset for question i(b)
def question_b_2():
    question_b(dataset=2)


# use the second dataset for question i(c)
def question_c_2():
    question_c(dataset=2)


# use the second dataset for question i(d)
def question_d_2():
    question_d(dataset=2)


def main():
    # call function with question number
    question_d()


if __name__ == '__main__':
    main()
