from tseries.tseries.double_learner import DoubleLearner
import pandas as pd
import numpy as np
from sklearn.linear_model.base import LinearRegression
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from tseries.tseries.alternating_learner import AlternatingLearner

dataset_size = 1000

treatment_dim = 4
confounder_dim = 30
result_dim = 1
iters = 20
theta0 = np.array([1, -1, 1, 5])


def gen_x_y(dataset_size=dataset_size,
            result_dim=result_dim,
            treatment_dim=treatment_dim,
            confounder_dim=confounder_dim,
            theta0=theta0):
    Z = np.random.randn(dataset_size, confounder_dim)

    m0_m = np.random.randn(confounder_dim, treatment_dim)
    m0 = lambda x: np.tanh(np.dot(x, m0_m))

    g0_m = np.random.randn(confounder_dim, result_dim)
    g0_b = np.random.randn(result_dim)
    g0 = lambda x: np.tanh(np.dot(x, g0_m) + g0_b)

    V = np.random.randn(dataset_size, treatment_dim)
    U = np.random.randn(dataset_size, result_dim)

    D = m0(Z) + V
    y = np.dot(D, np.expand_dims(theta0, 1)) + g0(Z) + U
    X = np.hstack([D, Z])

    return X, y


X, y = gen_x_y()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


def mse(y1, y2):
    return np.mean((y1 - y2) ** 2)


treatment_cols = range(0, treatment_dim)
confounder_cols = range(treatment_dim, treatment_dim + confounder_dim)

# lr = LinearRegression(fit_intercept=False).fit(X_train, y_train)
# dr = DoubleLearner(treatment_cols, confounder_cols).fit(X_train, y_train)
al = AlternatingLearner(treatment_cols, confounder_cols).fit(X_train, y_train)
dl = DoubleLearner(treatment_cols,
                   confounder_cols,
                   treatment_estimator=RandomForestRegressor(),
                   y_estimator=RandomForestRegressor()).fit(X_train, y_train)

al_pred = al.predict(X_test)
dl_pred = dl.predict(X_test)
print(mse(al_pred, y_test))
print(mse(dl_pred, y_test))
print(al.estimator_1.coef_[treatment_cols])
print(dl.effect_estimator.coef_)

'''
plt.subplot(2, 2, 1)
plt.plot(lr_pred, y_test, 'r.', alpha=.2)
plt.plot(np.linspace(-20, 20), np.linspace(-20, 20))
plt.subplot(2, 2, 2)
plt.plot(dr_pred, y_test, 'r.', alpha=.2)
plt.plot(np.linspace(-20, 20), np.linspace(-20, 20))
plt.subplot(2, 2, 3)
plt.hist(lr_pred - y_test)
plt.subplot(2, 2, 4)
plt.hist(dr_pred - y_test)

plt.show()

lr_coeffs = []
dl_lr_coeffs = []
for i in range(iters):
    X, y = gen_x_y(dataset_size)
    lr_coeffs.append(LinearRegression(fit_intercept=False).fit(X, y).coef_)
    print(DoubleLearner([0], [1]).fit(X, y).effect_estimator.coef_)
    print(lr_coeffs[i])

x_lr, n_lr = zip(*lr_coeffs)
plt.plot(x_lr, n_lr, 'ro', label='LR')
plt.plot([15], [0], 'b*', ms=20, label='truth')
plt.xlabel('x1 coeff')
plt.ylabel('y1 coeff')
plt.show()
'''
