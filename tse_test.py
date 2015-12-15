import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from utils import datasets, safe_shape, mse
from TimeSeriesEstimator import TimeSeriesRegressor, time_series_split, time_series_cv, cascade_cv
from sklearn.cross_validation import KFold


def train_test_plot(pred_train, y_train, pred_test, y_test, n_prev, titles, cap=4):
    output_dim = 1 if len(y_test.shape) == 1 else y_test.shape[1]
    if output_dim > cap:
        output_dim = cap
        print(output_dim)

    for i in range(output_dim):
        plt.subplot(output_dim, 2, 2 * i + 1)
        if output_dim == 1:
            plt.plot(pred_train, 'r', label="Predicted")
            plt.plot(y_train[n_prev:], 'b--', label="Actual")
        else:
            plt.plot(pred_train[:, i], 'r', label="Predicted")
            plt.plot(y_train[n_prev:, i], 'b--', label="Actual")
        # nprev: because the first predicted point needed n_prev steps of data
        # plt.title("Training performance of " + titles[i])
        # plt.legend(loc='lower right')

        plt.subplot(output_dim, 2, 2 * i + 2)
        if output_dim == 1:
            plt.plot(pred_test, 'r', label="Predicted")
            plt.plot(y_test[n_prev:], 'b--', label="Actual")
        else:
            plt.plot(pred_test[:, i], 'r', label="Predicted")
            plt.plot(y_test[n_prev:, i], 'b--', label="Actual")
            # nprev: because the first predicted point needed n_prev steps of data
            # plt.title("Testing performance of " + titles[i])
            # plt.legend(loc='lower left')

    plt.gcf().set_size_inches(15, 6)
    plt.show()


X = datasets('sp500')
X_train, X_test = time_series_split(X)


n_prev = 3
tsr = TimeSeriesRegressor(LinearRegression(), n_prev=n_prev)
tsr.fit(X_train)
fc = tsr.forecast(X_train, len(X_test))

def changes(X):
    return np.array([X[-1, i] - X[0, i] for i in range(X.shape[1])])

#X_test_change = np.log(changes(X_test))
#fc_change = np.log(changes(fc))
#plt.plot(np.linspace(0,8),np.linspace(0,8),'b')
#plt.plot(X_test_change,fc_change, 'ro')


for i in range(min(16,X_train.shape[1])):
    plt.subplot(4,4,i+1)
    plt.plot(fc[:, i],'r')
    plt.plot(X_test[:, i],'b')

plt.show()

