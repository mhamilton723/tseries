from sklearn.linear_model import LinearRegression
from tseries.tseries.models import LinearRegressionWithUncertainty
from tseries.tseries.utils import mse
import pandas as pd
import numpy as np

n = 300
dx = 5
dy = 1
X = np.random.randn(n, dx)
M = np.random.randn(dx, dy)
Y = np.dot(X, M) + .1 * np.random.randn(n, dy)


def test_linear_regression_with_uncertainty():
    fitLR = LinearRegression().fit(X, Y)
    fitLRU = LinearRegressionWithUncertainty().fit(X, Y)

    predLR = fitLR.predict(X)
    predLRU = fitLRU.predict(X)

    assert ((fitLRU.confidence_intervals()[1] - fitLRU.coef_) > 0).all()
    assert ((fitLRU.confidence_intervals()[0] - fitLRU.coef_) < 0).all()
    assert (np.isclose(fitLR.coef_, fitLRU.coef_).all)
    assert (np.isclose(mse(predLR, Y), mse(predLRU, Y), rtol=1e-2))
    # assert ()
    # assert (np.isclose(fit_model.x_pipe_.steps[-1][1].coef_, [1.0, 0.0, 0.0]).all())


if __name__ == '__main__':
    test_linear_regression_with_uncertainty()
