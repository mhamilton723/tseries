from sklearn.linear_model.base import LinearRegression
from TimeSeriesRegressor.tseries import DoublePipeline, DeltaTransformer
import pandas as pd
import numpy as np

X = pd.DataFrame([[1.0, 1.0, 1.0],
                  [2.0, 0.0, 4.0],
                  [3.0, -2.0, 25.0],
                  [5.0, -8.0, 2.0],
                  [6.0, -20.0, 5.0]], columns=["foo", 'bar', 'baz'])

Y = np.array(pd.DataFrame(X['foo']))+30


def test_delta_transformer():
    fit_model = DoublePipeline(
        [('xdelta', DeltaTransformer()), ('linreg', LinearRegression(fit_intercept=False))],
        [('ydelta', DeltaTransformer())]).fit(X, Y)

    assert (np.isclose(fit_model.predict(X), np.squeeze(Y)).all())
    assert (np.isclose(fit_model.x_pipe_.steps[-1][1].coef_, [1.0, 0.0, 0.0]).all())

if __name__ == '__main__':
    test_delta_transformer()
