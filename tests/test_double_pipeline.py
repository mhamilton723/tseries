from sklearn.linear_model.base import LinearRegression
from TimeSeriesRegressor.tseries import DoublePipeline, DeltaTransformer
import pandas as pd
import numpy as np

X = pd.DataFrame([[1.0, 1.0, 1.0],
                  [2.0, 0.0, 4.0],
                  [3.0, -2.0, 25.0]], columns=["foo", 'bar', 'baz'])
X_trans = pd.DataFrame([[1.0, -1.0, 3.0],
                        [1.0, -2.0, 21.0]], columns=["foo", 'bar', 'baz'])

Y = pd.DataFrame([1.0, 2.0, 3.0], index=[0,0,1])


def test_delta_transformer():
    fit_model = DoublePipeline(
        [('xdelta', DeltaTransformer()), ('linreg', LinearRegression())],
        [('ydelta', DeltaTransformer())]).fit(X, Y)

    #print(fit_model.x_pipe_.steps[1][1])
    #print fit_model.predict(X)
    assert (fit_model.predict(X).equals(Y))


if __name__ == '__main__':
    test_delta_transformer()
