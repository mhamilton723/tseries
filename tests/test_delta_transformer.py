from numpy.testing import assert_equal
from sklearn.grid_search import GridSearchCV
from TimeSeriesRegressor.tseries.delta_transformer import *
import pandas as pd

X = pd.DataFrame([[1.0, 1.0, 1.0],
                  [2.0, 0.0, 4.0],
                  [3.0, -1.0, 25.0]], columns=["foo",'bar','baz'])
X_trans = pd.DataFrame([[1.0, -1.0, 4.0],
                        [1.0, -1.0, 25.0]], index=[1,2], columns=["foo",'bar','baz'])


def test_delta_transformer():
    fit_model = DeltaTransformer(omit=['baz']).fit(X)
    X_trans_test = fit_model.transform(X)
    assert(X_trans_test.equals(X_trans))
    X_test = fit_model.inverse_transform(X_trans_test)
    assert(X_test.equals(X))


if __name__ == '__main__':
    test_delta_transformer()
