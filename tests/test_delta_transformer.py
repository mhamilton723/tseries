from tseries.tseries.delta_transformer import *
import pandas as pd

X = pd.DataFrame([[1.0, 1.0, 1.0],
                  [2.0, 0.0, 4.0],
                  [3.0, -1.0, 25.0],
                  [4.0, -1.0, 25.0],
                  [5.0, -1.0, 25.0]], columns=["foo", 'bar', 'baz'])

X_trans = pd.DataFrame([[1.0, -1.0, 4.0],
                        [1.0, -1.0, 25.0],
                        [1.0, 0.0, 25.0],
                        [1.0, 0.0, 25.0]], index=[1, 2, 3, 4], columns=["foo", 'bar', 'baz'])

X2 = X['foo']
X2_trans = X_trans['foo']


def test_delta_transformer(X, X_trans, omit, log=False):
    fit_model = DeltaTransformer(omit=omit).fit(X)
    X_trans_test = fit_model.transform(X)
    if log:
        print X_trans_test
        print X_trans
    assert (np.isclose(X_trans_test, X_trans).all())
    X_test = fit_model.inverse_transform(X_trans_test)
    if log:
        print X_test
        print X
    assert (np.isclose(X_test, np.array(X)).all())


if __name__ == '__main__':
    test_delta_transformer(X, X_trans, omit=[2])
    test_delta_transformer(X2, X2_trans, omit=None)
