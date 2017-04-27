"""
Delta Transformer Class

Author: Mark Hamilton, mhamilton723@gmail.com
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


# TODO fix indexing
class DeltaTransformer(BaseEstimator, TransformerMixin):
    """
    Base Class for Time Series Estimators
    """

    def __init__(self, omit=None):
        self.omit = omit
        self.is_fit = False
        self.biases_ = None

    def set_params(self, **params):
        return super(DeltaTransformer, self).set_params(**params)

    def __repr__(self):
        return "DeltaTransformer(power={})".format(self.power)

    def fit(self, X, Y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        ''' X and Y are datasets in chronological order, or X is a time series '''
        if len(X.shape) > 1:
            self.biases_ = X.iloc[0:1, :]
        else:
            self.biases_ = X.iloc[0]
        return self

    def _delta(self, data, omit=None):
        diff_data = data.diff()
        if omit is not None:
            for colnum in omit:
                diff_data.iloc[:, colnum] = data.iloc[:, colnum]
        return diff_data.iloc[1:, :]

    def transform(self, X):
        if len(X.shape)<2:
            X = np.expand_dims(X,1)
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        return np.squeeze(self._delta(X, self.omit))

    def inverse_transform(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if len(X.shape)<2:
            X = np.expand_dims(X,1)

        new_X = np.vstack((self.biases_, X))
        new_X = new_X.cumsum(axis=0)
        if self.omit is not None:
            for colnum in self.omit:
                new_X[1:, colnum] = X[:, colnum]
                new_X[0, colnum] = pd.DataFrame(self.biases_).iloc[0, colnum]
        return np.squeeze(new_X)
