"""
Delta Transformer Class

Author: Mark Hamilton, mhamilton723@gmail.com
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import warnings
from sklearn.decomposition import PCA
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
            print("Converting")
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
            for colname in omit:
                diff_data[colname] = data[colname]
        return diff_data.iloc[1:, :]

    def transform(self, X):
        return self._delta(X, self.omit)

    def inverse_transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        new_X = pd.concat([self.biases_, X])
        new_X = new_X.cumsum(axis=0)
        if self.omit is not None:
            for colname in self.omit:
                new_X[colname] = X[colname]
                new_X[colname].iloc[0] = self.biases_[colname].iloc[0]
        return new_X
