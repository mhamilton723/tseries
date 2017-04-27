"""
Double Pipeline Class

Author: Mark Hamilton, mhamilton723@gmail.com
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.pipeline import Pipeline


class DoublePipeline(BaseEstimator):
    """
    Base Class for Time Series Estimators
    """

    def __init__(self, x_steps, y_steps):
        self.x_steps = x_steps
        self.x_pipe_ = None
        self.y_steps = y_steps
        self.y_pipe_ = None

    # TODO handle parameter setting
    def set_params(self, **params):
        return super(DoublePipeline, self).set_params(**params)

    def __repr__(self):
        return "DoublePipeline(X_steps={}, y_steps={})".format(
            self.x_steps, self.y_steps)

    def fit(self, X, Y):
        ''' X and Y are datasets in chronological order, or X is a time series '''
        if len(self.y_steps) > 0:
            self.y_pipe_ = Pipeline(self.y_steps).fit(Y)
            Y_trans = self.y_pipe_.transform(Y)
        else:
            Y_trans = Y
        self.x_pipe_ = Pipeline(self.x_steps).fit(X, Y_trans)
        return self

    def predict(self, X):
        Y_trans = self.x_pipe_.predict(X)
        Y = self.y_pipe_.inverse_transform(Y_trans)
        return Y


