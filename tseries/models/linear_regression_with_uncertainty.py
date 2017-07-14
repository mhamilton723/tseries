import scipy
import scipy.stats as sp
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np


class LinearRegressionWithUncertainty(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.model = None
        self.coef_ = None

    def fit(self, X, y):
        self.model = sm.OLS(y, X).fit()
        if len(self.model.params.shape)==2:
            self.coef_ = np.transpose(self.model.params)
        else:
            self.coef_ = np.expand_dims(self.model.params,0)

        return self

    def confidence_intervals(self, confidence=95.):
        confidence = confidence * .01

        unc = self.model.conf_int(1-confidence)
        lows, highs = np.expand_dims(np.transpose(unc[:,0]),0), np.expand_dims(np.transpose(unc[:,1]),0)
        return lows, highs


    def predict(self, X):
        return np.dot(X, np.transpose(self.coef_))
