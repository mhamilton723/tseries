from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from sklearn.linear_model.base import LinearRegression


class DoubleLearner(BaseEstimator, RegressorMixin):
    def __init__(self,
                 treatment_cols,
                 nusiance_cols,
                 effect_estimator=LinearRegression(fit_intercept=False),
                 treatment_estimator=LinearRegression(fit_intercept=False),
                 y_estimator=LinearRegression(fit_intercept=False)):
        self.nusiance_cols = nusiance_cols
        self.treatment_cols = treatment_cols
        self.effect_estimator = effect_estimator
        self.treatment_estimator = treatment_estimator
        self.y_estimator = y_estimator

    def _normalize_x(self, X):
        if len(X.shape) < 2:
            X = np.expand_dims(X, 1)
        return np.array(X)

    def _normalize_y(self, y):
        return np.squeeze(y)

    def fit(self, X, y):
        X = self._normalize_x(X)
        y = self._normalize_y(y)

        self.y_estimator = self.y_estimator \
            .fit(X[:, self.nusiance_cols], y)
        y_resid = y - self.y_estimator.predict(X[:, self.nusiance_cols])

        self.treatment_estimator = self.treatment_estimator \
            .fit(X[:, self.nusiance_cols], X[:, self.treatment_cols])
        treatment_resid = X[:, self.treatment_cols] - self.treatment_estimator.predict(X[:, self.nusiance_cols])

        self.effect_estimator = self.effect_estimator.fit(treatment_resid, y_resid)

        return self

    def predict(self, X):
        X = self._normalize_x(X)
        treatment_resid = X[:, self.treatment_cols] - self.treatment_estimator.predict(X[:, self.nusiance_cols])
        return self.effect_estimator.predict(treatment_resid) + \
               self.y_estimator.predict(X[:, self.nusiance_cols])
