from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from sklearn.linear_model.base import LinearRegression
from sklearn.ensemble.forest import RandomForestRegressor


class AlternatingLearner(BaseEstimator, RegressorMixin):
    def __init__(self,
                 cols_1,
                 cols_2,
                 estimator_1=LinearRegression(fit_intercept=False),
                 estimator_2=RandomForestRegressor(),
                 iters=2):
        self.cols_1 = cols_1
        self.cols_2 = cols_2
        self.estimator_1 = estimator_1
        self.estimator_2 = estimator_2
        self.iters = iters

    def _normalize_x(self, X):
        if len(X.shape) < 2:
            X = np.expand_dims(X, 1)
        return np.array(X)

    def _normalize_y(self, y):
        return np.squeeze(y)


    def fit(self, X, y):
        X = self._normalize_x(X)
        y = self._normalize_y(y)

        for i in range(self.iters):
            if i == 0:
                target_1 = y
            else:
                target_1 = y - self.estimator_2.predict(X[:, self.cols_2])
            self.estimator_1 = self.estimator_1.fit(X[:, self.cols_1], target_1)

            target_2 = y - self.estimator_1.predict(X[:, self.cols_1])
            self.estimator_2 = self.estimator_2.fit(X[:, self.cols_2], target_2)

        return self

    def predict(self, X):
        X = self._normalize_x(X)
        return self.estimator_1.predict(X[:, self.cols_1]) + \
               self.estimator_2.predict(X[:, self.cols_2])
