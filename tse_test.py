import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from utils import datasets, safe_shape
from TimeSeriesEstimator import TimeSeriesRegressor, time_series_split, time_series_cv, cascade_cv
from sklearn.cross_validation import KFold

def mse(X1, X2, multioutput='raw_values'):
    if multioutput == 'raw_values':
        return np.mean((X1 - X2)**2, axis=0)**.5
    if multioutput == 'uniform_average':
        return np.mean(np.mean((X1 - X2)**2, axis=0)**.5)


X = datasets('sp500')
names = list(X.columns.values)
X_train, X_test = time_series_split(X)

n_prev = 2
tsr = TimeSeriesRegressor(LinearRegression(), n_prev=n_prev)
tsr.fit(X_train)
fc = tsr.forecast(X_train, len(X_test), noise=.2, n_paths=200)
fc_mean = tsr.forecast(X_train, len(X_test), noise=.2, n_paths=200, combine='mean')
#or for speed
#fc_mean = np.mean(fc, axis=0)

plt.plot(np.transpose(fc[:, :, 1]), 'r', alpha=.05)
plt.plot(np.transpose(fc_mean[:, 1]), 'b', label='Mean Forecast')
plt.plot(X_test[:,  1], 'g', label='Actual Price')
plt.legend()
plt.xlabel('days')
plt.ylabel('Price')
plt.title('Forecasting Alcoa (AA)')
plt.show()
