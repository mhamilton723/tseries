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

for n_prev in range(1, 5):
    tsr = TimeSeriesRegressor(LinearRegression(), n_prev=n_prev)
    tsr.fit(X_train)
    fc = tsr.forecast(X_train, len(X_test))

    def changes(X, start=0,end=-1):
        return np.array([X[end, i] - X[start, i] for i in range(X.shape[1])])

    n_aheads = range(1,len(X_test),1)
    X_test_changes = [changes(X_test,end=n_ahead) for n_ahead in n_aheads]
    fc_changes = [changes(fc,end=n_ahead) for n_ahead in n_aheads]
    mses = [mse(X_test_changes[i],fc_changes[i]) for i in range(len(n_aheads))]
    cors = [np.corrcoef(X_test_changes[i],fc_changes[i])[0,1] for i in range(len(n_aheads))]
    plt.plot(n_aheads,cors, label="n_prev={}".format(n_prev))

plt.legend()
plt.show()