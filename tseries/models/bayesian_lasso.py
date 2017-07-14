from sklearn.utils.validation import check_X_y
from sklearn.base import BaseEstimator, RegressorMixin
import pymc
import numpy as np


class BayesianLasso(BaseEstimator, RegressorMixin):
    def __init__(self, b=1.0, use_mcmc=False,
                 mcmc_burn=5000, mcmc_trials=10000,
                 feature_map=None, feature_weights=None):
        self.b = b
        self.use_mcmc = use_mcmc
        self._map = None
        self.coef_ = None
        self._confidence = None
        self.mcmc = None
        self.num_betas = None
        self.mcmc_burn = mcmc_burn
        self.mcmc_trials = mcmc_trials
        self.feature_map = feature_map
        self.feature_weights = feature_weights

    def get_coefficients(self):
        return [{str(variable): variable.value} for variable in self._map.variables
                if str(variable).startswith('beta')]

    def lasso_model(self, X, y):

        if self.feature_map is None:
            self.feature_map = {i: i for i in range(X.shape[1])}
        if self.feature_weights is None:
            self.feature_weights = {i: 1.0 for i in range(X.shape[1])}

        # Priors for unknown model parameters
        betas = []
        for i in range(max(self.feature_map.values()) + 1):
            betas.append(pymc.Laplace('beta_{}'.format(i), mu=0, tau=1.0 / self.b))

        @pymc.deterministic
        def y_hat_lasso(betas=betas, X=X):
            return sum(betas[self.feature_map[i]] *
                       self.feature_weights[i] * X[:, i]
                       for i in range(X.shape[1]))

        tau = pymc.Uniform('tau', lower=0.001, upper=100.0)
        Y_lasso = pymc.Normal('Y', mu=y_hat_lasso, tau=tau, value=y, observed=True)
        return locals()

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True, force_all_finite=False)

        if self.use_mcmc:
            self.mcmc = pymc.MCMC(self.lasso_model(X, y))
            self.mcmc.sample(self.mcmc_trials, self.mcmc_burn, 2)
            if self.feature_map is not None:
                self.num_betas = max(self.feature_map.values()) + 1
            else:
                self.num_betas = X.shape[1]

            traces = []
            for i in range(self.num_betas):
                traces.append(self.mcmc.trace('beta_{}'.format(i))[:])

            self.coef_ = np.array([np.mean(trace) for trace in traces])
        else:
            self._map = pymc.MAP(self.lasso_model(X, y))
            self._map.fit()
            self.coef_ = np.array([beta.value for beta in self._map.betas])

    def confidence_intervals(self, confidence=95.):
        if self.mcmc is None:
            raise ValueError('Need to fit the mcmc first')

        traces = []
        uppers = []
        lowers = []
        cutoff = (100 - confidence) / 2
        for i in range(self.num_betas):
            traces.append(self.mcmc.trace('beta_{}'.format(i))[:])
            uppers.append(np.percentile(traces[i], 100. - cutoff))
            lowers.append(np.percentile(traces[i], cutoff))

        return np.array(lowers), np.array(uppers)

    def predict(self, X):
        mapping = np.array(
            [self.coef_[self.feature_map[i]] *
             self.feature_weights[i]
             for i in range(len(self.feature_map))])
        return np.dot(X, mapping)
