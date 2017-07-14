import numpy as np
from tseries.tseries import DeltaTransformer
import pandas as pd

def mse(X1, X2, reduce_mean=True):
    if X1.shape != X2.shape:
        raise ValueError("shapes of x1: {} and x2: {} should be identical".format(X1.shape,X2.shape))
    if not reduce_mean:
        return np.mean((X1 - X2) ** 2, axis=0) ** .5
    else:
        return np.mean(np.mean((X1 - X2) ** 2, axis=0) ** .5)

def one_step_prediction(Y_hat, Y_true):
    Y_hat_delta = DeltaTransformer().transform(Y_hat)
    return Y_hat_delta + Y_true[0:-1]

def names_to_indicies(df, names):
    name_to_index = {v: i for (i, v) in enumerate(df.columns.values)}
    indicies = [name_to_index[n] for n in names]
    return indicies

