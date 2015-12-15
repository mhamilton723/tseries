import pandas.io.data as web
from datetime import datetime
import pickle
import os
import random
import numpy as np
import pandas as pd



def access(dictionary,keys):
    return [dictionary[key] for key in keys]


def safe_shape(array,i):
    try:
        return array.shape[i]
    except IndexError:
        if i > 0:
            return 1
        else:
            raise IndexError


def mse(X1, X2, multioutput='raw_values'):
    if multioutput == 'raw_values':
        return np.mean((X1 - X2)**2, axis=0)**.5
    if multioutput == 'uniform_average':
        return np.mean(np.mean((X1 - X2)**2, axis=0)**.5)


def datasets(name, tickers=None):
    if name == "sp500":
        ##### Real Stock Data
        print('Using sp500 data')
        data = load_s_and_p_data(start="2014-1-1", tickers=tickers)
    elif name == "synthetic":
        ##### Synthetic data for testing purposes
        print('Using Synthetic data')
        values = 10000
        s = pd.Series(range(values))
        noise = pd.Series(np.random.randn(values))
        s = s / 1000 + noise / 100.
        d = {'one': s * s * 100 / values,
             'two': np.sin(s * 10.),
             'three': np.cos(s * 10),
             'four': np.sin(s * s / 10) * np.sqrt(abs(s)+.002)}
        data = pd.DataFrame(d)
    elif name == "jigsaw":
        ##### Easy synthetic data for testing purposes
        print('Using jigsaw data')
        flow = (list(range(1, 10, 1)) + list(range(10, 1, -1))) * 1000
        pdata = pd.DataFrame({"a": flow, "b": flow})
        pdata.b = pdata.b.shift(9)
        data = pdata.iloc[10:] * random.random()  # some noise
    elif name == "linear":
        ##### Easy synthetic data for testing purposes
        print('Using linear data')
        flow = list(range(0, 10000, 2))
        pdata = pd.DataFrame({"a": flow, "b": flow})
        pdata.b = pdata.b + .5
        data = pdata
        #pdata.iloc[10:] * random.random()  # some noise
    elif name == "autocorr":
        ##### Easy synthetic data for testing purposes
        print('Using autocorr data')
        flow1 = gen_linear_seq(1.01, .002)
        flow2 = gen_linear_seq(1.02, .001)
        pdata = pd.DataFrame({"a": flow1, "b": flow2})
        data = pdata

    else:
        raise ValueError('Not a legal dataset name')

    return data


def gen_linear_seq(a, b, N=10000, start=1):
    out = [start]
    for i in range(1, N):
        out.append(out[-1] * a + b)
    return out


def cache(cache_file):
    def cache_decorator(func):
        def func_wrapper(*args, **kwargs):
            if os.path.exists(cache_file):
                loaded_args, loaded_kwargs, loaded_data = pickle.load(open(cache_file, 'r'))
                load_success = True
            else:
                load_success = False
                loaded_args, loaded_kwargs, loaded_data = None, None, None

            if load_success and loaded_args == args and loaded_kwargs == kwargs:
                return loaded_data
            else:
                if args:
                    data = func(*args, **kwargs)
                else:
                    data = func(**kwargs)
                obj = args, kwargs, data
                pickle.dump(obj, open(cache_file, 'w+'))
                return data
        return func_wrapper
    return cache_decorator

@cache("data/stock_data_cache.pkl")
def get_data(tickers, start="2014-1-1", end="2015-11-02"):
    start_time = datetime.strptime(start, "%Y-%m-%d")
    end_time = datetime.strptime(end, "%Y-%m-%d")
    df = web.DataReader(tickers, 'yahoo', start_time, end_time)
    # df = df['Adj Close']
    # df = df.diff()
    # df = df.iloc[1:len(df),:]
    return df

@cache("data/sp500_data_cache.pkl")
def load_s_and_p_data(start="2009-1-1", end="2015-11-02",
                      ticker_names="data/s_and_p_500_names.csv",
                      tickers=None, clean=True, only_close=True):
    if not tickers:
        s_and_p = pd.read_csv(ticker_names)
        tickers = list(s_and_p['Ticker'])
    data = get_data(tickers, start=start, end=end)
    if only_close:
        data = data['Adj Close']
    if clean:
        if only_close:
            data = data.dropna(axis=1)
        else:
            data = data.dropna(axis=2)

    return data

@cache("data/sp500_names_cache.pkl")
def s_and_p_names(start="2009-1-1", end="2015-11-02",
                  ticker_names="data/s_and_p_500_names.csv",
                  clean=True):

    s_and_p = pd.read_csv(ticker_names)
    tickers = list(s_and_p['Ticker'])
    data = get_data(tickers, start=start, end=end)
    if clean:
        data = data.dropna(axis=2)

    return list(data['Adj Close'].columns.values)



def window_dataset(data, n_prev=1):
    """
    data should be pd.DataFrame()
    """
    dlistX, dlistY = [], []
    for i in range(len(data) - n_prev):
        dlistX.append(data.iloc[i:i + n_prev].as_matrix())
        dlistY.append(data.iloc[i + n_prev].as_matrix())
    darrX = np.array(dlistX)
    darrY = np.array(dlistY)
    return darrX, darrY


def masked_dataset(data, n_prev=3, n_masked=2, predict_ahead=1):
    """
	data should be pd.DataFrame()
	"""
    docX, docY = [], []
    for i in range(len(data) - n_prev - n_masked - predict_ahead):
        x = data.iloc[i:i + n_prev].as_matrix()
        x_mask = np.zeros((n_masked, x.shape[1]))
        docX.append(np.concatenate((x, x_mask)))

        y = data.iloc[i + predict_ahead: i + n_prev + n_masked + predict_ahead].as_matrix()
        docY.append(y)
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY


def prediction_dataset(data, n_samples=50, n_ahead=1):
    """
	data should be pd.DataFrame()
	"""
    docX, docY = [], []
    for i in range(len(data) - n_samples - n_ahead):
        x = data.iloc[i:i + n_samples].as_matrix()
        docX.append(x)
        y = data.iloc[i + n_ahead: i + n_samples + n_ahead].as_matrix()
        docY.append(y)
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY


def seq2seq_dataset(data, n_samples=50, n_ahead=50):
    """
	data should be pd.DataFrame()
	"""
    docX, docY = [], []
    for i in range(len(data) - n_samples - n_ahead):
        x = data.iloc[i:i + n_samples].as_matrix()
        docX.append(x)
        y = data.iloc[i + n_samples:i + n_samples + n_ahead].as_matrix()
        docY.append(y)
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY


def test_train_split(df, test_size=.1, splitting_method='prediction', **kwargs):
    """
	This just splits data to training and testing parts
	"""

    ntrn = int(len(df) * (1 - test_size))

    splitting_methods = {'prediction': prediction_dataset,
                         'window': window_dataset,
                         'seq2seq': seq2seq_dataset,
                         'mask': masked_dataset}
    method = splitting_methods[splitting_method]

    X_train, y_train = method(df.iloc[0:ntrn], **kwargs)
    X_test, y_test = method(df.iloc[ntrn:], **kwargs)

    return (X_train, y_train), (X_test, y_test)


def forecast_old(model, seed, n_points=300, percent_noise=.002):
    output = np.empty((n_points, seed.shape[1]))
    values = np.empty((n_points, seed.shape[0], seed.shape[1]))
    values[0, :] = seed

    for i in range(n_points):
        y_pred = model.predict(values[[i], :])[0]
        y_pred = np.array([[y + y * (.5 - random.random()) * percent_noise for y in y_pred]])

        output[i, :] = y_pred
        if i < n_points - 1:
            if len(seed.shape) > 2:
                print(values[i + 1, :].shape, values[i, 1:, :].shape, y_pred.shape)
                values[i + 1, :] = np.hstack((values[i, 1:, :], y_pred))
            else:
                values[i + 1, :] = y_pred

    return output


def forecast(model, seed, n_ahead=300):
    output = np.empty((n_ahead, seed.shape[1]))

    prev = seed
    for i in range(n_ahead):
        pred = model.predict(np.expand_dims(prev, axis=0))
        new_val = pred[0, -1, :]
        output[i, :] = new_val
        # print(seed.shape,new_val.shape)
        prev = np.vstack((seed[:-1, :], new_val))

    return output
