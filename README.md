# tseries

Classes for working with time series in python using the sklearn ecosystem. Contains:

- TimeSeriesRegressor: Turn any sklearn estimator into a time series estimator. Eg Linear Regression -> Vector Auto Regression (VAR)
- Delta Transformer: Transform your time series to predict additive changes. When pipelining this one can create "Integrated"  time series models like the VARI
- Double Pipeline: Add invertable transformations to your labels so that you can regress in different spaces and pull back to the original
- Functions for proper cross validation of time series.
- Other utilities ive found useful 

Documentation is currently lagging behind development and pull requests are greatly appreciated! The latest stable release can be installed with:

`pip install tseries`


## Requires
Numpy, Pandas, SciKit-Learn,pickle

## Usage

To make a predictor of the stock market that maps the previous two days of the s&p500 stock prices and 
predicts the next day's price of AAPL stock try the following:
```
from TimeSeriesEstimator import TimeSeriesRegressor, time_series_split
from sklearn.linear_model import LinearRegression,Lasso
from utils import datasets


X = datasets('sp500')
y = X['AAPL']
X_train, X_test = time_series_split(X)
y_train, y_test = time_series_split(y)


n_prev=2
tsr = TimeSeriesRegressor(Lasso(), n_prev=n_prev)
tsr.fit(X_train, y_train)
pred_train = tsr.predict(X_train) #outputs a numpy array of length: len(X_train)-n_prev
pred_test = tsr.predict(X_test)
```

To forecast all stocks in the s&p500 100 days into the future use the forecast method:

```
tsr = TimeSeriesRegressor(LinearRegression(), n_prev=2)
tsr.fit(X_train)
fc = tsr.forecast(X_train, 100)
```
See the ipython notebook for a longer interactive example!

## Install
Clone this repo and call directly as a module. Have not added automatic install support yet.

##Mechanics

The TSR works by taking in a single (X) or two datasets (X,Y) of equal length. 
In the single dataset case, the TSR assumes you would like to predict the next element in the dataset using the previous elements.
In either case it forms a dataset by taking the previous n timesteps and flattening them into a vector. 

<table>
 <caption>Dataset X</caption>
<tr>
<th>Feature 1</th>
<th>Feature 2</th>
</tr>
<tr>
<td> 1</td>
<td> 1.5</td>
</tr>
<tr>
<td>2</td>
<td>2.5</td>
</tr>
<tr>
<td>3</td>
<td>3.5</td>
</tr>
<tr>
<td>4</td>
<td>4.5</td>
</tr>
<tr>
<td>5</td>
<td>5.5</td>
</tr>
</table>


<table>
<table style="float: left;">
 <caption>New X with n_prev = 2</caption>
<tr>
<th>Feature 1</th>
<th>Feature 2</th>
<th>Feature 3</th>
<th>Feature 4</th>
</tr>
<tr>
<td> 1</td>
<td> 1.5</td>
<td>2</td>
<td>2.5</td>
</tr>
<tr>
<td>2</td>
<td>2.5</td>
<td>3</td>
<td>3.5</td>
</tr>
<tr>
<td>3</td>
<td>3.5</td>
<td>4</td>
<td>4.5</td>
</tr>
</table>



<table>
<table style="float: middle-left;">
 <caption>New Y with n_prev = 2</caption>
<tr>
<th>Feature 1</th>
<th>Feature 2</th>
</tr>
<tr>
<td>3</td>
<td>3.5</td>
</tr>
<tr>
<td>4</td>
<td>4.5</td>
</tr>
<tr>
<td>5</td>
<td>5.5</td>
</tr>
</table>



Now the X and Y datasets can be fit by any regression technique in sklearn.
If the technique cannot handle vectors as outputs, use the "parallel_models" input to predict
each feature sequence with its own multi to single dim regressor.

