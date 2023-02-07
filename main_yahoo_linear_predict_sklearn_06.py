import yfinance as yf
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import pandas as pd

# Stock symbol
symbol = "CBA.AX"

# Get daily stock data
df = yf.download(symbol, start="2010-01-01", end="2023-02-06", interval="1d")

# Calculate Moving Averages
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()

# Generate signals
df['Signal'] = np.where(df['SMA_50'] > df['SMA_200'], 1, 0)
df['Signal'] = np.where(df['SMA_50'] < df['SMA_200'], -1, df['Signal'])

# Get the stock price and set it as the target
target = df['Close'].values

# Get the difference in days between each date and a reference date
reference_date = np.datetime64('2010-01-01')
dates = df.index.values
dates_X = (dates - reference_date) / np.timedelta64(1, 'D')
dates_X = dates_X.reshape(-1, 1)

# Scale the data
scaler = StandardScaler()
dates_X = scaler.fit_transform(dates_X)

# Train a neural network to predict the stock price
reg = MLPRegressor(hidden_layer_sizes=(100,), max_iter=10000, activation='relu')
reg.fit(dates_X, target)

# Prepare the input data by dropping non-numeric columns and missing values
df_reg = df[['Close']].copy().dropna()

# Generate predictions for the next 100 days
reference_date = df_reg.index[0]
future_dates = [df_reg.index[-1] + pd.Timedelta(days=x) for x in range(1, 101)]
future_dates_X = []
for date in future_dates:
    diff = date - reference_date
    diff_days = diff.total_seconds() / (24 * 3600)
    future_dates_X.append([diff_days])
future_dates_X = np.array(future_dates_X)
future_dates_X = scaler.transform(future_dates_X)
future_predictions = reg.predict(future_dates_X)

# Plot the stock price, Moving Averages, and prediction using plotly
fig = px.line(df, x=df.index, y='Close', title='Stock Price')
fig.add_scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50')
fig.add_scatter(x=df.index, y=df['SMA_200'], mode='lines', name='SMA 200')
fig.add_scatter(x=future_dates, y=future_predictions, mode='lines', name='Prediction')
fig.show()
