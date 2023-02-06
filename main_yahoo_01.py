import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Stock symbol
symbol = "CDA.AX"

# Get daily stock data
df = yf.download(symbol, start="2010-01-01", end="2023-02-06", interval="1d")

# Calculate Moving Averages
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()

# Generate signals
df['Signal'] = np.where(df['SMA_50'] > df['SMA_200'], 1, 0)
df['Signal'] = np.where(df['SMA_50'] < df['SMA_200'], -1, df['Signal'])

# Plot the stock price and Moving Averages
plt.plot(df['Close'], label='Stock Price')
plt.plot(df['SMA_50'], label='SMA 50')
plt.plot(df['SMA_200'], label='SMA 200')
plt.legend(loc='best')
plt.show()
