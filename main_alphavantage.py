import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Alpha Vantage API key
api_key = "MCX4QMLTMOLZ98QA"

# Stock symbol
#symbol = "ASX:CBA"
symbol = "IBM"

# Get daily stock data
url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={api_key}"

#url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={api_key}'

try:
    response = requests.get(url)
    data = response.json()
    time_series_key = 'Time Series (Daily)'
    if time_series_key not in data:
        raise KeyError(f"Key '{time_series_key}' not found in response.")
except Exception as e:
    print(f"An error occurred: {e}")

# Clean the data and convert it to a Pandas dataframe
df = pd.DataFrame(data[time_series_key])
df = df.transpose()
df.index = pd.to_datetime(df.index)
df = df.astype(float)

# Calculate Moving Averages
df['SMA_50'] = df['4. close'].rolling(window=50).mean()
df['SMA_200'] = df['4. close'].rolling(window=200).mean()

# Generate signals
df['Signal'] = np.where(df['SMA_50'] > df['SMA_200'], 1, 0)
df['Signal'] = np.where(df['SMA_50'] < df['SMA_200'], -1, df['Signal'])

# Plot the stock price and Moving Averages
plt.plot(df['4. close'], label='Stock Price')
plt.plot(df['SMA_50'], label='SMA 50')
plt.plot(df['SMA_200'], label='SMA 200')
plt.legend(loc='best')
plt.show()

