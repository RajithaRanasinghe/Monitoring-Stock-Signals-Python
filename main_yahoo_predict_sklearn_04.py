import yfinance as yf
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
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

# Create a linear regression model
reg = LinearRegression()

# Prepare the input data by dropping non-numeric columns and missing values
df_reg = df[['Close']].dropna()

# Define the target variable and convert it to a numpy array
target = df_reg['Close'].values

# Define the input variable and convert it to a numpy array
#X = np.array(df_reg.index).reshape(-1, 1)
X = pd.to_numeric(df_reg.index, errors='coerce').values.reshape(-1, 1)


# Fit the model
reg.fit(X, target)

# Predict the next 30 days
reference_date = df_reg.index[0]
future_dates = [df_reg.index[-1] + pd.Timedelta(days=x) for x in range(1, 101)]
future_dates_X = []
for date in future_dates:
    diff = date - reference_date
    diff_days = diff.total_seconds() / (24 * 3600)
    future_dates_X.append([diff_days])
future_dates_X = np.array(future_dates_X)
future_predictions = reg.predict(future_dates_X)




# Plot the stock price, Moving Averages, and predictions
fig = px.line(df, x=df.index, y='Close', title='Stock Price')
fig.add_scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50')
fig.add_scatter(x=df.index, y=df['SMA_200'], mode='lines', name='SMA 200')
fig.add_scatter(x=future_dates, y=future_predictions, mode='lines', name='Prediction')

# Add zoom in/out functionality using mouse wheel
updatemenus=[
    dict(
        buttons=list([
            dict(
                args=[{"yaxis.type": "linear"}],
                label="Linear",
                method="relayout"
            ),
            dict(
                args=[{"yaxis.type": "log"}],
                label="Log",
                method="relayout"
            )
        ]),
        direction="down",
        pad={"r": 10, "t": 10},
        showactive=True,
        x=0.1,
        xanchor="left",
        y=1.1,
        yanchor="top"
    ),
]

fig.update_layout(updatemenus=updatemenus)
fig.show()