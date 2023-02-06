import yfinance as yf
import numpy as np
import plotly.express as px

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

# Plot the stock price and Moving Averages using plotly
fig = px.line(df, x=df.index, y='Close', title='Stock Price')
fig.add_scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50')
fig.add_scatter(x=df.index, y=df['SMA_200'], mode='lines', name='SMA 200')

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
