# SPY Momentum Strategy
# Author: Your Name
# Description: A simple moving average crossover strategy on SPY (2020-2024)
# Compares 50-day and 200-day SMAs to generate buy/sell signals
# Benchmarked against buy and hold with cumulative returns and Sharpe Ratio

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf   
ticker = 'SPY'
data = yf.download(ticker, start='2020-01-01', end='2024-01-01')

print(data.head())
print(data.shape)

# Calculate 50-day and 200-day moving averages
data["SMA50"] = data["Close"].rolling(window=50).mean()
data["SMA200"] = data["Close"].rolling(window=200).mean()

print(data[["Close", "SMA50", "SMA200"]].tail(10))

# Generate buy and sell signals
data["Signal"] = 0
data.loc[data["SMA50"] > data["SMA200"], "Signal"] = 1
data.loc[data["SMA50"] < data["SMA200"], "Signal"] = -1

print(data[["Close", "SMA50", "SMA200", "Signal"]].tail(10))

print(data["Signal"].value_counts())

# Plot the strategy
plt.figure(figsize=(14, 7))
plt.plot(data["Close"], label="SPY Price", alpha=0.5)
plt.plot(data["SMA50"], label="50-Day SMA", alpha=0.8)
plt.plot(data["SMA200"], label="200-Day SMA", alpha=0.8)

# Plot buy signals
plt.scatter(data[data["Signal"] == 1].index, 
            data[data["Signal"] == 1]["Close"], 
            marker="^", color="green", label="Buy", alpha=0.5, s=10)

# Plot sell signals
plt.scatter(data[data["Signal"] == -1].index, 
            data[data["Signal"] == -1]["Close"], 
            marker="v", color="red", label="Sell", alpha=0.5, s=10)

plt.title("SPY Momentum Strategy")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()

# Calculate strategy returns
data["Daily_Return"] = data["Close"].pct_change()
data["Strategy_Return"] = data["Daily_Return"] * data["Signal"].shift(1)

# Cumulative returns
data["Cumulative_Market"] = (1 + data["Daily_Return"]).cumprod()
data["Cumulative_Strategy"] = (1 + data["Strategy_Return"]).cumprod()

# Plot performance comparison
plt.figure(figsize=(14, 7))
plt.plot(data["Cumulative_Market"], label="Buy and Hold SPY", alpha=0.8)
plt.plot(data["Cumulative_Strategy"], label="Momentum Strategy", alpha=0.8)
plt.title("Momentum Strategy vs Buy and Hold")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()


# Print final performance metrics
total_market = data["Cumulative_Market"].iloc[-1] - 1
total_strategy = data["Cumulative_Strategy"].iloc[-1] - 1

print(f"Buy and Hold Return: {total_market:.2%}")
print(f"Momentum Strategy Return: {total_strategy:.2%}")

# Sharpe Ratio (assumes 252 trading days per year, 0% risk free rate for simplicity)
sharpe_market = (data["Daily_Return"].mean() / data["Daily_Return"].std()) * np.sqrt(252)
sharpe_strategy = (data["Strategy_Return"].mean() / data["Strategy_Return"].std()) * np.sqrt(252)

print(f"Buy and Hold Sharpe Ratio: {sharpe_market:.2f}")
print(f"Momentum Strategy Sharpe Ratio: {sharpe_strategy:.2f}")