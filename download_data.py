# download_data.py

from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import os
import time

# Your Alpha Vantage API key
API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"  # ⬅️ Replace this with your actual API key

# List of tickers to download
TICKERS = ['AAPL', 'TSLA', 'JPM', 'JNJ', 'XOM']

# Create directory to save CSVs
CACHE_DIR = "data"
os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize Alpha Vantage TimeSeries
ts = TimeSeries(key=API_KEY, output_format='pandas')

# Function to download and save data
def download_ticker(ticker):
    print(f"📥 Downloading {ticker}...")
    try:
        data, _ = ts.get_daily(symbol=ticker, outputsize='full')
        data = data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        data.sort_index(inplace=True)
        filepath = os.path.join(CACHE_DIR, f"{ticker}.csv")
        data.to_csv(filepath)
        print(f"✅ Saved to {filepath}")
    except Exception as e:
        print(f"❌ Error downloading {ticker}: {e}")

# Download each ticker, respecting the 5-per-minute limit
for i, ticker in enumerate(TICKERS):
    download_ticker(ticker)
    if (i + 1) % 5 == 0 and i != len(TICKERS) - 1:
        print("⏳ Waiting 60 seconds to respect API rate limits...")
        time.sleep(60)
