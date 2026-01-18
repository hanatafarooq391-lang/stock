import yfinance as yf
import pandas as pd

# Teen alag tarah ke tickers
tickers = ["AAPL", "TSLA", "BTC-USD"]

def fetch_multi_data():
    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        # 3 saal ka data le rahe hain taake patterns milein
        data = yf.download(ticker, start="2021-01-01", end="2024-01-01")
        
        if not data.empty:
            # Har stock ki alag CSV banegi
            filename = f"{ticker.replace('-', '_')}_data.csv"
            data.to_csv(filename)
            print(f"Saved: {filename}")
        else:
            print(f"Failed to download {ticker}")

if __name__ == "__main__":
    fetch_multi_data()