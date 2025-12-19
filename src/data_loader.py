import yfinance as yf
import pandas as pd
from typing import Optional, List

class DataFetcher:
    def __init__(self):
        self.tickers = {
            "TDK": "6762.T",
            "TEL": "8035.T",
            "ADVANTEST": "6857.T"
        }

    def fetch_data(self, ticker_key: str, period: str = "2y", interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Fetches historical data for a given ticker key (e.g., 'TDK').
        
        Args:
            ticker_key: Key from self.tickers (e.g. 'TDK', 'TEL')
            period: Data period to download (e.g. '1y', '2y', 'max')
            interval: Data interval (e.g. '1d', '1h')
            
        Returns:
            pd.DataFrame with OHLCV data or None if failed.
        """
        symbol = self.tickers.get(ticker_key)
        if not symbol:
            print(f"Error: Ticker key '{ticker_key}' not found.")
            return None

        print(f"Fetching data for {symbol}...")
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            if df.empty:
                print("Warning: No data found.")
                return None
            
            # Ensure index is datetime
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

if __name__ == "__main__":
    fetcher = DataFetcher()
    df = fetcher.fetch_data("TDK")
    if df is not None:
        print(df.head())
        print(df.tail())
