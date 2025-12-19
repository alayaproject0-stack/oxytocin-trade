import pandas as pd
import numpy as np
import ta

class FeatureEngineer:
    def __init__(self):
        pass

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds RSI, Bollinger Bands, and Returns to the DataFrame.
        """
        df_copy = df.copy()
        
        # Ensure correct 'Close' column.
        # yfinance often returns MultiIndex columns or DataFrame for single column access depending on version
        close = None
        if 'Close' in df_copy.columns:
            close = df_copy['Close']
        else:
            # Fallback (Open, High, Low, Close...)
            close = df_copy.iloc[:, 3]
            
        # Squeeze to Series if it is a DataFrame (e.g. Shape (N, 1))
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        
        # If still DataFrame (e.g. if squeeze failed or wasn't 1D), force selection
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        # 1. RSI (Relative Strength Index) - 14 periods
        df_copy['RSI'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

        # 2. Bollinger Bands - 20 periods, 2 std dev
        indicator_bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        df_copy['BB_High'] = indicator_bb.bollinger_hband()
        df_copy['BB_Low'] = indicator_bb.bollinger_lband()
        df_copy['BB_Width'] = (df_copy['BB_High'] - df_copy['BB_Low']) / df_copy['BB_Low']

        # 3. Simple Returns (Rate of Change)
        df_copy['Return'] = close.pct_change()
        
        # 4. Log Returns
        df_copy['Log_Return'] = np.log(close / close.shift(1))

        # 5. Volume Change
        # Similar logic for Volume
        volume = None
        if 'Volume' in df_copy.columns:
            volume = df_copy['Volume']
            if isinstance(volume, pd.DataFrame):
                volume = volume.squeeze()
            if isinstance(volume, pd.DataFrame):
                volume = volume.iloc[:, 0]
            df_copy['Volume_Change'] = volume.pct_change()
        else:
             df_copy['Volume_Change'] = 0.0

        # Drop NaN values created by window functions
        df_clean = df_copy.dropna()
        
        return df_clean

    def normalize_data(self, df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
        """
        Min-Max normalization for SNN input [0, 1].
        """
        df_norm = df.copy()
        for col in feature_columns:
            if col in df_norm.columns:
                # Need to handle if col is still a DataFrame (unlikely after squeeze but possible in dataframe ops)
                series = df_norm[col]
                if isinstance(series, pd.DataFrame):
                    series = series.squeeze()
                
                min_val = series.min()
                max_val = series.max()
                
                if max_val - min_val != 0:
                    df_norm[col] = (series - min_val) / (max_val - min_val)
                else:
                    df_norm[col] = 0.0
        return df_norm

if __name__ == "__main__":
    # Mock data for testing
    data = {'Close': np.random.rand(100) * 100, 'Volume': np.random.randint(100, 1000, 100)}
    df = pd.DataFrame(data)
    
    fe = FeatureEngineer()
    df_features = fe.add_technical_indicators(df)
    print(df_features.head())
