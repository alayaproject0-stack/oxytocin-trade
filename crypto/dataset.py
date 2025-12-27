import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class StockDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: list, window_size: int = 20, prediction_horizon: int = 1):
        """
        Args:
            df: Normalized DataFrame with all indicators.
            feature_cols: List of column names to use as features.
            window_size: Number of past days to look at (N).
            prediction_horizon: How many days ahead to predict (usually 1).
        """
        self.df = df
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        
        # Pre-compute X and y to avoid overhead during training
        self.X, self.y = self._prepare_data()

    def _prepare_data(self):
        data = self.df[self.feature_cols].values
        # Create target: Close price direction
        # If Close[t+1] > Close[t], then 1 (Up), else 0 (Down/Stay)
        close_prices = self.df['Close'].values
        
        X_list = []
        y_list = []
        
        # Loop through valid start indices
        # We need `window_size` past data AND `prediction_horizon` future data
        for i in range(len(self.df) - self.window_size - self.prediction_horizon + 1):
            # Input: Data from [i : i+window]
            window_data = data[i : i + self.window_size]
            # Flatten inputs: (20, 5) -> (100,)
            X_list.append(window_data.flatten())
            
            # Target: Compare Close at (i+window-1) vs Close at (i+window-1+1)
            current_close = close_prices[i + self.window_size - 1]
            future_close = close_prices[i + self.window_size - 1 + self.prediction_horizon]
            
            label = 1 if future_close > current_close else 0
            y_list.append(label)
            
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
        
        return torch.tensor(X), torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
