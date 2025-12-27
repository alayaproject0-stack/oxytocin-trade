import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from typing import List

from data_loader import DataFetcher
from features import FeatureEngineer
from dataset import StockDataset
from models_snn import SNNClassifier

def train_model(
    ticker_key: str = "BTC",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    window_size: int = 20,
    hidden_dim: int = 128
):
    # 1. Fetch and Prepare Data
    fetcher = DataFetcher()
    engineer = FeatureEngineer()
    
    df = fetcher.fetch_data(ticker_key, period="5y")
    if df is None:
        print("Failed to fetch data.")
        return

    # Add technical indicators
    df = engineer.add_technical_indicators(df)
    
    # Define features to use
    feature_cols = ['RSI', 'BB_Width', 'Return', 'Log_Return', 'Volume_Change']
    
    # Normalize features
    df_norm = engineer.normalize_data(df, feature_cols)
    
    # Create Dataset and DataLoader
    dataset = StockDataset(df_norm, feature_cols, window_size=window_size)
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Initialize Model
    in_dim = len(feature_cols) * window_size # Flattened input
    num_classes = 2 # Up / Down
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SNNClassifier(in_dim=in_dim, hidden=hidden_dim, n_classes=num_classes).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Starting training on {device} for {ticker_key}...")
    
    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_spikes = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            logits, mean_spikes = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            total_spikes += mean_spikes
            
        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                logits, _ = model(data)
                pred = logits.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                
        avg_loss = train_loss / len(train_loader)
        train_acc = train_correct / len(train_dataset)
        val_acc = val_correct / len(val_dataset)
        avg_spikes = total_spikes / len(train_loader)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Spikes: {avg_spikes:.2f}")

    # 4. Save Model
    model_path = "snn_model_latest.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
