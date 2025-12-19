import torch
from torch.utils.data import DataLoader, random_split
from src.data_loader import DataFetcher
from src.features import FeatureEngineer
from src.dataset import StockDataset
from src.models_snn import SNNClassifier
import numpy as np
import os

def train_sys1():
    print("=== Starting Phase 2: System 1 (SNN) Training ===")
    
    # 1. Load Data
    fetcher = DataFetcher()
    ticker = "TDK"
    print(f"Fetching data for {ticker}...")
    df = fetcher.fetch_data(ticker, period="2y") # 2 years of daily data
    
    if df is None:
        print("Error: No data.")
        return

    # 2. Features
    print("Processing features...")
    fe = FeatureEngineer()
    df = fe.add_technical_indicators(df)
    
    # Select features for SNN
    features = ['RSI', 'BB_Width', 'Return', 'Log_Return', 'Volume_Change']
    # Normalize
    df_norm = fe.normalize_data(df, features)
    
    # 3. Create Dataset
    print("Creating Dataset...")
    window_size = 20
    dataset = StockDataset(df_norm, features, window_size=window_size)
    print(f"Total samples: {len(dataset)}")
    
    # Split Train/Test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # 4. Initialize SNN
    # Input Dim = window_size * feature_count
    input_dim = window_size * len(features)
    hidden_dim = 128
    n_classes = 2 # Up / Down
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SNNClassifier(in_dim=input_dim, hidden=hidden_dim, n_classes=n_classes, steps=20).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # 5. Training Loop
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, mean_spikes = model(X)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
        avg_loss = total_loss / len(train_loader)
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
    
    # 6. Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            logits, _ = model(X)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    test_acc = 100 * correct / total
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    
    # Save Model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/sys1_snn.pth")
    print("Model saved to models/sys1_snn.pth")

if __name__ == "__main__":
    train_sys1()
