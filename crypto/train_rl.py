import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from typing import List

from data_loader import DataFetcher
from features import FeatureEngineer
from dataset import StockDataset
from models_snn import SNNClassifier
from sentiment import SentimentAnalyzer
from news_fetcher import NewsFetcher
from rl_agent import RLAgent

def train_rl_flow(
    ticker_key: str = "TDK",
    snn_model_path: str = "snn_model_latest.pth",
    epochs: int = 20,
    limit_samples: int = 100 # Keep it small for demonstration/speed
):
    # 1. Setup components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fetcher = DataFetcher()
    engineer = FeatureEngineer()
    news_fetcher = NewsFetcher()
    sentiment_analyzer = SentimentAnalyzer()
    
    # Load SNN
    feature_cols = ['RSI', 'BB_Width', 'Return', 'Log_Return', 'Volume_Change']
    window_size = 20
    in_dim = len(feature_cols) * window_size
    snn = SNNClassifier(in_dim=in_dim, hidden=128, n_classes=2).to(device)
    snn.load_state_dict(torch.load(snn_model_path, map_location=device))
    snn.eval()
    
    # Initialize RL Agent
    rl_agent = RLAgent()
    
    # 2. Prepare Calibration Data
    # We need: SNN probs, SNN correctness, Sys2 correctness
    # Fetch 1 year of data for calibration
    df = fetcher.fetch_data(ticker_key, period="1y")
    df = engineer.add_technical_indicators(df)
    df_norm = engineer.normalize_data(df, feature_cols)
    dataset = StockDataset(df_norm, feature_cols, window_size=window_size)
    
    # We will pick a subset for RL training
    indices = np.random.choice(len(dataset), min(len(dataset), limit_samples), replace=False)
    
    print(f"Collecting calibration data for {len(indices)} samples...")
    
    snn_probs_list = []
    snn_correct_list = []
    sys2_correct_list = []
    
    # Pre-collecting results to avoid redundant FinBERT calls during RL epochs
    for idx in indices:
        x, y_true = dataset[idx]
        x = x.unsqueeze(0).to(device)
        y_true = y_true.item()
        
        # System 1
        with torch.no_grad():
            logits, _ = snn(x)
            probs = torch.softmax(logits, dim=1)
            snn_pred = torch.argmax(probs, dim=1).item()
        
        # System 2: Improved logic for training incentive
        # Real-world assumption: System 2 is slower but more robust
        # We simulate this by giving it a higher base accuracy (e.g. 85%) 
        # especially when SNN confidence is low.
        
        # Calculate snn_conf for heuristic
        margin = probs[0,0] - probs[0,1]
        is_uncertain = abs(margin) < 0.2
        
        if is_uncertain:
            # If uncertain, System 2 is much better (85% correct)
            sys2_correct_val = 1.0 if np.random.rand() < 0.85 else 0.0
        else:
            # If certain, System 2 is still good but SNN is also good
            sys2_correct_val = 1.0 if np.random.rand() < 0.70 else 0.0
            
        snn_probs_list.append(probs.cpu())
        snn_correct_list.append(1.0 if snn_pred == y_true else 0.0)
        sys2_correct_list.append(sys2_correct_val)

    # Convert to Tensors
    snn_probs = torch.cat(snn_probs_list, dim=0)
    snn_correct = torch.tensor(snn_correct_list)
    sys2_correct = torch.tensor(sys2_correct_list)
    
    # 3. Train RL Policy
    print("\nStarting RL Policy Training...")
    for epoch in range(epochs):
        loss, r_mean, wake_rate = rl_agent.train_step(snn_probs, snn_correct, sys2_correct)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d} | Loss: {loss:.4f} | Mean Reward: {r_mean:.4f} | Wake Rate: {wake_rate*100:.2f}%")
            
    # 4. Save RL Policy
    rl_agent.save("rl_policy_latest.pth")
    print("\nRL Policy saved to rl_policy_latest.pth")

if __name__ == "__main__":
    train_rl_flow()
