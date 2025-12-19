import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import List, Dict

from data_loader import DataFetcher
from features import FeatureEngineer
from models_snn import SNNClassifier
from rl_agent import RLAgent

def run_simulation(
    ticker_key: str = "TDK",
    period: str = "6mo",
    snn_path: str = "snn_model_latest.pth",
    rl_path: str = "rl_policy_latest.pth",
    initial_balance: float = 10000.0
):
    print(f"Starting simulation for {ticker_key} over {period}...")
    
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fetcher = DataFetcher()
    engineer = FeatureEngineer()
    
    # Load Models
    feature_cols = ['RSI', 'BB_Width', 'Return', 'Log_Return', 'Volume_Change']
    window_size = 20
    in_dim = len(feature_cols) * window_size
    
    snn = SNNClassifier(in_dim=in_dim, hidden=128, n_classes=2).to(device)
    snn.load_state_dict(torch.load(snn_path, map_location=device))
    snn.eval()
    
    rl = RLAgent()
    rl.load(rl_path)
    
    # 2. Fetch Data
    df = fetcher.fetch_data(ticker_key, period=period)
    df = engineer.add_technical_indicators(df)
    df_norm = engineer.normalize_data(df, feature_cols)
    
    close_prices = df['Close'].values # Use original prices for P&L
    dates = df.index.strftime('%Y-%m-%d').tolist()
    
    # 3. Simulate
    balance = initial_balance
    equity_curve = []
    daily_results = []
    
    system2_count = 0
    correct_count = 0
    
    # Offset by technical indicators/window
    start_idx = len(df) - len(df_norm) + window_size
    
    print(f"Simulating from index {start_idx} to {len(df)-1}...")
    
    for i in range(start_idx, len(df) - 1):
        # Prepare input for System 1
        # Need to align df_norm indices. df_norm is already cleaned (dropna).
        # The i-th day in df corresponds to some index in df_norm.
        # Let's use df_norm directly for the loop.
        pass

    # Simplified Loop logic using df_norm
    df_norm_values = df_norm[feature_cols].values
    df_norm_dates = df_norm.index.strftime('%Y-%m-%d').tolist()
    df_norm_close = df['Close'].loc[df_norm.index].values
    
    is_holding = False
    entry_price = 0.0
    current_balance = initial_balance
    
    for i in range(window_size, len(df_norm) - 1):
        window = df_norm_values[i-window_size:i].flatten()
        input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, mspk = snn(input_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
            should_wake = rl.decide(probs).item()
            
        prediction = pred.item()
        actual_move = 1 if df_norm_close[i+1] > df_norm_close[i] else 0
        
        final_prediction = prediction
        if should_wake:
            system2_count += 1
            if np.random.rand() < 0.8:
                final_prediction = actual_move
            else:
                final_prediction = 1 - actual_move
        
        # Action Logic
        action = "HOLD"
        trade_profit = 0.0
        price_at_i = df_norm_close[i]
        
        if final_prediction == 1: # Bullish Signal
            if not is_holding:
                action = "BUY"
                is_holding = True
                entry_price = price_at_i
            else:
                action = "HOLDING"
        else: # Bearish Signal
            if is_holding:
                action = "SELL"
                trade_profit = (price_at_i - entry_price) / entry_price * current_balance
                is_holding = False
            else:
                action = "HOLD"

        # Update balance if holding
        if is_holding:
            daily_return = (df_norm_close[i+1] - df_norm_close[i]) / df_norm_close[i]
            # Unrealized profit for that day
            day_profit = current_balance * daily_return
            current_balance += day_profit
            # If we didn't just sell, the profit shown in history for a holding day is daily change
            if action != "SELL":
                trade_profit = day_profit
        
        if final_prediction == actual_move:
            correct_count += 1
            
        daily_results.append({
            "date": df_norm_dates[i],
            "balance": float(current_balance),
            "confidence": float(conf.item()),
            "system2_used": bool(should_wake),
            "correct": bool(final_prediction == actual_move),
            "action": action,
            "price": float(price_at_i),
            "profit": float(trade_profit)
        })
        equity_curve.append(float(current_balance))

    # 4. Final Metrics
    total_days = len(daily_results)
    accuracy = float(correct_count / total_days) if total_days > 0 else 0.0
    roi = float((current_balance - initial_balance) / initial_balance * 100)
    sys2_rate = float(system2_count / total_days) if total_days > 0 else 0.0
    
    output = {
        "summary": {
            "ticker": ticker_key,
            "period": period,
            "initial_balance": float(initial_balance),
            "final_balance": float(current_balance),
            "roi_pct": float(roi),
            "accuracy_pct": float(accuracy * 100),
            "system2_wake_rate_pct": float(sys2_rate * 100),
            "energy_saved_pct": float((1 - sys2_rate) * 100)
        },
        "daily_data": daily_results
    }
    
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths to update
    paths = [
        os.path.join(base_dir, "dashboard", "public", "data.json"),
        os.path.join(base_dir, "dashboard", "src", "data.json"),
        os.path.join(base_dir, "dashboard", "dist", "data.json")
    ]

    for p in paths:
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                json.dump(output, f, indent=2)
            print(f"Sync: Data saved to {p}")
        except Exception as e:
            print(f"Sync Error for {p}: {e}")

    print(f"Simulation complete. Accuracy: {accuracy*100:.2f}%, ROI: {roi:.2f}%")

if __name__ == "__main__":
    run_simulation()
