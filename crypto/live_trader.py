"""
Live Trading Daemon - Real-time virtual trading with hybrid AI system.
Runs as a persistent background process, fetching data every minute during trading hours.
"""
import os
import json
import time
import torch
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, timedelta
import ta

# Local imports
from models_snn import SNNClassifier
from rl_agent import RLAgent

# --- Configuration ---
TICKERS = ["6762.T", "7203.T", "6758.T", "9984.T"]  # TDK, Toyota, Sony, Softbank
INITIAL_BALANCE = 10000.0
DATA_INTERVAL = "1m"
FETCH_PERIOD = "1d"
WINDOW_SIZE = 20
CHECK_INTERVAL_SECONDS = 60  # 1 minute

# Trading hours (JST)
TRADING_START_HOUR = 9
TRADING_START_MIN = 0
TRADING_END_HOUR = 15
TRADING_END_MIN = 0

# File paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(SCRIPT_DIR, "live_state.json")
DASHBOARD_DATA_PATH = os.path.join(SCRIPT_DIR, "dashboard", "public", "data.json")
DASHBOARD_DIST_PATH = os.path.join(SCRIPT_DIR, "dashboard", "dist", "data.json")

# JST Timezone
JST = timezone(timedelta(hours=9))


def load_state():
    """Load persistent state from JSON file."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "balance": INITIAL_BALANCE,
        "positions": {},  # {ticker: {"shares": n, "entry_price": p}}
        "trade_history": [],
        "daily_pnl": []
    }


def save_state(state):
    """Save state to JSON file."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def is_trading_hours():
    """Check if current time is within trading hours (JST)."""
    now = datetime.now(JST)
    weekday = now.weekday()
    
    # Skip weekends
    if weekday >= 5:
        return False
    
    trading_start = now.replace(hour=TRADING_START_HOUR, minute=TRADING_START_MIN, second=0)
    trading_end = now.replace(hour=TRADING_END_HOUR, minute=TRADING_END_MIN, second=0)
    
    return trading_start <= now <= trading_end


def fetch_latest_data(ticker):
    """Fetch latest minute-level data for a ticker."""
    try:
        df = yf.download(ticker, period=FETCH_PERIOD, interval=DATA_INTERVAL, progress=False)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"[ERROR] Failed to fetch data for {ticker}: {e}")
        return None


def prepare_features(df):
    """Prepare technical features for the model."""
    if len(df) < WINDOW_SIZE + 14:
        return None
    
    df = df.copy()
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['BB_High'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
    df['BB_Low'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
    df['Returns'] = df['Close'].pct_change()
    
    df = df.dropna()
    
    if len(df) < WINDOW_SIZE:
        return None
    
    # Normalize
    df_norm = df[['Returns', 'RSI', 'MACD', 'BB_High', 'BB_Low']].copy()
    df_norm = (df_norm - df_norm.mean()) / (df_norm.std() + 1e-8)
    
    return df_norm, df['Close'].values


def run_inference(snn, rl, features, device):
    """Run SNN + RL inference on the latest window."""
    window = features[-WINDOW_SIZE:].flatten()
    input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits, _ = snn(input_tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        should_wake = rl.decide(probs).item()
    
    return pred.item(), conf.item(), bool(should_wake)


def update_dashboard(state, tickers_data):
    """Update dashboard JSON with current state."""
    total_value = state["balance"]
    for ticker, pos in state["positions"].items():
        if ticker in tickers_data and tickers_data[ticker] is not None:
            current_price = tickers_data[ticker]
            total_value += pos["shares"] * current_price
    
    roi = (total_value - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    
    # Get recent trades
    recent_trades = state["trade_history"][-20:]
    
    output = {
        "summary": {
            "tickers": TICKERS,
            "initial_balance": INITIAL_BALANCE,
            "current_balance": state["balance"],
            "total_value": total_value,
            "roi_pct": roi,
            "positions": state["positions"],
            "last_update": datetime.now(JST).isoformat()
        },
        "daily_data": recent_trades
    }
    
    for path in [DASHBOARD_DATA_PATH, DASHBOARD_DIST_PATH]:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(output, f, indent=2, default=str)
        except Exception as e:
            print(f"[WARN] Could not save to {path}: {e}")


def main():
    print("=" * 60)
    print("Oxytocin Live Trader - Starting...")
    print(f"Tickers: {TICKERS}")
    print(f"Check Interval: {CHECK_INTERVAL_SECONDS}s")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    input_dim = WINDOW_SIZE * 5
    snn = SNNClassifier(input_dim=input_dim, hidden_dim=128, output_dim=2).to(device)
    snn_path = os.path.join(SCRIPT_DIR, "snn_model.pth")
    if os.path.exists(snn_path):
        snn.load_state_dict(torch.load(snn_path, map_location=device, weights_only=True))
        print("[OK] SNN model loaded.")
    else:
        print("[WARN] SNN model not found, using untrained model.")
    snn.eval()
    
    rl = RLAgent(state_dim=2)
    rl_path = os.path.join(SCRIPT_DIR, "rl_agent.pth")
    if os.path.exists(rl_path):
        rl.load(rl_path)
        print("[OK] RL agent loaded.")
    else:
        print("[WARN] RL agent not found, using default policy.")
    
    # Load state
    state = load_state()
    print(f"[OK] State loaded. Balance: ${state['balance']:.2f}")
    
    while True:
        now = datetime.now(JST)
        
        if not is_trading_hours():
            next_open = now.replace(hour=TRADING_START_HOUR, minute=TRADING_START_MIN, second=0)
            if now > next_open:
                # Already past today's open, wait for tomorrow
                next_open += timedelta(days=1)
            sleep_seconds = (next_open - now).total_seconds()
            print(f"[{now.strftime('%H:%M:%S')}] Market closed. Sleeping until {next_open.strftime('%Y-%m-%d %H:%M')}...")
            time.sleep(min(sleep_seconds, 3600))  # Sleep max 1 hour at a time
            continue
        
        print(f"\n[{now.strftime('%H:%M:%S')}] === Processing Tick ===")
        tickers_data = {}
        
        for ticker in TICKERS:
            df = fetch_latest_data(ticker)
            if df is None:
                continue
            
            result = prepare_features(df)
            if result is None:
                continue
            
            features_df, close_prices = result
            current_price = close_prices[-1]
            tickers_data[ticker] = current_price
            
            prediction, confidence, system2_used = run_inference(
                snn, rl, features_df.values, device
            )
            
            pos = state["positions"].get(ticker)
            action = "HOLD"
            profit = 0.0
            
            if prediction == 1:  # Bullish
                if pos is None:
                    # BUY
                    shares = state["balance"] * 0.2 / current_price  # Invest 20% per ticker
                    if shares > 0:
                        state["positions"][ticker] = {
                            "shares": shares,
                            "entry_price": current_price
                        }
                        state["balance"] -= shares * current_price
                        action = "BUY"
                        print(f"  [{ticker}] BUY {shares:.4f} shares @ ${current_price:.2f}")
                else:
                    action = "HOLDING"
            else:  # Bearish
                if pos is not None:
                    # SELL
                    profit = (current_price - pos["entry_price"]) * pos["shares"]
                    state["balance"] += pos["shares"] * current_price
                    del state["positions"][ticker]
                    action = "SELL"
                    print(f"  [{ticker}] SELL @ ${current_price:.2f} | Profit: ${profit:.2f}")
            
            state["trade_history"].append({
                "date": now.isoformat(),
                "ticker": ticker,
                "action": action,
                "price": float(current_price),
                "profit": float(profit),
                "confidence": float(confidence),
                "system2_used": system2_used,
                "balance": float(state["balance"])
            })
        
        # Save and update dashboard
        save_state(state)
        update_dashboard(state, tickers_data)
        print(f"  Balance: ${state['balance']:.2f} | Positions: {list(state['positions'].keys())}")
        
        # Wait for next tick
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
