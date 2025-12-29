"""
Live Trading Daemon (Cloud Version) - Real-time virtual trading with Supabase storage.
Designed for deployment on Render.com as a Background Worker.
"""
import os
import time
import torch
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import ta

# Load environment variables
load_dotenv()

# Supabase client
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("[WARN] Supabase credentials not found. Using local fallback mode.")
    supabase = None
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("[OK] Supabase client initialized.")

# Local imports
from models_snn import SNNClassifier
from rl_agent import RLAgent

# --- Configuration ---
# --- Configuration (Crypto) ---
TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD"] # Removed XRP, DOGE (Low performance)
INITIAL_BALANCE = 10000.0 # $10,000 Start
DATA_INTERVAL = "1m"
FETCH_PERIOD = "1d"
WINDOW_SIZE = 20
CHECK_INTERVAL_SECONDS = 60  # 1 minute

# Risk Management - Stop Loss / Take Profit
# Risk Management - Stop Loss / Take Profit
# Risk Management (Crypto)
STOP_LOSS_PCT = -0.05      # -5% Stop (Wider for crypto)
TAKE_PROFIT_PCT = 10.0     # Disabled
POSITION_SIZE_PCT = 0.40   # 40% Position (Reduced leverage)

# Entry Filter - 取引頻度を減らす
# Entry Filter - 取引頻度を増やす
MIN_CONFIDENCE = 0.51      # 自信度51%（迷ったらGO）
MIN_HOLD_TICKS = 5         # 最低5ティック（5分）保有してから売却判断

# Trading hours (JST)
# Trading hours (24/7 for Crypto) - Logic disabled in main loop
# TRADING_START/END ignored


# JST Timezone
JST = timezone(timedelta(hours=9))


def load_state_from_supabase():
    """Load persistent state from Supabase."""
    if not supabase:
        return {"balance": INITIAL_BALANCE, "positions": {}}
    
    try:
        # Get balance
        result = supabase.table("portfolio").select("balance").limit(1).execute()
        balance = float(result.data[0]["balance"]) if result.data else INITIAL_BALANCE
        
        # Get positions
        result = supabase.table("positions").select("*").execute()
        positions = {}
        for row in result.data:
            positions[row["ticker"]] = {
                "shares": float(row["shares"]),
                "entry_price": float(row["entry_price"])
            }
        
        return {"balance": balance, "positions": positions}
    except Exception as e:
        print(f"[ERROR] Failed to load state: {e}")
        return {"balance": INITIAL_BALANCE, "positions": {}}


def save_balance_to_supabase(balance):
    """Save current balance to Supabase."""
    if not supabase:
        return
    try:
        supabase.table("portfolio").update({"balance": balance, "updated_at": datetime.now(JST).isoformat()}).eq("id", supabase.table("portfolio").select("id").limit(1).execute().data[0]["id"]).execute()
    except Exception as e:
        print(f"[ERROR] Failed to save balance: {e}")


def save_position_to_supabase(ticker, shares, entry_price):
    """Save or update a position in Supabase."""
    if not supabase:
        return
    try:
        supabase.table("positions").upsert({
            "ticker": ticker,
            "shares": shares,
            "entry_price": entry_price,
            "updated_at": datetime.now(JST).isoformat()
        }, on_conflict="ticker").execute()
    except Exception as e:
        print(f"[ERROR] Failed to save position: {e}")


def delete_position_from_supabase(ticker):
    """Delete a position from Supabase."""
    if not supabase:
        return
    try:
        supabase.table("positions").delete().eq("ticker", ticker).execute()
    except Exception as e:
        print(f"[ERROR] Failed to delete position: {e}")


def save_trade_to_supabase(ticker, action, price, profit, confidence, system2_used, balance_after):
    """Save a trade record to Supabase."""
    if not supabase:
        return
    try:
        supabase.table("trade_history").insert({
            "ticker": ticker,
            "action": action,
            "price": price,
            "profit": profit,
            "confidence": confidence,
            "system2_used": system2_used,
            "balance_after": balance_after
        }).execute()
    except Exception as e:
        print(f"[ERROR] Failed to save trade: {e}")


def update_dashboard_summary(balance, total_value, roi_pct):
    """Update the dashboard summary in Supabase."""
    if not supabase:
        return
    try:
        supabase.table("dashboard_summary").update({
            "current_balance": balance,
            "total_value": total_value,
            "roi_pct": roi_pct,
            "last_update": datetime.now(JST).isoformat()
        }).eq("id", 1).execute()
    except Exception as e:
        print(f"[ERROR] Failed to update dashboard: {e}")


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
    
    df_norm = df[['Returns', 'RSI', 'MACD', 'BB_High', 'BB_Low']].copy()
    df_norm = (df_norm - df_norm.mean()) / (df_norm.std() + 1e-8)
    
    # Calculate SMA20 for Trend Filter (not used as model feature but for separate logic)
    sma_20 = df['Close'].rolling(window=20).mean().values
    
    return df_norm, df['Close'].values, sma_20


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


def main():
    print("=" * 60)
    print("Oxytocin Live Trader (Cloud) - Starting...")
    print(f"Tickers: {TICKERS}")
    print(f"Check Interval: {CHECK_INTERVAL_SECONDS}s")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dim = WINDOW_SIZE * 5
    snn = SNNClassifier(in_dim=input_dim, hidden=128, n_classes=2).to(device)
    snn_path = os.path.join(script_dir, "snn_model.pth")
    if os.path.exists(snn_path):
        snn.load_state_dict(torch.load(snn_path, map_location=device, weights_only=True))
        print("[OK] SNN model loaded.")
    else:
        print("[WARN] SNN model not found, using untrained model.")
    snn.eval()
    
    rl = RLAgent()
    rl_path = os.path.join(script_dir, "rl_agent.pth")
    if os.path.exists(rl_path):
        rl.load(rl_path)
        print("[OK] RL agent loaded.")
    else:
        print("[WARN] RL agent not found, using default policy.")
    
    # Load state
    state = load_state_from_supabase()
    print(f"[OK] State loaded. Balance: ${state['balance']:.2f}")
    
    while True:
        now = datetime.now(JST)
        
        # Crypto is 24/7 - removed market hours check
        # if not is_trading_hours(): ...
        
        print(f"\n[{now.strftime('%H:%M:%S')}] === Processing Tick ===")
        tickers_data = {}
        
        for ticker in TICKERS:
            df = fetch_latest_data(ticker)
            if df is None:
                continue
            
            result = prepare_features(df)
            if result is None:
                continue
            
            features_df, close_prices, sma_20_values = result
            current_price = close_prices[-1]
            current_sma_20 = sma_20_values[-1] if not np.isnan(sma_20_values[-1]) else current_price
            
            is_uptrend = current_price > current_sma_20
            tickers_data[ticker] = current_price
            
            prediction, confidence, system2_used = run_inference(
                snn, rl, features_df.values, device
            )
            
            pos = state["positions"].get(ticker)
            action = "HOLD"
            profit = 0.0
            
            # Check existing position for stop-loss / take-profit / trailing-stop
            if pos is not None:
                entry_price = pos["entry_price"]
                
                # Update Highest Price for Trailing Stop
                highest_price = pos.get("highest_price", entry_price)
                if current_price > highest_price:
                    highest_price = current_price
                    # Update in local state implies update in logic, need to persist if we were fully stateful, but for now in-mem
                    pos["highest_price"] = highest_price
                    # Note: We should ideally update Supabase too if we want persistence across restarts for this, 
                    # but for this iteration we'll keep it simple or assume reliable runtime.
                
                price_change_pct = (current_price - entry_price) / entry_price
                trailing_stop_price = highest_price * (1 - 0.03) # 3.0% trailing stop (Aggressive)
                
                # Stop-loss triggered
                if price_change_pct <= STOP_LOSS_PCT:
                    profit = (current_price - entry_price) * pos["shares"]
                    state["balance"] += pos["shares"] * current_price
                    del state["positions"][ticker]
                    action = "STOP_LOSS"
                    delete_position_from_supabase(ticker)
                    print(f"  [{ticker}] STOP_LOSS @ ${current_price:.2f} | Loss: ${profit:.2f} ({price_change_pct*100:.1f}%)")
                
                # Trailing Stop Triggered
                elif current_price < trailing_stop_price and price_change_pct > 0:
                    profit = (current_price - entry_price) * pos["shares"]
                    state["balance"] += pos["shares"] * current_price
                    del state["positions"][ticker]
                    action = "TRAILING_STOP"
                    delete_position_from_supabase(ticker)
                    print(f"  [{ticker}] TRAILING_STOP @ ${current_price:.2f} | Profit: ${profit:.2f} (Peak: ${highest_price:.2f})")

                # Take-profit triggered (Hardcap fallback or just keep trailing. Let's keep a hard cap at 10% for safety)
                elif price_change_pct >= 0.10:
                    profit = (current_price - entry_price) * pos["shares"]
                    state["balance"] += pos["shares"] * current_price
                    del state["positions"][ticker]
                    action = "TAKE_PROFIT_Max"
                    delete_position_from_supabase(ticker)
                    print(f"  [{ticker}] TAKE_PROFIT_Max @ ${current_price:.2f} | Profit: ${profit:.2f} ({price_change_pct*100:.1f}%)")
                
                # Model says sell (bearish)
                elif prediction == 0:
                    profit = (current_price - entry_price) * pos["shares"]
                    state["balance"] += pos["shares"] * current_price
                    del state["positions"][ticker]
                    action = "SELL"
                    delete_position_from_supabase(ticker)
                    print(f"  [{ticker}] SELL @ ${current_price:.2f} | P/L: ${profit:.2f}")
                
                else:
                    action = "HOLDING"
                    pos["hold_ticks"] = pos.get("hold_ticks", 0) + 1
            
            # No position - consider buying (with confidence filter AND Trend Filter)
            elif prediction == 1 and confidence >= MIN_CONFIDENCE and is_uptrend:  # Bullish + High confidence + Uptrend
                shares = state["balance"] * POSITION_SIZE_PCT / current_price
                if shares > 0 and state["balance"] > 0:
                    state["positions"][ticker] = {
                        "shares": shares,
                        "entry_price": current_price,
                        "hold_ticks": 0,
                        "highest_price": current_price
                    }
                    state["balance"] -= shares * current_price
                    action = "BUY"
                    save_position_to_supabase(ticker, shares, current_price)
                    print(f"  [{ticker}] BUY {shares:.4f} shares @ ${current_price:.2f} (conf: {confidence:.2f}, sma20: {current_sma_20:.2f})")
            
            # Save trade (only if action is meaningful)
            if action not in ["HOLD", "HOLDING"]:
                save_trade_to_supabase(ticker, action, current_price, profit, confidence, system2_used, state["balance"])
        
        # Calculate total value
        total_value = state["balance"]
        for ticker, pos in state["positions"].items():
            if ticker in tickers_data:
                total_value += pos["shares"] * tickers_data[ticker]
        
        roi = (total_value - INITIAL_BALANCE) / INITIAL_BALANCE * 100
        
        # Save state
        save_balance_to_supabase(state["balance"])
        update_dashboard_summary(state["balance"], total_value, roi)
        
        print(f"  Balance: ${state['balance']:.2f} | Total Value: ${total_value:.2f} | ROI: {roi:.2f}%")
        
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
