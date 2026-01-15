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
import smtplib
from email.mime.text import MIMEText

def send_email_alert(subject, body):
    """Send an email notification."""
    host = os.getenv("SMTP_HOST")
    port = os.getenv("SMTP_PORT")
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    to_addr = os.getenv("EMAIL_TO")

    if not all([host, port, user, password, to_addr]):
        return # Silently skip if not configured

    try:
        msg = MIMEText(body)
        msg['Subject'] = f"[Oxytocin] {subject}"
        msg['From'] = user
        msg['To'] = to_addr

        # Use SMTP_SSL for 465, else starttls for 587
        if str(port) == "465":
            with smtplib.SMTP_SSL(host, int(port)) as server:
                server.login(user, password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(host, int(port)) as server:
                server.starttls()
                server.login(user, password)
                server.send_message(msg)
        print(f"[MAIL] Sent: {subject}")
    except Exception as e:
        print(f"[MAIL] Failed: {e}")

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
TICKERS = ["9984.T", "7741.T", "8035.T", "7203.T"]  # Softbank, HOYA, Tokyo Electron, Toyota
INITIAL_BALANCE = 1000000.0
DATA_INTERVAL = "1m"       # Revert to 1m (Model requirement)
FETCH_PERIOD = "1d"
WINDOW_SIZE = 20
CHECK_INTERVAL_SECONDS = 60

# Risk Management - Stop Loss / Take Profit
STOP_LOSS_PCT = -0.04
TAKE_PROFIT_PCT = 10.0
POSITION_SIZE_PCT = 0.60

# Entry Filter - Balanced Approach
MIN_CONFIDENCE = 0.52      # 0.51 -> 0.52 (Filter noise)
MIN_HOLD_TICKS = 10        # 10 mins
COOLDOWN_TICKS = 60        # 60 mins cooldown
COOLDOWN_TICKS = 60        # Cooldown 60 minutes after sell

# Trading hours (JST)
TRADING_START_HOUR = 9
TRADING_START_MIN = 0
TRADING_END_HOUR = 15
TRADING_END_MIN = 0

# JST Timezone
JST = timezone(timedelta(hours=9))


def load_state_from_supabase():
    """Load persistent state from Supabase."""
    if not supabase:
        return {"balance": INITIAL_BALANCE, "positions": {}, "last_sell_time": {}}
    
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
                "entry_price": float(row["entry_price"]),
                "hold_ticks": 0, # Reset hold ticks on restart if not persisted used
                "highest_price": float(row["entry_price"])
            }
        
        # Note: last_sell_time should ideally be persisted but for now we start fresh on restart
        # or we could use trade_history to infer it. Simplified: Fresh start.
        return {"balance": balance, "positions": positions, "last_sell_time": {}}
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
            "initial_balance": INITIAL_BALANCE,  # Sync Initial Balance
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


    # Load System 2 Models
    print("Initializing System 2 (Sentiment Analysis)...")
    from news_fetcher import NewsFetcher
    from sentiment import SentimentAnalyzer
    
    news_fetcher = NewsFetcher()
    sentiment_analyzer = SentimentAnalyzer()
    
    # News Cache: {ticker: {"prediction": int, "confidence": float, "time": timestamp}}
    news_cache = {}
    NEWS_CACHE_SECONDS = 300 # 5 minutes

    # Load state
    state = load_state_from_supabase()
    print(f"[OK] State loaded. Balance: ${state['balance']:.2f}")
    
    while True:
        now = datetime.now(JST)
        
        if not is_trading_hours():
            next_open = now.replace(hour=TRADING_START_HOUR, minute=TRADING_START_MIN, second=0)
            if now > next_open:
                next_open += timedelta(days=1)
            sleep_seconds = (next_open - now).total_seconds()
            print(f"[{now.strftime('%H:%M:%S')}] Market closed. Sleeping until {next_open.strftime('%Y-%m-%d %H:%M')}...")
            time.sleep(min(sleep_seconds, 3600))
            continue
        
        print(f"\n[{now.strftime('%H:%M:%S')}] === Processing Tick (System 2 Only) ===")
        tickers_data = {}
        
        for ticker in TICKERS:
            # 1. Fetch Price Data (Necessary for execution)
            df = fetch_latest_data(ticker)
            if df is None:
                continue
            
            # Use 'prepare_features' just to get SMA/Close, ignoring SNN input
            result = prepare_features(df)
            if result is None:
                continue
            
            _, close_prices, sma_20_values = result
            current_price = close_prices[-1]
            current_sma_20 = sma_20_values[-1] if not np.isnan(sma_20_values[-1]) else current_price
            
            is_uptrend = current_price > current_sma_20
            tickers_data[ticker] = current_price
            
            # 2. System 2 Inference (News Gradient)
            # Check cache
            cached = news_cache.get(ticker)
            if cached and (time.time() - cached["time"] < NEWS_CACHE_SECONDS):
                prediction = cached["prediction"]
                confidence = cached["confidence"]
                # print(f"  [{ticker}] Reuse Cached Sentiment ({'UP' if prediction==1 else 'DOWN'} conf:{confidence:.2f})")
            else:
                # Fetch Check
                headlines = news_fetcher.fetch_latest_news(ticker)
                if headlines:
                    sent_df = sentiment_analyzer.analyze(headlines)
                    if not sent_df.empty:
                        # Logic: Sum scores. Positive=1, Negative=-1.
                        # FinBERT labels: 'Current FinBERT-tone' uses "Positive, Neutral, Negative" (case sensitive check needed)
                        # The code in hybrid_trader used 'positive' lower. sentiment.py output comes from pipeline.
                        # Usually pipeline returns 'Positive', 'Negative', 'Neutral'.
                        # Let's normalize to lowercase to be safe.
                        
                        pos_score = 0.0
                        neg_score = 0.0
                        total_score = 0.0
                        
                        for _, row in sent_df.iterrows():
                            lbl = row['label'].lower()
                            sc = row['score']
                            if lbl == 'positive':
                                pos_score += sc
                            elif lbl == 'negative':
                                neg_score += sc
                            total_score += 1 # Count articles
                        
                        # Normalize?
                        # If Pos > Neg -> Pred 1. Confidence = PosScore / TotalArticles?
                        if pos_score >= neg_score:
                            prediction = 1
                            # Confidence: How strong is the positive signal relative to count?
                            # Example: 3 articles. 2 Pos (0.9, 0.8), 1 Neg (0.1). PosSum=1.7. Mean=0.56.
                            # Let's use Mean Positive Score if Pred=1, Mean Negative Score if Pred=0.
                            confidence = (pos_score / total_score) if total_score > 0 else 0.0
                        else:
                            prediction = 0
                            confidence = (neg_score / total_score) if total_score > 0 else 0.0
                    else:
                        prediction = 0 # Default neutral/bearish
                        confidence = 0.0
                else:
                    prediction = 0
                    confidence = 0.0
                
                # Update Cache
                news_cache[ticker] = {
                    "prediction": prediction, 
                    "confidence": confidence, 
                    "time": time.time()
                }
                print(f"  [{ticker}] New Sentiment: {'UP' if prediction==1 else 'DOWN'} (Conf: {confidence:.2f}, Articles: {len(headlines)})")

            system2_used = True # Always True now
            
            # --- Logic Body (Same as before) ---
            pos = state["positions"].get(ticker)
            action = "HOLD"
            profit = 0.0
            
            # Check existing position...
            if pos is not None:
                entry_price = pos["entry_price"]
                highest_price = pos.get("highest_price", entry_price)
                if current_price > highest_price:
                    highest_price = current_price
                    pos["highest_price"] = highest_price
                
                price_change_pct = (current_price - entry_price) / entry_price
                trailing_stop_price = highest_price * (1 - 0.03)
                
                if price_change_pct <= STOP_LOSS_PCT:
                    profit = (current_price - entry_price) * pos["shares"]
                    state["balance"] += pos["shares"] * current_price
                    del state["positions"][ticker]
                    action = "STOP_LOSS"
                    delete_position_from_supabase(ticker)
                    msg = f"STOP LOSS Triggered for {ticker}.\nPrice: {current_price}\nLoss: {profit}"
                    print(f"  [{ticker}] STOP_LOSS @ ¥{current_price:.0f} | Loss: ¥{profit:.0f} ({price_change_pct*100:.1f}%)")
                    send_email_alert(f"STOP LOSS: {ticker}", msg)
                
                elif current_price < trailing_stop_price and price_change_pct > 0:
                    profit = (current_price - entry_price) * pos["shares"]
                    state["balance"] += pos["shares"] * current_price
                    del state["positions"][ticker]
                    action = "TRAILING_STOP"
                    delete_position_from_supabase(ticker)
                    msg = f"TRAILING STOP for {ticker}.\nPrice: {current_price}\nProfit: {profit}"
                    print(f"  [{ticker}] TRAILING_STOP @ ¥{current_price:.0f} | Profit: ¥{profit:.0f} (Peak: ¥{highest_price:.0f})")
                    send_email_alert(f"TRAILING STOP: {ticker}", msg)

                elif price_change_pct >= 0.10:
                    profit = (current_price - entry_price) * pos["shares"]
                    state["balance"] += pos["shares"] * current_price
                    del state["positions"][ticker]
                    action = "TAKE_PROFIT_Max"
                    delete_position_from_supabase(ticker)
                    msg = f"Max Profit Taken for {ticker}.\nPrice: {current_price}\nProfit: {profit}"
                    print(f"  [{ticker}] TAKE_PROFIT_Max @ ¥{current_price:.0f} | Profit: ¥{profit:.0f} ({price_change_pct*100:.1f}%)")
                    send_email_alert(f"TAKE PROFIT: {ticker}", msg)
                
                # Sell Logic based on System 2
                elif prediction == 0:
                    current_hold_ticks = pos.get("hold_ticks", 0)
                    if current_hold_ticks >= MIN_HOLD_TICKS:
                        profit = (current_price - entry_price) * pos["shares"]
                        state["balance"] += pos["shares"] * current_price
                        del state["positions"][ticker]
                        action = "SELL"
                        state["last_sell_time"][ticker] = time.time()
                        delete_position_from_supabase(ticker)
                        msg = f"SELL Signal (System 2) for {ticker}.\nPrice: {current_price}\nP/L: {profit}"
                        print(f"  [{ticker}] SELL @ ¥{current_price:.0f} | P/L: ¥{profit:.0f}")
                        send_email_alert(f"SELL: {ticker}", msg)
                    else:
                        action = "HOLDING_MIN_TIME"
                        pos["hold_ticks"] = current_hold_ticks + 1
                        print(f"  [{ticker}] HOLD (Min Time: {current_hold_ticks}/{MIN_HOLD_TICKS})")
                
                else:
                    action = "HOLDING"
                    pos["hold_ticks"] = pos.get("hold_ticks", 0) + 1
            
            # Buy Logic
            elif prediction == 1 and confidence >= MIN_CONFIDENCE and is_uptrend:
                last_sell = state.get("last_sell_time", {}).get(ticker, 0)
                if time.time() - last_sell < COOLDOWN_TICKS * 60:
                    print(f"  [{ticker}] SKIP BUY (Cooldown active)")
                    continue

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
                    msg = f"BUY Signal (System 2) for {ticker}.\nPrice: {current_price}\nShares: {shares}"
                    print(f"  [{ticker}] BUY {shares:.4f} shares @ ¥{current_price:.0f} (Sentiment Conf: {confidence:.2f})")
                    send_email_alert(f"BUY: {ticker}", msg)
            
            if action not in ["HOLD", "HOLDING"]:
                save_trade_to_supabase(ticker, action, current_price, profit, confidence, system2_used, state["balance"])
        
        # Calculate total value and update dashboard
        total_value = state["balance"]
        for ticker, pos in state["positions"].items():
            if ticker in tickers_data:
                total_value += pos["shares"] * tickers_data[ticker]
        
        roi = (total_value - INITIAL_BALANCE) / INITIAL_BALANCE * 100
        
        save_balance_to_supabase(state["balance"])
        update_dashboard_summary(state["balance"], total_value, roi)
        
        print(f"  Balance: ${state['balance']:.2f} | Total Value: ${total_value:.2f} | ROI: {roi:.2f}%")
        
        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
