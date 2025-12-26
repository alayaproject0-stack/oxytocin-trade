#!/usr/bin/env python3
"""
Backtest Script for Oxytocin Trade
過去データを使用してトレード戦略を検証する

使用方法:
  python backtest.py --period 6mo --ticker 7203.T
  python backtest.py --period 1y --all-tickers
"""

import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import torch

from models_snn import SNNClassifier
from rl_agent import RLAgent, GatePolicy

# --- Configuration (same as live_trader_cloud.py) ---
TICKERS = ["6762.T", "7203.T", "6758.T", "9984.T"]
INITIAL_BALANCE = 1000000.0
WINDOW_SIZE = 20

# Risk Management
# Risk Management
STOP_LOSS_PCT = -0.04
TAKE_PROFIT_PCT = 10.0  # Effectively disabled, rely on Trailing Stop
POSITION_SIZE_PCT = 0.60
MIN_CONFIDENCE = 0.51


def fetch_historical_data(ticker: str, period: str = "6mo"):
    """Fetch historical daily data for backtesting."""
    print(f"[INFO] Fetching {period} data for {ticker}...")
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if df.empty:
        return None
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators as features."""
    features = pd.DataFrame(index=df.index)
    
    close = df['Close'].values.flatten()
    high = df['High'].values.flatten()
    low = df['Low'].values.flatten()
    volume = df['Volume'].values.flatten()
    
    # Returns
    features['return_1d'] = np.concatenate([[0], np.diff(close) / close[:-1]])
    features['return_5d'] = pd.Series(close).pct_change(5).fillna(0).values
    
    # Moving averages
    features['sma_5'] = pd.Series(close).rolling(5).mean().fillna(close[0]).values
    features['sma_20'] = pd.Series(close).rolling(20).mean().fillna(close[0]).values
    features['sma_ratio'] = features['sma_5'] / features['sma_20']
    
    # Volatility
    features['volatility'] = pd.Series(close).rolling(20).std().fillna(0).values
    
    # RSI
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    features['rsi'] = (100 - (100 / (1 + rs))).fillna(50).values
    
    # Volume ratio
    features['volume_ratio'] = volume / (pd.Series(volume).rolling(20).mean().fillna(1).values + 1e-10)
    
    # Normalize
    for col in features.columns:
        mean = features[col].mean()
        std = features[col].std()
        if std > 0:
            features[col] = (features[col] - mean) / std
    
    return features.fillna(0)


def compute_system2_signal(df: pd.DataFrame, idx: int) -> int:
    """
    System 2 Logic: Rigid technical confirmation
    Returns: 1 (Buy), 0 (Sell/Hold)
    """
    if idx < 30: return 0
    
    # Slice window for calculation
    window = df.iloc[idx-30:idx+1].copy()
    close = window['Close']
    
    # 1. RSI (14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    # 2. Bollinger Bands (20, 2)
    sma = close.rolling(20).mean()
    std = close.rolling(20).std()
    bb_low = sma - 2 * std
    current_bb_low = bb_low.iloc[-1]
    current_close = close.iloc[-1]
    
    # 3. MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    
    # Logic:
    # - RSI < 40 (Oversold but not extreme)
    # - Price near Lower Band (Dip buying)
    # - MACD Histogram increasing (Momentum shift)
    
    score = 0
    if float(current_rsi) < 45: score += 1
    if float(current_close) <= float(current_bb_low) * 1.02: score += 1
    if float(hist.iloc[-1]) > float(hist.iloc[-2]): score += 1
    
    # Relaxed: Need 2/3 confirmation
    return 1 if score >= 2 else 0


def run_inference(snn, rl, features: np.ndarray, df_raw: pd.DataFrame, idx: int, device):
    """Run model inference (Pure SNN)."""
    with torch.no_grad():
        x = torch.tensor(features[-WINDOW_SIZE:], dtype=torch.float32).unsqueeze(0).to(device)
        spk, mem = snn(x)
        spike_sum = spk.sum(dim=1)
        pred = spike_sum.argmax(dim=1).item()
        conf = torch.softmax(spike_sum, dim=1).max().item()
        
        # System 2 DISABLED for Aggressive Mode
        system2_used = False
                
    return pred, conf, system2_used


def run_backtest(ticker: str, period: str, device):
    """Run backtest on historical data."""
    # Fetch data
    df = fetch_historical_data(ticker, period)
    if df is None or len(df) < WINDOW_SIZE + 10:
        print(f"[ERROR] Insufficient data for {ticker}")
        return None
    
    # Compute features
    features_df = compute_features(df)
    close_prices = df['Close'].values.flatten()
    dates = df.index.strftime('%Y-%m-%d').tolist()
    
    # Initialize models
    n_features = features_df.shape[1]
    snn = SNNClassifier(in_dim=n_features, hidden=64, n_classes=2).to(device)
    rl = RLAgent()
    
    # Load weights if available
    try:
        snn.load_state_dict(torch.load("snn_model.pt", map_location=device, weights_only=True))
        print("[INFO] Loaded SNN model weights")
    except:
        print("[WARN] No SNN weights found, using random initialization")
    
    try:
        rl.policy.load_state_dict(torch.load("gate_policy.pt", map_location=device, weights_only=True))
        print("[INFO] Loaded RL policy weights")
    except:
        print("[WARN] No RL weights found, using random initialization")
    
    # Backtest state
    balance = INITIAL_BALANCE
    position = None  # {"shares": float, "entry_price": float, "entry_idx": int}
    
    trades = []
    equity_curve = []
    
    # Run through each day
    for i in range(WINDOW_SIZE, len(close_prices)):
        current_price = close_prices[i]
        current_date = dates[i]
        features = features_df.iloc[:i+1].values
        
        # Run inference
        pred, conf, system2_used = run_inference(snn, rl, features, df, i, device)
        
        # Trend Filter: Price > SMA20
        # sma_20 is at index -3 in features (based on compute_features)
        # However, features are normalized. Let's use raw price > sma_20 from fetch logic if possible
        # Simplified: Use normalized sma_20 from features. 
        # If current price > sma_20, it's an uptrend.
        # In normalized features: price is not directly there, but sma_20 is.
        # Let's calculate SMA20 explicitly here for clarity
        sma_20 = np.mean(close_prices[i-20:i]) if i >= 20 else current_price
        is_uptrend = current_price > sma_20
        
        action = "HOLD"
        profit = 0.0
        
        # Check existing position
        if position is not None:
            entry_price = position["entry_price"]
            
            # Trailing Stop Management
            # Update Highest Price for Trailing Stop
            highest_price = position.get("highest_price", entry_price)
            if current_price > highest_price:
                highest_price = current_price
                # Update in local state implies update in logic, need to persist if we were fully stateful, but for now in-mem
                position["highest_price"] = highest_price
            
            price_change_pct = (current_price - entry_price) / entry_price
            trailing_stop_price = highest_price * (1 - 0.03) # 3.0% trailing stop (Aggressive)
            
            # Trailing Stop Triggered
            if current_price < trailing_stop_price and price_change_pct > 0:
                 profit = (current_price - entry_price) * position["shares"]
                 balance += position["shares"] * current_price
                 action = "TRAILING_STOP"
                 position = None
            
            # Stop-loss
            elif price_change_pct <= STOP_LOSS_PCT:
                profit = (current_price - entry_price) * position["shares"]
                balance += position["shares"] * current_price
                action = "STOP_LOSS"
                position = None
            
            # Take-profit (Partial or Full - here Full for simplicity, but Trailing Stop handles big runners)
            # Replaced fixed TAKE_PROFIT with Trailing Stop preference, but keep emergency take profit if price spikes
            elif price_change_pct >= 0.10: # 10% hard target
                profit = (current_price - entry_price) * position["shares"]
                balance += position["shares"] * current_price
                action = "TAKE_PROFIT_Max"
                position = None
            
            # Model sell signal
            elif pred == 0:
                profit = (current_price - entry_price) * position["shares"]
                balance += position["shares"] * current_price
                action = "SELL"
                position = None
            
            else:
                action = "HOLDING"
        
        # Consider buying - ONLY in Uptrend
        elif pred == 1 and conf >= MIN_CONFIDENCE and is_uptrend:
            shares = balance * POSITION_SIZE_PCT / current_price
            if shares > 0:
                position = {
                    "shares": shares,
                    "entry_price": current_price,
                    "entry_idx": i
                }
                balance -= shares * current_price
                action = "BUY"
        
        # Calculate total value
        total_value = balance
        if position:
            total_value += position["shares"] * current_price
        
        equity_curve.append({
            "date": current_date,
            "balance": balance,
            "total_value": total_value,
            "price": current_price
        })
        
        if action != "HOLD" and action != "HOLDING":
            trades.append({
                "date": current_date,
                "ticker": ticker,
                "action": action,
                "price": current_price,
                "profit": profit,
                "confidence": conf,
                "system2_used": system2_used,
                "balance_after": balance
            })
    
    # Close any remaining position
    if position:
        final_price = close_prices[-1]
        profit = (final_price - position["entry_price"]) * position["shares"]
        balance += position["shares"] * final_price
        trades.append({
            "date": dates[-1],
            "ticker": ticker,
            "action": "CLOSE_EOD",
            "price": final_price,
            "profit": profit,
            "confidence": 0,
            "system2_used": False,
            "balance_after": balance
        })
    
    # Calculate metrics
    total_value = balance
    roi = (total_value - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    
    sell_trades = [t for t in trades if t["action"] in ["SELL", "STOP_LOSS", "TAKE_PROFIT", "CLOSE_EOD"]]
    win_trades = [t for t in sell_trades if t["profit"] > 0]
    lose_trades = [t for t in sell_trades if t["profit"] < 0]
    
    win_rate = len(win_trades) / len(sell_trades) * 100 if sell_trades else 0
    avg_win = np.mean([t["profit"] for t in win_trades]) if win_trades else 0
    avg_loss = np.mean([t["profit"] for t in lose_trades]) if lose_trades else 0
    
    # Max drawdown
    peak = INITIAL_BALANCE
    max_dd = 0
    for eq in equity_curve:
        if eq["total_value"] > peak:
            peak = eq["total_value"]
        dd = (eq["total_value"] - peak) / peak
        if dd < max_dd:
            max_dd = dd
    
    results = {
        "ticker": ticker,
        "period": period,
        "initial_balance": INITIAL_BALANCE,
        "final_balance": balance,
        "roi_pct": roi,
        "total_trades": len(sell_trades),
        "win_rate_pct": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown_pct": max_dd * 100,
        "trades": trades,
        "equity_curve": equity_curve
    }
    
    return results


def print_results(results):
    """Print backtest results."""
    print("\n" + "="*60)
    print(f"BACKTEST RESULTS: {results['ticker']} ({results['period']})")
    print("="*60)
    print(f"初期残高:       ¥{results['initial_balance']:,.0f}")
    print(f"最終残高:       ¥{results['final_balance']:,.0f}")
    print(f"ROI:            {results['roi_pct']:+.2f}%")
    print(f"総取引数:       {results['total_trades']}")
    print(f"勝率:           {results['win_rate_pct']:.1f}%")
    print(f"平均利益:       ¥{results['avg_win']:,.0f}")
    print(f"平均損失:       ¥{results['avg_loss']:,.0f}")
    print(f"最大DD:         {results['max_drawdown_pct']:.2f}%")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Backtest Oxytocin Trade Strategy")
    parser.add_argument("--ticker", type=str, help="Single ticker to test (e.g., 7203.T)")
    parser.add_argument("--all-tickers", action="store_true", help="Test all configured tickers")
    parser.add_argument("--period", type=str, default="6mo", help="Period: 1mo, 3mo, 6mo, 1y, 2y")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    tickers = TICKERS if args.all_tickers else [args.ticker or "7203.T"]
    all_results = []
    
    for ticker in tickers:
        results = run_backtest(ticker, args.period, device)
        if results:
            print_results(results)
            all_results.append(results)
    
    # Save results
    if args.output:
        output_data = {
            "generated_at": datetime.now().isoformat(),
            "config": {
                "stop_loss_pct": STOP_LOSS_PCT,
                "take_profit_pct": TAKE_PROFIT_PCT,
                "position_size_pct": POSITION_SIZE_PCT,
                "min_confidence": MIN_CONFIDENCE
            },
            "results": all_results
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\n[INFO] Results saved to {args.output}")
    
    # Summary for all tickers
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("SUMMARY: ALL TICKERS")
        print("="*60)
        total_roi = np.mean([r["roi_pct"] for r in all_results])
        avg_win_rate = np.mean([r["win_rate_pct"] for r in all_results])
        print(f"平均ROI:        {total_roi:+.2f}%")
        print(f"平均勝率:       {avg_win_rate:.1f}%")
        print("="*60)


if __name__ == "__main__":
    main()
