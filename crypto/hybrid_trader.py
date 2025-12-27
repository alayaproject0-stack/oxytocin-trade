import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Optional, Dict

from data_loader import DataFetcher
from features import FeatureEngineer
from models_snn import SNNClassifier
from news_fetcher import NewsFetcher
from sentiment import SentimentAnalyzer

from rl_agent import RLAgent

class HybridTrader:
    def __init__(
        self,
        ticker_key: str = "TDK",
        model_path: str = "snn_model_latest.pth",
        rl_policy_path: str = "rl_policy_latest.pth",
        confidence_threshold: float = 0.6,
        hidden_dim: int = 128,
        window_size: int = 20
    ):
        self.ticker_key = ticker_key
        self.confidence_threshold = confidence_threshold # Fallback
        self.window_size = window_size
        
        # Initialize Components
        self.fetcher = DataFetcher()
        self.engineer = FeatureEngineer()
        self.news_fetcher = NewsFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Load SNN Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feature_cols = ['RSI', 'BB_Width', 'Return', 'Log_Return', 'Volume_Change']
        in_dim = len(feature_cols) * window_size
        
        self.model = SNNClassifier(in_dim=in_dim, hidden=hidden_dim, n_classes=2).to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Loaded SNN model from {model_path}")
        except FileNotFoundError:
            print(f"Warning: SNN model file {model_path} not found.")

        # Load RL Agent
        self.rl_agent = RLAgent()
        try:
            self.rl_agent.load(rl_policy_path)
            self.use_rl = True
            print(f"Loaded RL Policy from {rl_policy_path}")
        except FileNotFoundError:
            self.use_rl = False
            print(f"RL Policy not found. Using fixed threshold {self.confidence_threshold}")

    def run_inference(self) -> Dict:
        """
        Runs the hybrid inference flow:
        1. SNN (System 1) prediction.
        2. If confidence < threshold, trigger Sentiment Analysis (System 2).
        """
        print(f"\n--- Running Hybrid Inference for {self.ticker_key} ---")
        
        # 1. System 1: SNN
        df = self.fetcher.fetch_data(self.ticker_key, period="3mo")
        if df is None:
            return {"error": "Failed to fetch data"}
            
        df = self.engineer.add_technical_indicators(df)
        feature_cols = ['RSI', 'BB_Width', 'Return', 'Log_Return', 'Volume_Change']
        df_norm = self.engineer.normalize_data(df, feature_cols)
        
        # Ensure we have enough data after technical indicators
        if len(df_norm) < self.window_size:
            return {"error": f"Insufficient clean data (needed {self.window_size}, got {len(df_norm)})"}
            
        # Get last window
        window_df = df_norm[feature_cols].tail(self.window_size)
        last_window = window_df.values.flatten()
        
        print(f"Input window shape: {window_df.shape} -> Flattened: {last_window.shape}")
        input_tensor = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, mean_spikes = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
            
        confidence = confidence.item()
        prediction = prediction.item() # 0: Down, 1: Up
        
        result = {
            "ticker": self.ticker_key,
            "system1": {
                "prediction": "UP" if prediction == 1 else "DOWN",
                "confidence": confidence,
                "mean_spikes": mean_spikes
            },
            "system2_triggered": False
        }
        
        print(f"System 1 (SNN) Prediction: {result['system1']['prediction']} (Conf: {confidence:.4f})")
        
        # 2. System 2 Trigger Decision
        should_wake = False
        if self.use_rl:
            should_wake = self.rl_agent.decide(probs).item()
            print(f"RL Agent Decision: {'WAKE' if should_wake else 'STAY'}")
        else:
            should_wake = confidence < self.confidence_threshold
            print(f"Fixed Threshold Decision: {'WAKE' if should_wake else 'STAY'}")

        # 3. System 2: Sentiment Analysis
        if should_wake:
            result["system2_triggered"] = True
            
            headlines = self.news_fetcher.fetch_latest_news(self.ticker_key)
            sentiment_df = self.sentiment_analyzer.analyze(headlines)
            
            # Simple aggregation: Positive count vs Negative count
            if not sentiment_df.empty:
                pos_score = sentiment_df[sentiment_df['label'] == 'positive']['score'].sum()
                neg_score = sentiment_df[sentiment_df['label'] == 'negative']['score'].sum()
                
                sentiment_pred = "UP" if pos_score >= neg_score else "DOWN"
                
                result["system2"] = {
                    "prediction": sentiment_pred,
                    "pos_score": float(pos_score),
                    "neg_score": float(neg_score),
                    "headlines_count": len(headlines)
                }
                print(f"System 2 (Sentiment) Prediction: {sentiment_pred} (Pos: {pos_score:.2f}, Neg: {neg_score:.2f})")
            else:
                result["system2"] = {"prediction": "NEUTRAL", "error": "No news found"}
                print("System 2: No news found to analyze.")
        
        return result

if __name__ == "__main__":
    # Test with low threshold (likely won't trigger System 2)
    trader = HybridTrader(confidence_threshold=0.8) # Set higher to trigger for testing
    res = trader.run_inference()
    print("\nFinal Result:", res)
