from src.data_loader import DataFetcher
from src.features import FeatureEngineer
import pandas as pd

def main():
    print("=== Starting Oxytocin-Trade Data Pipeline ===")
    
    # 1. Fetch Data
    fetcher = DataFetcher()
    ticker = "TDK"
    print(f"Fetching data for {ticker}...")
    df = fetcher.fetch_data(ticker)
    
    if df is None or df.empty:
        print("Failed to fetch data.")
        return

    print(f"Successfully fetched {len(df)} rows.")
    print(df.head(3))

    # 2. Feature Engineering
    print("\nCalculating technical indicators...")
    fe = FeatureEngineer()
    try:
        df_features = fe.add_technical_indicators(df)
        print(f"Data shape after feature engineering: {df_features.shape}")
        print(df_features[['Close', 'RSI', 'BB_High', 'BB_Low', 'Return']].tail())
        
        # 3. Normalization (Preview)
        print("\nNormalizing data (Preview)...")
        cols_to_norm = ['RSI', 'BB_Width', 'Return']
        df_norm = fe.normalize_data(df_features, cols_to_norm)
        print(df_norm[cols_to_norm].tail())
        
        print("\n=== Pipeline Verified Successfully ===")
        
    except Exception as e:
        print(f"\nError during feature engineering: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
