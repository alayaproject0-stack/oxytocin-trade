from src.news_fetcher import NewsFetcher
from src.sentiment import SentimentAnalyzer

def main():
    print("=== Starting Phase 3: System 2 (News AI) Test ===")
    
    # 1. Fetch News
    nf = NewsFetcher()
    ticker = "TDK"
    headlines = nf.fetch_latest_news(ticker)
    
    print(f"\n--- Headlines for {ticker} ---")
    for i, h in enumerate(headlines[:5]):
        print(f"{i+1}. {h}")
        
    # 2. Analyze Sentiment
    sa = SentimentAnalyzer()
    df_result = sa.analyze(headlines)
    
    print("\n--- Sentiment Analysis Results ---")
    print(df_result)
    
    # Simple aggregation
    if not df_result.empty:
        sentiment_counts = df_result['label'].value_counts()
        print("\nSummary:")
        print(sentiment_counts)
        
        # Logic for "Reasoning"
        # FinBERT labels: Positive, Negative, Neutral
        pos = sentiment_counts.get('Positive', 0)
        neg = sentiment_counts.get('Negative', 0)
        
        print("\n[System 2 Conclusion]")
        if pos > neg:
            print("Bullish (強気): More positive news.")
        elif neg > pos:
            print("Bearish (弱気): More negative news.")
        else:
            print("Neutral (中立): Balanced or unclear.")

if __name__ == "__main__":
    main()
