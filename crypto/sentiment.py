from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
import pandas as pd
from typing import List

class SentimentAnalyzer:
    def __init__(self, model_name="yiyanghkust/finbert-tone"):
        print(f"Loading FinBERT model: {model_name}...")
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("sentiment-analysis", model=model_name, device=self.device)
        print("Model loaded.")

    def analyze(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyzes a list of text headlines.
        Returns a DataFrame with columns: ['text', 'label', 'score']
        """
        if not texts:
            return pd.DataFrame(columns=['text', 'label', 'score'])
            
        print(f"Analyzing {len(texts)} headlines...")
        results = self.pipe(texts)
        
        # Structure the data
        data = []
        for text, res in zip(texts, results):
            data.append({
                'text': text,
                'label': res['label'], # 'Positive', 'Negative', 'Neutral' usually
                'score': res['score']
            })
            
        return pd.DataFrame(data)

if __name__ == "__main__":
    # Test
    sa = SentimentAnalyzer()
    texts = ["Stocks rally on good earnings", "Market crashes due to inflation fears"]
    df = sa.analyze(texts)
    print(df)
