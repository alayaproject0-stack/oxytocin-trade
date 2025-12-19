import yfinance as yf
import pandas as pd
from typing import List, Optional
import os
import requests
from dotenv import load_dotenv

class NewsFetcher:
    def __init__(self):
        # Load environment variables from .env
        load_dotenv()
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.av_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    def fetch_latest_news(self, ticker_key: str) -> List[str]:
        """
        Attempts to fetch live news. Priorities:
        1. AlphaVantage (Excellent for finance)
        2. NewsAPI.org 
        3. yfinance news
        4. Mock data
        """
        headlines = []

        # 1. Try AlphaVantage
        if self.av_api_key and self.av_api_key != "your_alpha_vantage_key_here":
            print(f"Fetching news for {ticker_key} via AlphaVantage...")
            headlines = self._fetch_via_alphavantage(ticker_key)
            if headlines:
                return headlines

        # 2. Try NewsAPI.org
        if self.news_api_key and self.news_api_key != "your_news_api_key_here":
            print(f"Fetching news for {ticker_key} via NewsAPI...")
            headlines = self._fetch_via_news_api(ticker_key)
            if headlines:
                return headlines

        # 2. Try yfinance news (Standard real data)
        print(f"Fetching news for {ticker_key} via yfinance...")
        headlines = self._fetch_via_yfinance(ticker_key)
        if headlines:
            return headlines

        # 3. Fallback to Mock
        print("Using mock news data...")
        return self.get_mock_news()

    def _fetch_via_alphavantage(self, ticker_key: str) -> List[str]:
        # TDK -> 6762.T is sometimes just TDK or TSE:6762 in AV. 
        # For simplicity, we use the ticker_key but in a real case we might need mapping.
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker_key,
            "limit": 5,
            "apikey": self.av_api_key
        }
        try:
            response = requests.get(url, params=params)
            data = response.json()
            if "feed" in data:
                return [article["title"] for article in data["feed"]]
        except Exception as e:
            print(f"AlphaVantage error: {e}")
        return []

    def _fetch_via_news_api(self, query: str) -> List[str]:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "pageSize": 5,
            "apiKey": self.news_api_key
        }
        try:
            response = requests.get(url, params=params)
            data = response.json()
            if data.get("status") == "ok":
                return [article["title"] for article in data.get("articles", [])]
        except Exception as e:
            print(f"NewsAPI error: {e}")
        return []

    def _fetch_via_yfinance(self, ticker_key: str) -> List[str]:
        # TDK -> 6762.T
        mapping = {"TDK": "6762.T", "TEL": "8035.T", "ADVANTEST": "6857.T"}
        symbol = mapping.get(ticker_key, ticker_key)
        
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            if news:
                return [item['title'] for item in news[:5]]
        except Exception as e:
            print(f"yfinance news error: {e}")
        return []

    def get_mock_news(self) -> List[str]:
        return [
            "Japan's tech sector sees growth in semiconductor demand.",
            "TDK announces new high-capacity battery for electric vehicles.",
            "Strong quarterly results expected for major Tokyo-listed tech firms.",
            "Market remains cautious ahead of central bank's policy meeting.",
            "Semiconductor equipment manufacturers ramp up production capacity."
        ]
