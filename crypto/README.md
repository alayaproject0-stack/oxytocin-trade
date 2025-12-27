# oxytocin-trade: SNNと感情分析によるハイブリッド・トレーディング・システム

## プロジェクト概要
`oxytocin-trade`は、人間の認知機能である「システム1（速い思考）」と「システム2（遅い思考）」のモデルをヒントにした、次世代のハイブリッド・アルゴトレーディング・システムです。

低消費電力かつリアルタイム性に優れた**スパイキング・ニューラル・ネットワーク（SNN）**によるテクニカル分析と、高度なコンテキスト理解を可能にする**Transformer（FinBERT）**によるニュース感情分析を組み合わせ、効率的かつ精度の高い市場予測を目指します。

## システムアーキテクチャ

本システムは、以下の2つのエネルギー効率が異なるコンポーネントで構成されます。

### 1. システム1: 高速・低エネルギー予測 (SNN)
- **役割**: 市場の短期的な動きを監視し、テクニカル指標に基づいて高速に予測を行います。
- **技術スタック**: PyTorch、サロゲート勾配を用いた LIF (Leaky Integrate-and-Fire) モデル。
- **特徴**: イベント駆動型の計算により、必要最小限のエネルギーで継続的な市場監視が可能です。

### 2. システム2: 高度なコンテキスト分析 (Sentiment Analysis)
- **役割**: 市場に大きな影響を与えるニュースやSNSのヘッドラインを分析し、マクロな視点での判断を提供します。
- **技術スタック**: FinBERT (yiyanghkust/finbert-tone)。
- **特徴**: システム1が「確信が持てない」場合、または特定のイベントが発生した際に「ウェイクオン」され、重厚な分析を実行します。

---

## 各モジュールの機能

### データ収集・加工
- **[data_loader.py](file:///c:/Users/ikaru/.gemini/antigravity/scratch/oxytocin-trade/src/data_loader.py)**: `yfinance` を通じて、TDK、東京エレクトロン、アドバンテストなどの主要銘柄の株価データを取得します。
- **[features.py](file:///c:/Users/ikaru/.gemini/antigravity/scratch/oxytocin-trade/src/features.py)**: RSI、ボリンジャーバンド、騰落率などのテクニカル指標を計算し、SNN用に正規化を行います。
- **[dataset.py](file:///c:/Users/ikaru/.gemini/antigravity/scratch/oxytocin-trade/src/dataset.py)**: 時系列データをウィンドウ分割し、SNN学習用のバイナリターゲット（上昇/下落）を生成します。

### 機械学習と推論
- **[train_snn.py](file:///c:/Users/ikaru/.gemini/antigravity/scratch/oxytocin-trade/src/train_snn.py)**: SNNモデルを学習させ、`snn_model_latest.pth` として保存します。
- **[hybrid_trader.py](file:///c:/Users/ikaru/.gemini/antigravity/scratch/oxytocin-trade/src/hybrid_trader.py)**: SNN（システム1）と感情分析（システム2）を統合した統合推論エンジン。
- **[rl_agent.py](file:///c:/Users/ikaru/.gemini/antigravity/scratch/oxytocin-trade/src/rl_agent.py)**: システム2を起動するかどうかを判断する強化学習エージェントの実装。
- **[train_rl.py](file:///c:/Users/ikaru/.gemini/antigravity/scratch/oxytocin-trade/src/train_rl.py)**: 精度とエネルギーコストを最適化するためのRLポリシーを学習させます。

### シミュレーションと可視化
- **[simulate_trading.py](file:///c:/Users/ikaru/.gemini/antigravity/scratch/oxytocin-trade/src/simulate_trading.py)**: 過去の市場データを用いたバックテストを実行し、成果をJSON形式で出力します。
- **[live_trader.py](file:///c:/Users/ikaru/.gemini/antigravity/scratch/oxytocin-trade/src/live_trader.py)**: リアルタイム仮想トレーディング用の常駐プログラム。
- **[dashboard/](file:///c:/Users/ikaru/.gemini/antigravity/scratch/oxytocin-trade/src/dashboard)**: Vite + React によるプレミアム・パフォーマンス・ダッシュボード。

---

## セットアップと実行

1. **依存ライブラリのインストール**:
   ```bash
   pip install torch pandas yfinance transformers ta python-dotenv requests
   ```

2. **APIキーの設定**:
   - [NEWS_API_KEY](https://newsapi.org/) または [ALPHA_VANTAGE_API_KEY](https://www.alphavantage.co/support/#api-key) を取得します。
   - `.env.example` を `.env` にコピーし、取得したキーを入力します。
   ```bash
   cp .env.example .env
   ```

3. **SNNの学習**:
   ```bash
   python train_snn.py
   ```

4. **RLウェイクオン・ポリシーの学習**:
   ```bash
   python train_rl.py
   ```

5. **バックテストの実行**:
   ```bash
   python simulate_trading.py
   ```

6. **リアルタイム仮想トレーディングの開始**:
   ```bash
   python live_trader.py
   ```
   - 取引時間（9:00〜15:00 JST）中に1分ごとにデータを取得・推論・取引を実行
   - 取引時間外は自動でスリープし、翌営業日の取引開始時に再開
   - ポートフォリオ状態は `live_state.json` に永続化
   - 対象銘柄はスクリプト先頭の `TICKERS` リストで設定可能

7. **ダッシュボードの表示**:
   ```bash
   cd dashboard
   npm install
   npm run preview
   ```

---

## プロジェクトの成果
本プロジェクトでは、人間の認知モデルに基づいた「ハイブリッド・インテリジェンス」を実装しました。
- **エネルギー効率**: SNNによる常時監視と、必要な時のみ重いモデルを起動するRLポリシー。
- **高精度**: 感情分析の適切なタイミングでの統合。
- **可視化**: バックテスト結果を即座に確認できるモダンなダッシュボード。
- **リアルタイム対応**: 常駐プログラムによる仮想リアルタイムトレーディング。

## 今後の展望
- 実際の暗号資産やFX市場への対応。
- システム2にマルチモーダルLLMを導入した更なる状況判断の強化。
- WebSocket によるダッシュボードのリアルタイム更新。
