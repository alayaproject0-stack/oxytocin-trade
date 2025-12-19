# プロジェクト仕様書：Hybrid Stock Trading Agent "Oxytocin-Trade"

## 1. プロジェクトの目的
「直感（テクニカル）」と「熟考（ファンダメンタルズ）」を使い分けるハイブリッドAIにより、情報の処理コスト（API代や計算資源）を最小限に抑えつつ、トレードの収益と勝率を最大化するエージェントを構築する。

## 2. ターゲット（投資対象）
あなたの関心が高い**「日本の半導体関連株」**をモデルケースとします。これらはボラティリティ（価格変動）が大きく、SNN（スパイク検知）と相性が良いためです。

*   **メイン対象**: TDK (6762)
*   **サブ/比較対象**: 東京エレクトロン (8035), アドバンテスト (6857)
*   **市場データ**: 日足（スイングトレード想定）または1時間足（デイトレード想定）
    *   ※まずはデータ取得が容易な「日足」または「1時間足」でのプロトタイプ作成を目指します。

## 3. システムアーキテクチャ詳細

### System 1 (The Intuition) - SNN分類器「テクニカル分析 SNN」
*   **入力**: 過去$N$期間の株価変動率、出来高、RSI（相対力指数）、ボリンジャーバンド位置
*   **出力**: 3クラス分類（買い / 売り /  様子見）
*   **特徴**: チャートの形状だけを見て、0.01秒で「上がりそう」かを判断する。

### System 2 (The Reason) - BERT分類器「ニュース・センチメント AI」
*   **入力**: 対象銘柄や半導体市場に関する最新ニュースのテキスト
*   **出力**: センチメントスコア（強気 / 弱気）
*   **動作**: LLM（軽量モデルまたはAPI）を使用し、文脈（「増益」や「工場新設」など）を理解する。

### The Gate (Oxytocin Policy) - 強化学習「リスク管理マネージャー」
*   **入力**: System 1の自信度(Confidence) + 市場の恐怖指数(VIX等)
*   **判断**: 「テクニカルだけでエントリーするか？」それとも「コストを払ってSystem 2にニュースを確認させるか？」
*   **学習目標**: 騙し（ダマシ）による損失回避と、調査コストの削減。

## 4. データフロー設計
1.  **Market Data Feed**: 株価データ（Open, High, Low, Close, Volume）を取得。
2.  **Feature Engineering**: SNN用に正規化（0.0〜1.0）およびスパイク符号化。
3.  **News Feed**: 関連ニュースの見出しを取得。
4.  **Agent Loop**:
    *   System 1がチャートを見る。
    *   Gateが「自信度」を評価。
    *   **High Confidence**: そのまま注文（Buy/Sell）。
    *   **Low Confidence**: System 2（ニュースAI）を起動 → ニュース判定と統合して最終判断。
    *   市場の結果（損益）を受け取り、Gate（オキシトシン係数）を更新。

## 5. オキシトシン報酬関数（Reward Function）の定義
このエージェントの「性格」を決定づける重要な数式です。

$$R = \text{PL} - C_{info} + \alpha_{oxy} \cdot (\text{Patience} + \text{Efficiency})$$

*   $\text{PL}$ (Profit/Loss): トレードによる損益（円）。
*   $C_{info}$ (Information Cost): System 2（ニュースAI）を起動した場合のペナルティ。
*   $\alpha_{oxy}$ (Oxytocin Bonus): 以下の行動に対するボーナス。
    *   **Patience (我慢)**: System 1が「買い」と言ったが、Gateが「怪しい」と止めて、結果的に暴落を回避できた場合（危機回避の成功）。
    *   **Efficiency (効率)**: System 2を使わずに、System 1だけで利益を出せた場合（自己効力感）。

## 6. 技術スタック（使用ツール）
*   言語: Python 3.10+
*   株価データ取得: yfinance (Yahoo Finance API・無料枠で開発可能)
*   テクニカル計算: ta (Technical Analysis Library) または pandas-ta
*   System 1 (SNN): torch (PyTorch) - LIFモデル
*   System 2 (NLP): transformers (Hugging Face) - 金融特化BERT (例: yiyanghkust/finbert-tone)
*   環境: Google Colab または ローカルGPU環境

## 7. 開発フェーズ（ロードマップ）
*   **Phase 1**: データ収集と前処理（Data Pipeline）
*   **Phase 2**: System 1（テクニカルSNN）の構築と学習
*   **Phase 3**: System 2（ニュースAI）の組み込み
*   **Phase 4**: Gate (RL) の統合
