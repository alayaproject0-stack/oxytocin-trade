# Supabase デプロイガイド

このドキュメントでは、Oxytocin-Trade プロジェクトを Supabase にデプロイする方法を説明します。

## 前提条件

- [Supabase アカウント](https://supabase.com/) を作成済み
- Python 3.10+ がインストール済み
- (オプション) [Render.com](https://render.com/) アカウント（クラウド実行用）

---

## 0. Supabase CLI のインストール

Supabase CLI を使用してプロジェクトを管理できます。

### 方法 1: Scoop（推奨）

```powershell
# Scoop がインストールされていない場合は先にインストール
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex

# Supabase bucket を追加してインストール
scoop bucket add supabase https://github.com/supabase/scoop-bucket.git
scoop install supabase
```

### 方法 2: 手動ダウンロード

1. [Supabase CLI Releases](https://github.com/supabase/cli/releases/latest) にアクセス
2. `supabase_windows_amd64.exe` をダウンロード
3. ファイル名を `supabase.exe` に変更
4. PATH の通ったディレクトリ（例: `C:\Windows\System32` または `%USERPROFILE%\bin`）に配置

### 方法 3: npm（Node.js が必要）

```powershell
npm install -g supabase
```

### 確認

```powershell
supabase --version
```

---

## 1. Supabase プロジェクトの作成

1. [Supabase Dashboard](https://app.supabase.com/) にログイン
2. **New Project** をクリック
3. 以下を設定:
   - **Name**: `oxytocin-trade`（任意）
   - **Database Password**: 安全なパスワードを設定（後で使用）
   - **Region**: `Northeast Asia (Tokyo)` を推奨
4. **Create new project** をクリック

---

## 2. データベーススキーマの作成

### 方法 A: Supabase CLI（推奨）

プロジェクトがすでにCLIでセットアップされている場合:

```powershell
# 環境変数にアクセストークンを設定
$env:SUPABASE_ACCESS_TOKEN = "your_access_token"

# プロジェクトをリンク
supabase link --project-ref zsuistuqqoflambognpp

# マイグレーションをプッシュ
supabase db push
```

### 方法 B: SQL Editor（手動）

1. Supabase Dashboard で **SQL Editor** を開く
2. `src/supabase_schema.sql` の内容をコピーしてエディタに貼り付け
3. **Run** をクリックしてスキーマを作成

### 作成されるテーブル

| テーブル名 | 説明 |
|-----------|------|
| `portfolio` | 現在の残高を管理 |
| `positions` | 保有ポジション情報 |
| `trade_history` | 取引履歴 |
| `dashboard_summary` | ダッシュボード用サマリー |

---

## 3. API キーの取得

1. Supabase Dashboard で **Settings** → **API** へ移動
2. 以下の値をメモ:
   - **Project URL**: `https://xxxxxxxx.supabase.co`
   - **anon public key**: `eyJhbGci...`（公開キー）

> [!CAUTION]
> **service_role key** は外部に公開しないでください。通常は anon key を使用します。

---

## 4. 環境変数の設定

### ローカル実行の場合

`src/.env` ファイルを作成:

```bash
# src/.env
NEWS_API_KEY=your_news_api_key_here
ALPHA_VANTAGE_API_KEY=your_alphavantage_key_here

# Supabase
SUPABASE_URL=https://xxxxxxxx.supabase.co
SUPABASE_KEY=eyJhbGci... # anon public key
```

### Render.com デプロイの場合

Render Dashboard で環境変数を設定:
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `NEWS_API_KEY`
- `ALPHA_VANTAGE_API_KEY`

---

## 5. 依存関係のインストール

```bash
cd src
pip install -r requirements.txt
```

`requirements.txt` には以下が必要:
```
supabase
python-dotenv
```

---

## 6. クラウドトレーダーの実行

### ローカル実行

```bash
cd src
python live_trader_cloud.py
```

### Render.com へのデプロイ

1. GitHub にリポジトリをプッシュ
2. [Render Dashboard](https://dashboard.render.com/) で **New** → **Blueprint** を選択
3. リポジトリを接続して `src/render.yaml` を検出させる
4. 環境変数を設定
5. **Create Blueprint** をクリック

> [!TIP]
> `render.yaml` は Background Worker としてトレーダーを実行するよう設定済みです。

---

## 7. データの確認

Supabase Dashboard で以下を確認できます:

- **Table Editor** → `trade_history` で取引履歴を閲覧
- **Table Editor** → `positions` で現在のポジション確認
- **Table Editor** → `dashboard_summary` でパフォーマンスサマリー確認

---

## トラブルシューティング

### 「Supabase credentials not found」エラー

`.env` ファイルが正しい場所にあり、環境変数が設定されているか確認:

```bash
echo $SUPABASE_URL
echo $SUPABASE_KEY
```

### Row Level Security (RLS) エラー

スキーマ SQL を再実行して RLS ポリシーが作成されていることを確認。

### 接続タイムアウト

Supabase の Region が適切か確認（日本からは Tokyo リージョン推奨）。

---

## 参考リンク

- [Supabase Python Client](https://supabase.com/docs/reference/python/introduction)
- [Render.com Documentation](https://render.com/docs)
