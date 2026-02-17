# Financial News Sentiment-Driven Stock Price Predictor

An end-to-end ML pipeline that scrapes financial news from 4 sources, extracts sentiment using FinBERT, builds vector embeddings for similarity search, and predicts stock price movements using a universal bidirectional LSTM trained on 25 diverse stocks. Deployed as an interactive 5-tab Streamlit dashboard with real-time predictions.

## Architecture

```
News Sources                    NLP Pipeline                     ML Pipeline
─────────────                   ────────────                     ───────────
Yahoo Finance RSS ─┐
Google News RSS  ──┤
Reddit (4 subs)  ──┤──► Text Preprocessing ──► FinBERT ──► Sentiment Scores
Seeking Alpha    ──┘         │                    │              │
                             │                    ▼              │
Kaggle Dataset ──────────────┘         Vector Embeddings         │
                                       (768-dim FAISS)           │
                                                                 ▼
yfinance OHLCV ──► Technical Indicators ──────────────► Feature Matrix
                   (RSI, MACD, BB, ATR, OBV)                │
                                                             ▼
                                              Universal BiLSTM Model
                                              (trained on 25 stocks)
                                                     │
                                          ┌──────────┼──────────┐
                                          ▼          ▼          ▼
                                     Direction   Confidence  Return Est.
                                     (UP/DOWN)   (sigmoid)   (magnitude)
                                          │
                                          ▼
                               Streamlit Dashboard (5 tabs)
```

## Key Features

### Multi-Source News Scraping (No API Keys)
- **Yahoo Finance RSS** — Direct company news feed
- **Google News RSS** — Broad financial news search
- **Reddit** — Scrapes r/wallstreetbets, r/stocks, r/investing, r/CryptoCurrency via public JSON API
- **Seeking Alpha RSS** — Professional analyst articles and earnings coverage
- All sources run in parallel per ticker with automatic deduplication

### NLP & Sentiment Analysis
- **ProsusAI/FinBERT** — Pre-trained financial sentiment model (positive/negative/neutral classification)
- **768-dimensional CLS token embeddings** extracted for each headline
- **FAISS similarity search** — Find historically similar news headlines in milliseconds
- **Sentiment shift alerts** — Real-time banners when a stock's sentiment turns strongly positive or negative

### Universal Stock Prediction Model
- **Single LSTM trained on 25 diverse stocks** across Tech, Finance, Healthcare, Energy, and Consumer sectors
- Works on **any tradeable stock** — not limited to training tickers
- **Bidirectional LSTM** with dual prediction heads (direction + return magnitude)
- **Universal scaler** — Features normalized consistently across all stocks
- **Per-ticker chronological splits** during training to prevent data leakage

### Technical Indicators
Computed via the `ta` library:
- **RSI** (Relative Strength Index) — Momentum oscillator
- **MACD** (Moving Average Convergence Divergence) — Trend following
- **Bollinger Bands** (upper, lower, %B) — Volatility
- **ATR** (Average True Range) — Volatility measure
- **OBV** (On-Balance Volume) — Volume-based confirmation

### Smart Stock Lookup
- Type **company names** instead of ticker symbols (e.g., "Apple", "Reliance", "Adani Power")
- Dynamic resolution via `yfinance.Search` — works for stocks, crypto, ETFs, international markets
- **Dynamic currency detection** — Displays prices in the correct currency (USD, INR, GBP, EUR, JPY, etc.)

### Interactive Dashboard (5 Tabs)
1. **Live Sentiment** — Real-time news scraping with sentiment analysis, alerts, per-article sentiment graph, and headline feed with filtering
2. **Stock Price Chart** — Interactive candlestick charts with RSI, MACD, Bollinger Bands overlays and key metrics
3. **Predictions** — LSTM direction prediction (UP/DOWN) with confidence %, sentiment score, and technical indicators for each selected stock
4. **Similar Headlines** — FAISS-powered search to find historically similar news and see how the market reacted
5. **Backtesting** — Strategy simulation results with Sharpe ratio, max drawdown, directional accuracy, and return comparison vs buy-and-hold

## Quick Start

### 1. Clone and Install

```bash
cd financial-news-predictor
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add Dataset

Download the [Sentiment Analysis for Financial News](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) dataset from Kaggle and place the CSV in the `dataset/` folder:

```
dataset/all-data.csv
```

The file should have two columns: `sentiment` and `headline` (the loader auto-detects headerless CSVs and handles Latin-1 encoding).

### 3. Train the Universal Model

```bash
python -m src.pipeline --mode train
```

This trains a single model on 25 stocks (configurable in `config.yaml`):

| Step | What Happens |
|------|-------------|
| 1 | Loads and preprocesses the Kaggle dataset (~5,000 headlines) |
| 2 | Runs FinBERT sentiment analysis on all headlines |
| 3 | Builds a FAISS vector embedding index (768-dim) |
| 4 | Fetches historical OHLCV data for 25 stocks via yfinance |
| 5 | Computes technical indicators and merges with sentiment features |
| 6 | Normalizes all features with a universal StandardScaler |
| 7 | Performs per-ticker chronological train/val/test splits |
| 8 | Trains the BiLSTM with early stopping and learning rate scheduling |
| 9 | Evaluates on test set — directional accuracy, Sharpe ratio, backtesting |
| 10 | Saves model checkpoint, universal scaler, and FAISS index |

**Output files:**
- `models/best_model.pth` — Trained model weights
- `models/universal_scaler.joblib` — Fitted feature scaler
- `data/embeddings/faiss_index.bin` — Vector embedding index
- `results/` — Evaluation metrics and backtest results

### 4. Run CLI Predictions

```bash
python -m src.pipeline --mode predict --tickers AAPL TSLA NVDA
```

Scrapes live news, analyzes sentiment, and predicts next-day direction for each ticker.

### 5. Launch Dashboard

```bash
python -m streamlit run app/streamlit_app.py
```

Open http://localhost:8501 — type any company name or ticker in the sidebar.

## Project Structure

```
financial-news-predictor/
├── app/
│   └── streamlit_app.py              # Streamlit dashboard (5 tabs)
├── src/
│   ├── data_collection/
│   │   ├── kaggle_loader.py          # Kaggle CSV loader (auto-detects headers, Latin-1)
│   │   ├── stock_data.py             # yfinance wrapper with local caching
│   │   └── news_scraper.py           # Multi-source scraper (Yahoo, Google, Reddit, Seeking Alpha)
│   ├── preprocessing/
│   │   ├── text_preprocessor.py      # Text cleaning, normalization, deduplication
│   │   └── feature_engineer.py       # Technical indicators, feature merging, normalization
│   ├── sentiment/
│   │   ├── finbert_model.py          # FinBERT sentiment + 768-dim embedding extraction
│   │   └── embedding_store.py        # FAISS index management (build, search, save/load)
│   ├── forecasting/
│   │   ├── dataset.py                # PyTorch sliding-window dataset
│   │   ├── lstm_model.py             # Bidirectional LSTM with dual heads (direction + return)
│   │   └── trainer.py                # Training loop, early stopping, checkpointing
│   ├── evaluation/
│   │   └── metrics.py                # Accuracy, Sharpe ratio, max drawdown, backtesting
│   └── pipeline.py                   # End-to-end orchestrator (TrainingPipeline + PredictionPipeline)
├── dataset/                          # Kaggle CSV (not tracked in git)
├── data/                             # Generated data — cached prices, embeddings
├── models/                           # Saved checkpoints and scaler (not tracked in git)
├── results/                          # Evaluation output and backtest results
├── config.yaml                       # All hyperparameters and settings
├── requirements.txt                  # Python dependencies
└── .gitignore                        # Excludes venv, data, models, dataset
```

## Configuration

All settings are centralized in `config.yaml`:

| Section | Key Settings |
|---------|-------------|
| `data` | `default_tickers`, `training_tickers` (25 stocks), `start_date`, `kaggle_dataset_path` |
| `indicators` | RSI period, MACD fast/slow/signal windows, Bollinger window/std |
| `model` | `seq_len` (30), `hidden_size` (128), `num_layers` (2), `dropout` (0.3), `lr` (0.001), `epochs` (100) |
| `embeddings` | FAISS index path, `top_k` for similarity search |
| `news` | `max_articles_per_ticker` |

### Training Tickers (Universal Model)

The model trains on 25 stocks across 5 sectors for generalization:

| Sector | Tickers |
|--------|---------|
| Tech | AAPL, GOOGL, MSFT, AMZN, META, NVDA, TSLA, AMD, CRM, INTC |
| Finance | JPM, BAC, GS, V, MA |
| Healthcare | JNJ, PFE, UNH |
| Energy | XOM, CVX |
| Consumer | WMT, KO, DIS, NFLX, NKE |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Sentiment Analysis | ProsusAI/FinBERT (HuggingFace Transformers) |
| Vector Embeddings | FAISS (Facebook AI Similarity Search) |
| Price Forecasting | PyTorch Bidirectional LSTM |
| Technical Indicators | ta (Technical Analysis library) |
| Stock Data | yfinance |
| News Scraping | requests + feedparser + BeautifulSoup |
| Dashboard | Streamlit + Plotly |
| Data Processing | pandas, NumPy, scikit-learn |
| Serialization | joblib (scaler), torch.save (model) |

## How It Works

### Prediction Flow (for any stock)

1. **Scrape live news** from 4 sources (Yahoo RSS, Google News, Reddit, Seeking Alpha)
2. **Analyze sentiment** with FinBERT — score each headline as positive/negative/neutral
3. **Compute daily sentiment** — aggregate headline scores by date
4. **Fetch price data** from yfinance and compute technical indicators
5. **Build feature matrix** — merge sentiment scores + technical indicators + price data
6. **Normalize features** using the universal scaler (fitted during training on 25 stocks)
7. **Run LSTM inference** on the last 30-day sequence window
8. **Output**: Direction (UP/DOWN), confidence (%), estimated return magnitude

### Sentiment Alert System

The dashboard flags unusual sentiment in real-time:
- **Red alert**: Average sentiment < -0.3 (strongly bearish news)
- **Green alert**: Average sentiment > +0.3 (strongly bullish news)
- **Yellow warning**: >60% of articles are negative
- **Gray warning**: No news found (low coverage risk)

## Performance Notes

- **Directional Accuracy**: ~52-58% (typical for sentiment-based stock prediction)
- **Best used as**: A research and sentiment monitoring tool, not a trading oracle
- **GPU**: Recommended for faster FinBERT inference, but CPU works fine
- **First run**: Downloads FinBERT model (~440MB) from HuggingFace
- **News cache**: Results cached for 15 minutes; click "Refresh News" for latest

## Disclaimer

This is a research and educational project. It is **not financial advice**. Stock market predictions are inherently uncertain, and model accuracy depends on market conditions, data quality, and many other factors. Do not use this tool as the sole basis for investment decisions.
