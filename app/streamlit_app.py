"""
Financial News Sentiment Stock Predictor - Streamlit Dashboard.

Launch: streamlit run app/streamlit_app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yaml
import yfinance as yf

import joblib

from src.data_collection.stock_data import StockDataFetcher
from src.data_collection.news_scraper import YahooRSSScraper
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.preprocessing.feature_engineer import FeatureEngineer
from src.sentiment.finbert_model import FinBERTSentiment
from src.sentiment.embedding_store import EmbeddingStore
from src.forecasting.trainer import StockTrainer
from src.forecasting.dataset import StockSequenceDataset

st.set_page_config(
    page_title="Financial Sentiment Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.pos {background:#d4edda;color:#155724;padding:4px 10px;border-radius:12px;font-weight:600;display:inline-block;font-size:.85em}
.neg {background:#f8d7da;color:#721c24;padding:4px 10px;border-radius:12px;font-weight:600;display:inline-block;font-size:.85em}
.neu {background:#fff3cd;color:#856404;padding:4px 10px;border-radius:12px;font-weight:600;display:inline-block;font-size:.85em}
.hl-row {padding:12px 0;border-bottom:1px solid #eee}
</style>
""", unsafe_allow_html=True)

# Resolve project root so relative paths in config.yaml work regardless of cwd
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")


# ── Dynamic Ticker Lookup via Yahoo Finance Search ────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def _yf_search(query: str) -> list[dict]:
    """Search Yahoo Finance for a company name or ticker. Cached for 24h."""
    try:
        results = yf.Search(
            query, max_results=5, news_count=0, enable_fuzzy_query=True,
        ).quotes
        return results if results else []
    except Exception:
        return []


def resolve_input(text: str) -> str:
    """Resolve a company name or ticker to a Yahoo Finance ticker symbol."""
    text = text.strip()
    if not text:
        return text

    results = _yf_search(text)
    if results:
        return results[0].get("symbol", text.upper())

    return text.upper()


@st.cache_data(ttl=86400, show_spinner=False)
def _get_short_name(ticker: str) -> str:
    """Look up the short company name for a ticker symbol. Cached for 24h."""
    try:
        results = yf.Search(
            ticker, max_results=1, news_count=0,
        ).quotes
        if results:
            return results[0].get("shortname") or results[0].get("longname") or ticker
    except Exception:
        pass
    return ticker


def display_name(ticker: str) -> str:
    """Get a friendly display name like 'Apple Inc. (AAPL)' for any ticker."""
    name = _get_short_name(ticker)
    if name and name != ticker:
        return f"{name} ({ticker})"
    return ticker


CURRENCY_SYMBOLS = {
    "USD": "$", "INR": "₹", "GBP": "£", "EUR": "€", "JPY": "¥",
    "CNY": "¥", "KRW": "₩", "HKD": "HK$", "CAD": "C$", "AUD": "A$",
    "SGD": "S$", "CHF": "CHF ", "SEK": "kr", "NOK": "kr", "DKK": "kr",
    "NZD": "NZ$", "ZAR": "R", "BRL": "R$", "MXN": "MX$", "TWD": "NT$",
    "THB": "฿", "IDR": "Rp", "MYR": "RM", "PHP": "₱", "TRY": "₺",
    "RUB": "₽", "PLN": "zł", "ILS": "₪", "AED": "د.إ", "SAR": "﷼",
    "PKR": "₨", "BDT": "৳", "LKR": "Rs", "NGN": "₦", "EGP": "E£",
    "VND": "₫", "CLP": "CLP$", "COP": "COL$", "PEN": "S/.",
    "ARS": "AR$", "CZK": "Kč", "HUF": "Ft", "RON": "lei",
}


@st.cache_data(ttl=86400, show_spinner=False)
def get_currency_symbol(ticker: str) -> str:
    """Fetch the currency symbol for a ticker from Yahoo Finance."""
    try:
        info = yf.Ticker(ticker).fast_info
        code = getattr(info, "currency", None) or "USD"
        return CURRENCY_SYMBOLS.get(code.upper(), code + " ")
    except Exception:
        return "$"


@st.cache_resource
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@st.cache_resource
def get_finbert():
    m = FinBERTSentiment(CONFIG_PATH)
    m.load_model()
    return m


@st.cache_resource
def get_embedding_store():
    s = EmbeddingStore(CONFIG_PATH)
    try:
        s.load()
    except FileNotFoundError:
        pass
    return s


def badge(label, score):
    c = {"positive": "pos", "negative": "neg"}.get(label, "neu")
    return f'<span class="{c}">{label.upper()} ({score:+.2f})</span>'


# ── Sidebar ──────────────────────────────────────────────────────────────
config = load_config()

st.sidebar.title("Settings")

# Popular options shown as "Company (TICKER)"
POPULAR_TICKERS = [
    "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA",
    "META", "NVDA", "JPM", "BAC", "NFLX",
    "DIS", "V", "MA", "WMT", "KO", "PEP",
    "INTC", "AMD", "CRM", "PYPL", "UBER",
]

with st.sidebar.form(key="stock_form"):
    st.subheader("Add Stocks")
    custom_input = st.text_input(
        "Type a company name or ticker",
        placeholder="e.g. Apple, Reliance, Adani Power, Bitcoin",
        help="Enter a **company name** (Apple, Tesla, Adani Power, Bitcoin) or "
             "a **ticker symbol** (AAPL, RELIANCE.NS, BTC-USD). "
             "Separate multiple entries with commas. Press Enter or click Submit.",
    )

    date_range = st.slider(
        "Chart Days", 30, 365,
        value=config["dashboard"].get("default_chart_days", 90),
    )

    submitted = st.form_submit_button("Submit", type="primary", use_container_width=True)

# Resolve tickers after form submission (or on initial load)
custom_tickers = []
if custom_input:
    raw_entries = [e.strip() for e in custom_input.split(",") if e.strip()]
    custom_tickers = [resolve_input(e) for e in raw_entries]

all_ticker_options = list(dict.fromkeys(custom_tickers + POPULAR_TICKERS))

default_tickers = config["data"].get("default_tickers", ["AAPL", "GOOGL", "MSFT"])
default_selection = custom_tickers if custom_tickers else [t for t in default_tickers[:3] if t in all_ticker_options]

selected_tickers = st.sidebar.multiselect(
    "Select Stocks",
    options=all_ticker_options,
    default=default_selection,
    format_func=display_name,
)

if st.sidebar.button("🔄 Refresh Data", type="secondary", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Status**")
st.sidebar.markdown(
    f"LSTM Model: {'✅' if os.path.exists('models/best_model.pth') else '❌ Not trained'}"
)
st.sidebar.markdown(
    f"FAISS Index: {'✅' if os.path.exists(config['embeddings']['faiss_index_path']) else '❌ Not built'}"
)

# ── Header ───────────────────────────────────────────────────────────────
st.title("Financial News Sentiment Stock Predictor")
st.caption("Real-time sentiment analysis & stock price prediction — FinBERT + LSTM")

if not selected_tickers:
    st.warning("Select at least one ticker from the sidebar.")
    st.stop()

tab1, tab2, tab5, tab3, tab4 = st.tabs([
    "📰 Live Sentiment",
    "📈 Stock Price Chart",
    "🎯 Predictions",
    "🔍 Similar Headlines",
    "📊 Backtesting",
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Sentiment Feed
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Live Sentiment Feed")

    @st.cache_data(ttl=900, show_spinner=False)
    def fetch_news(tickers):
        scraper = YahooRSSScraper(CONFIG_PATH)
        prep = TextPreprocessor(CONFIG_PATH)
        name_map = {t: _get_short_name(t) for t in tickers}
        df = scraper.scrape_multiple(tickers, company_names=name_map)
        if not df.empty:
            df = prep.process_dataframe(df)
        return df

    with st.spinner("Scraping live news from Yahoo, Google News, Reddit & Seeking Alpha..."):
        news_df = fetch_news(tuple(selected_tickers))

    if news_df.empty:
        st.info("No recent news found. Try different stocks or click Refresh.")
    else:
        @st.cache_data(ttl=900, show_spinner=False)
        def run_sentiment(df_hash, _df):
            fb = get_finbert()
            return fb.analyze_dataframe(_df)

        with st.spinner("Running FinBERT..."):
            _sentiment_key = hash(tuple(sorted(selected_tickers)))
            news_df = run_sentiment(_sentiment_key, news_df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Articles", len(news_df))
        c2.metric("Positive", f"{(news_df['label']=='positive').mean():.0%}")
        c3.metric("Negative", f"{(news_df['label']=='negative').mean():.0%}")
        c4.metric("Avg Score", f"{news_df['sentiment_score'].mean():+.3f}")

        # ── Sentiment Shift Alerts ──
        if "ticker" in news_df.columns:
            for tk in selected_tickers:
                tk_news = news_df[news_df["ticker"] == tk]
                if tk_news.empty:
                    st.warning(f"**{display_name(tk)}** — No news found. Low coverage or unusual ticker.")
                    continue

                avg = tk_news["sentiment_score"].mean()
                neg_count = (tk_news["label"] == "negative").sum()
                pos_count = (tk_news["label"] == "positive").sum()
                total = len(tk_news)

                if avg < -0.3:
                    st.error(
                        f"**{display_name(tk)}** — Strongly negative sentiment "
                        f"({avg:+.3f}) — {neg_count} negative out of {total} articles"
                    )
                elif avg > 0.3:
                    st.success(
                        f"**{display_name(tk)}** — Strongly positive sentiment "
                        f"({avg:+.3f}) — {pos_count} positive out of {total} articles"
                    )
                elif neg_count > total * 0.6:
                    st.warning(
                        f"**{display_name(tk)}** — Majority negative coverage "
                        f"({neg_count}/{total} articles negative, avg: {avg:+.3f})"
                    )

        # ── Sentiment line graph (per article, newest to oldest) ──
        if len(news_df) > 1:
            sorted_news = news_df.sort_values("date", ascending=False).reset_index(drop=True)
            scores = sorted_news["sentiment_score"].values
            labels = sorted_news["label"].values
            colors = ["#4CAF50" if l == "positive" else "#ef5350" if l == "negative" else "#FFC107" for l in labels]

            # Build hover text with headline preview
            hover = [
                f"{display_name(r.get('ticker',''))} | {r.get('label','').upper()} ({r.get('sentiment_score',0):+.3f})<br>{r['headline'][:80]}"
                for _, r in sorted_news.iterrows()
            ]

            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=list(range(1, len(scores) + 1)),
                y=scores,
                mode="lines+markers",
                line=dict(color="#2196F3", width=2),
                marker=dict(color=colors, size=8, line=dict(width=1, color="white")),
                hovertext=hover,
                hoverinfo="text",
                name="Sentiment",
            ))
            fig_line.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig_line.update_layout(
                title="Sentiment Score per Article (newest → oldest)",
                yaxis_title="Sentiment Score",
                xaxis_title="Article #",
                height=320,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_line, width="stretch")

        st.subheader("Headlines")
        filter_options = ["All"] + [display_name(t) for t in selected_tickers]
        filt = st.selectbox("Filter", filter_options, key="sf")
        if filt == "All":
            show = news_df
        else:
            filt_ticker = selected_tickers[filter_options.index(filt) - 1]
            show = news_df[news_df.get("ticker") == filt_ticker] if "ticker" in news_df.columns else news_df

        for _, r in show.head(25).iterrows():
            b = badge(r.get("label", "neutral"), r.get("sentiment_score", 0))
            dt = r["date"].strftime("%b %d, %H:%M") if pd.notna(r.get("date")) else ""
            tk = r.get("ticker", "")
            tk_label = display_name(tk) if tk else ""
            st.markdown(
                f'<div class="hl-row">{b} &nbsp;<b>{tk_label}</b> &nbsp;{r["headline"]}'
                f' &nbsp;<span style="color:#999;font-size:.8em">{dt}</span></div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Stock Price Chart (Candlestick + Indicators)
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Stock Price Chart")

    pick = st.selectbox("Stock", selected_tickers, key="chart_tk", format_func=display_name)

    @st.cache_data(ttl=3600)
    def get_prices(ticker, days):
        f = StockDataFetcher(CONFIG_PATH)
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return f.fetch(ticker, start_date=start)

    pick_label = display_name(pick)

    try:
        with st.spinner(f"Loading {pick_label}..."):
            pdf = get_prices(pick, date_range)
    except Exception as e:
        st.error(
            f"Could not fetch data for **{pick_label}**. "
            f"Make sure the company name or ticker is valid.\n\n"
            f"Error: {e}"
        )
        pdf = pd.DataFrame()

    if pdf.empty:
        st.error(f"No data for {pick_label}.")
    else:
        fe = FeatureEngineer(CONFIG_PATH)
        pdf = fe.add_technical_indicators(pdf)

        # ── Main chart: Candlestick + BB + Volume + RSI ──
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f"{pick_label} Price", "Volume", "RSI (14)"),
        )

        fig.add_trace(go.Candlestick(
            x=pdf["Date"], open=pdf["Open"],
            high=pdf["High"], low=pdf["Low"], close=pdf["Close"],
            name="OHLC",
        ), row=1, col=1)

        if "bb_upper" in pdf.columns:
            fig.add_trace(go.Scatter(
                x=pdf["Date"], y=pdf["bb_upper"],
                line=dict(color="rgba(100,181,246,0.5)", width=1),
                name="BB Upper",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=pdf["Date"], y=pdf["bb_lower"],
                line=dict(color="rgba(100,181,246,0.5)", width=1),
                fill="tonexty", fillcolor="rgba(100,181,246,0.08)",
                name="BB Lower",
            ), row=1, col=1)

        vol_colors = [
            "#26a69a" if c >= o else "#ef5350"
            for c, o in zip(pdf["Close"], pdf["Open"])
        ]
        fig.add_trace(go.Bar(
            x=pdf["Date"], y=pdf["Volume"],
            marker_color=vol_colors, name="Volume",
        ), row=2, col=1)

        if "rsi" in pdf.columns:
            fig.add_trace(go.Scatter(
                x=pdf["Date"], y=pdf["rsi"],
                line=dict(color="#7E57C2", width=1.5), name="RSI",
            ), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(
            height=720, showlegend=False,
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, width="stretch")

        # ── Key metrics row ──
        lat = pdf.iloc[-1]

        _currency = get_currency_symbol(pick)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Close", f"{_currency}{lat['Close']:.2f}")
        m2.metric("RSI", f"{lat.get('rsi', 0):.1f}")
        m3.metric("MACD", f"{lat.get('macd', 0):.3f}")
        m4.metric("ATR", f"{lat.get('atr', 0):.2f}")
        m5.metric("BB %B", f"{lat.get('bb_pct', 0):.2f}")

        # ── MACD subplot ──
        if "macd" in pdf.columns:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(
                x=pdf["Date"], y=pdf["macd"],
                line=dict(color="#2196F3", width=1.5), name="MACD",
            ))
            fig_macd.add_trace(go.Scatter(
                x=pdf["Date"], y=pdf["macd_signal_line"],
                line=dict(color="#FF9800", width=1.5), name="Signal",
            ))
            hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in pdf["macd_hist"]]
            fig_macd.add_trace(go.Bar(
                x=pdf["Date"], y=pdf["macd_hist"],
                marker_color=hist_colors, name="Histogram",
            ))
            fig_macd.update_layout(
                title="MACD", height=300,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_macd, width="stretch")


# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — LSTM Predictions
# ══════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Stock Direction Predictions")

    model_path = "models/best_model.pth"
    scaler_path = "models/universal_scaler.joblib"

    if not os.path.exists(model_path):
        st.warning(
            "No trained model found. Train first:\n```\n"
            "python -m src.pipeline --mode train\n```"
        )
    else:
        st.info("Using universal LSTM model trained on 25 stocks. Predictions update every 15 min.")

        @st.cache_resource
        def load_predictor():
            trainer = StockTrainer(CONFIG_PATH)
            trainer.load_for_inference()
            fe = FeatureEngineer(CONFIG_PATH)
            if os.path.exists(scaler_path):
                fe.scaler = joblib.load(scaler_path)
                fe._is_fitted = True
            return trainer, fe

        pred_trainer, pred_fe = load_predictor()

        @st.cache_data(ttl=900, show_spinner=False)
        def run_prediction(ticker):
            """Full prediction pipeline for a single ticker."""
            try:
                fetcher = StockDataFetcher(CONFIG_PATH)
                price_df = fetcher.fetch(ticker)

                scraper = YahooRSSScraper(CONFIG_PATH)
                prep = TextPreprocessor(CONFIG_PATH)
                fb = get_finbert()

                news = scraper.scrape_ticker(ticker, company_name=_get_short_name(ticker))
                if news.empty:
                    news = pd.DataFrame({"headline": ["No recent news"]})
                news = prep.process_dataframe(news)
                news = fb.analyze_dataframe(news)

                daily_sent = fb.get_daily_sentiment(news)
                feature_df = pred_fe.prepare_final_dataset(
                    price_df, daily_sent, ticker, normalize=False
                )

                if pred_fe._is_fitted:
                    feature_df = pred_fe.normalize_features(feature_df, fit=False)
                else:
                    feature_df = pred_fe.normalize_features(feature_df, fit=True)

                dataset = StockSequenceDataset(feature_df)
                preds = pred_trainer.predict(dataset)

                latest_dir = float(preds["directions"][-1]) if len(preds["directions"]) > 0 else 0.5
                latest_ret = float(preds["returns"][-1]) if len(preds["returns"]) > 0 else 0.0
                latest_label = int(preds["predicted_labels"][-1]) if len(preds["predicted_labels"]) > 0 else 0

                sentiment_score = float(news["sentiment_score"].mean()) if "sentiment_score" in news.columns else 0.0
                news_count = len(news)

                last_price = price_df.iloc[-1]
                rsi_val = float(last_price.get("rsi", 0)) if "rsi" in price_df.columns else 0.0

                fe_temp = FeatureEngineer(CONFIG_PATH)
                price_with_ti = fe_temp.add_technical_indicators(price_df)
                last_ti = price_with_ti.iloc[-1]

                return {
                    "direction": "UP" if latest_label == 1 else "DOWN",
                    "confidence": float(max(latest_dir, 1 - latest_dir)) * 100,
                    "predicted_return": latest_ret * 100,
                    "sentiment": sentiment_score,
                    "news_count": news_count,
                    "rsi": float(last_ti.get("rsi", 0)),
                    "macd": float(last_ti.get("macd", 0)),
                    "bb_pct": float(last_ti.get("bb_pct", 0)),
                    "close": float(last_ti.get("Close", 0)),
                    "error": None,
                }
            except Exception as e:
                return {"error": str(e)}

        # Run predictions for all selected tickers
        st.markdown("---")

        for ticker in selected_tickers:
            name = display_name(ticker)
            currency = get_currency_symbol(ticker)

            with st.spinner(f"Predicting {name}..."):
                result = run_prediction(ticker)

            if result.get("error"):
                st.error(f"**{name}** — Prediction failed: {result['error']}")
                continue

            direction = result["direction"]
            confidence = result["confidence"]
            sentiment = result["sentiment"]
            pred_return = result["predicted_return"]

            dir_color = "#4CAF50" if direction == "UP" else "#ef5350"
            dir_icon = "▲" if direction == "UP" else "▼"
            sent_color = "#4CAF50" if sentiment > 0 else "#ef5350" if sentiment < 0 else "#FFC107"

            st.markdown(f"""
            <div style="border:2px solid {dir_color};border-radius:12px;padding:20px;margin-bottom:16px">
                <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap">
                    <div>
                        <h3 style="margin:0">{name}</h3>
                        <span style="color:#999">Close: {currency}{result['close']:.2f}</span>
                    </div>
                    <div style="text-align:center">
                        <span style="font-size:2.5em;font-weight:bold;color:{dir_color}">{dir_icon} {direction}</span><br>
                        <span style="font-size:1.1em;color:#666">Confidence: <b>{confidence:.1f}%</b></span>
                    </div>
                    <div style="text-align:right">
                        <span style="color:{sent_color};font-weight:600">Sentiment: {sentiment:+.3f}</span><br>
                        <span style="color:#666">RSI: {result['rsi']:.1f} | MACD: {result['macd']:.3f}</span><br>
                        <span style="color:#999;font-size:.85em">{result['news_count']} articles analyzed</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.caption(
            "Predictions are based on the universal LSTM model using technical indicators + live news sentiment. "
            "This is NOT financial advice. Model accuracy is ~52-58%."
        )


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — Similar Historical News (FAISS)
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Similar Historical News")

    emb_store = get_embedding_store()

    if emb_store.size == 0:
        st.warning(
            "FAISS index empty. Train first:\n```\n"
            "python -m src.pipeline --mode train --tickers AAPL GOOGL\n```"
        )
    else:
        st.info(f"Searching across {emb_store.size:,} indexed headlines")

        mode = st.radio("Mode", ["Custom headline", "Latest scraped news"], horizontal=True)

        if mode == "Custom headline":
            q = st.text_input(
                "Enter a headline:",
                placeholder="Apple reports record quarterly revenue",
            )
            if q:
                fb = get_finbert()
                with st.spinner("Searching..."):
                    res = emb_store.search_text([q], fb, top_k=10)
                    qs = fb.predict_sentiment([q])[0]

                st.markdown(f"**Query:** {badge(qs['label'], qs['sentiment_score'])}", unsafe_allow_html=True)
                st.markdown("---")

                if res and res[0]:
                    for i, r in enumerate(res[0], 1):
                        b = badge(r.get("label", "neutral"), r.get("sentiment_score", 0))
                        st.markdown(
                            f"**{i}.** {r.get('headline','')} {b}  "
                            f"— sim: **{r['similarity_score']:.3f}**  "
                            f"date: {r.get('date','N/A')}",
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No matches.")

        else:
            stk = st.selectbox("Stock", selected_tickers, key="sim_tk", format_func=display_name)
            if st.button("Search", type="primary"):
                with st.spinner(f"Scraping {display_name(stk)}..."):
                    sc = YahooRSSScraper(CONFIG_PATH)
                    pp = TextPreprocessor(CONFIG_PATH)
                    fb = get_finbert()
                    ldf = sc.scrape_ticker(stk, company_name=_get_short_name(stk))
                    if ldf.empty:
                        st.warning("No news found.")
                    else:
                        ldf = pp.process_dataframe(ldf)
                        ldf = fb.analyze_dataframe(ldf)
                        for _, row in ldf.head(5).iterrows():
                            st.markdown(f"### {row['headline']}")
                            st.markdown(badge(row.get("label","neutral"), row.get("sentiment_score",0)), unsafe_allow_html=True)
                            hits = emb_store.search_text([row["headline"]], fb, top_k=5)
                            if hits and hits[0]:
                                for j, h in enumerate(hits[0], 1):
                                    hb = badge(h.get("label","neutral"), h.get("sentiment_score",0))
                                    st.markdown(f"&nbsp;&nbsp;{j}. {h.get('headline','')} {hb} (sim:{h['similarity_score']:.3f})", unsafe_allow_html=True)
                            st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — Backtesting Results
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Backtesting Results")

    summary_path = "data/processed/evaluation_summary.csv"

    if not os.path.exists(summary_path):
        st.warning(
            "No results yet. Train first:\n```\n"
            "python -m src.pipeline --mode train --tickers AAPL GOOGL MSFT\n```"
        )
    else:
        sdf = pd.read_csv(summary_path, index_col=0)

        # Show universal model results first if present
        if "_UNIVERSAL" in sdf.index:
            row = sdf.loc["_UNIVERSAL"]
            st.markdown("### Universal Model (All Stocks Combined)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{row.get('directional_accuracy',0):.1%}")
            c2.metric("F1", f"{row.get('f1_score',0):.2%}")
            c3.metric("Sharpe", f"{row.get('strategy_sharpe',0):.2f}")
            c4.metric("Strategy Ret", f"{row.get('strategy_return',0):.1%}")
            st.markdown("---")

        st.subheader("Per-Stock Breakdown")
        for tk in sdf.index:
            if tk == "_UNIVERSAL":
                continue
            row = sdf.loc[tk]
            st.markdown(f"### {display_name(tk)}")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", f"{row.get('directional_accuracy',0):.1%}")
            c2.metric("F1", f"{row.get('f1_score',0):.2%}")
            c3.metric("Sharpe", f"{row.get('strategy_sharpe',0):.2f}")
            c4.metric("Strategy Ret", f"{row.get('strategy_return',0):.1%}")
            excess = row.get("excess_return", 0)
            c5.metric("Excess", f"{excess:+.1%}",
                       delta=f"{excess:+.1%}",
                       delta_color="normal" if excess > 0 else "inverse")
            st.markdown("---")

        ticker_sdf = sdf.drop("_UNIVERSAL", errors="ignore")
        if len(ticker_sdf) > 0:
            chart_labels = [display_name(t) for t in ticker_sdf.index]

            fig_c = go.Figure()
            fig_c.add_trace(go.Bar(
                x=chart_labels, y=ticker_sdf["strategy_return"] * 100,
                name="Strategy", marker_color="#4CAF50",
            ))
            fig_c.add_trace(go.Bar(
                x=chart_labels, y=ticker_sdf["buyhold_return"] * 100,
                name="Buy & Hold", marker_color="#2196F3",
            ))
            fig_c.update_layout(
                title="Strategy vs Buy & Hold Returns (%)",
                barmode="group", yaxis_title="Return (%)", height=400,
            )
            st.plotly_chart(fig_c, width="stretch")

            fig_s = go.Figure()
            fig_s.add_trace(go.Bar(
                x=chart_labels, y=ticker_sdf["strategy_sharpe"],
                marker_color="#9C27B0",
            ))
            fig_s.add_hline(y=1.2, line_dash="dash", line_color="red",
                            annotation_text="Target (1.2)")
            fig_s.update_layout(
                title="Sharpe Ratio by Stock",
                yaxis_title="Sharpe Ratio", height=350,
            )
            st.plotly_chart(fig_s, width="stretch")

st.markdown("---")
st.caption("Financial News Sentiment Predictor | FinBERT + LSTM | Not financial advice")
