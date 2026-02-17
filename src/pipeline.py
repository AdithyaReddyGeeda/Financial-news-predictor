"""
End-to-End Pipeline Orchestrator.

Two modes:
  1. Training: Kaggle data + stock prices -> preprocess -> sentiment ->
     embeddings -> LSTM training -> evaluation
  2. Prediction: Live news scraping -> sentiment -> FAISS search ->
     LSTM inference -> results

Usage:
  python -m src.pipeline --mode train --tickers AAPL GOOGL MSFT
  python -m src.pipeline --mode predict --tickers AAPL
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import yaml

from .data_collection.kaggle_loader import KaggleNewsLoader
from .data_collection.stock_data import StockDataFetcher
from .data_collection.news_scraper import YahooRSSScraper
from .preprocessing.text_preprocessor import TextPreprocessor
from .preprocessing.feature_engineer import FeatureEngineer
from .sentiment.finbert_model import FinBERTSentiment
from .sentiment.embedding_store import EmbeddingStore
from .forecasting.dataset import create_data_splits
from .forecasting.trainer import StockTrainer
from .evaluation.metrics import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Full training pipeline: data -> model -> evaluation."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.kaggle_loader = KaggleNewsLoader(config_path)
        self.stock_fetcher = StockDataFetcher(config_path)
        self.text_processor = TextPreprocessor(config_path)
        self.feature_engineer = FeatureEngineer(config_path)
        self.finbert = FinBERTSentiment(config_path)
        self.embedding_store = EmbeddingStore(config_path)
        self.trainer = StockTrainer(config_path)
        self.evaluator = ModelEvaluator()

    def run(self, tickers: list[str] = None) -> dict:
        """
        Execute the full training pipeline.

        Steps:
            1. Load Kaggle news data
            2. Fetch stock price data
            3. Preprocess text
            4. Run FinBERT sentiment analysis
            5. Build FAISS embedding index
            6. Engineer features (technical indicators + sentiment)
            7. Train LSTM model
            8. Evaluate on test set

        Args:
            tickers: List of stock tickers. Defaults to config.

        Returns:
            Dict with evaluation results.
        """
        tickers = tickers or self.config["data"].get(
            "training_tickers",
            self.config["data"].get("default_tickers", ["AAPL", "GOOGL", "MSFT"]),
        )

        # Step 1: Load news data
        logger.info("=" * 60)
        logger.info("STEP 1: Loading Kaggle news dataset")
        logger.info("=" * 60)
        news_df = self.kaggle_loader.load()
        logger.info(f"Loaded {len(news_df)} headlines")

        # Step 2: Preprocess text
        logger.info("=" * 60)
        logger.info("STEP 2: Preprocessing headlines")
        logger.info("=" * 60)
        news_df = self.text_processor.process_dataframe(news_df)
        news_df = self.text_processor.deduplicate(news_df)

        # Step 3: Run sentiment analysis
        logger.info("=" * 60)
        logger.info("STEP 3: Running FinBERT sentiment analysis")
        logger.info("=" * 60)
        news_df = self.finbert.analyze_dataframe(news_df)

        # Step 4: Build embedding index
        logger.info("=" * 60)
        logger.info("STEP 4: Building FAISS embedding index")
        logger.info("=" * 60)
        headlines = news_df["headline"].tolist()
        embeddings = self.finbert.extract_embeddings(headlines)

        metadata = news_df[["headline", "date", "ticker", "label", "sentiment_score"]].to_dict("records")
        self.embedding_store.build_index(embeddings, metadata)
        self.embedding_store.save()

        # Step 5: Build UNIVERSAL feature dataset from all training tickers
        training_tickers = tickers or self.config["data"].get(
            "training_tickers",
            self.config["data"].get("default_tickers", ["AAPL", "GOOGL", "MSFT"]),
        )

        logger.info("=" * 60)
        logger.info(f"STEP 5: Building universal dataset from {len(training_tickers)} stocks")
        logger.info("=" * 60)

        all_feature_dfs = []
        daily_sentiment = self.finbert.get_daily_sentiment(news_df)

        for ticker in training_tickers:
            try:
                logger.info(f"Fetching and processing {ticker}...")
                price_df = self.stock_fetcher.fetch(ticker)
                feature_df = self.feature_engineer.prepare_final_dataset(
                    price_df, daily_sentiment, ticker, normalize=False
                )
                feature_df["_ticker"] = ticker
                all_feature_dfs.append(feature_df)
                logger.info(f"  {ticker}: {len(feature_df)} rows")
            except Exception as e:
                logger.error(f"  Skipping {ticker}: {e}")
                continue

        if not all_feature_dfs:
            raise RuntimeError("No tickers processed successfully")

        # Combine all tickers into one dataset and normalize together
        combined_df = pd.concat(all_feature_dfs, ignore_index=True)
        logger.info(
            f"Combined dataset: {len(combined_df)} rows "
            f"from {len(all_feature_dfs)} stocks"
        )

        # Normalize features across ALL stocks at once (fit=True)
        combined_df = self.feature_engineer.normalize_features(combined_df, fit=True)
        logger.info("Normalized combined dataset with universal scaler")

        # Step 6: Train universal model
        logger.info("=" * 60)
        logger.info("STEP 6: Training universal LSTM model")
        logger.info("=" * 60)

        # Per-ticker chronological split to avoid data leakage, then combine
        train_dfs, val_dfs, test_dfs = [], [], []
        model_cfg = self.config.get("model", {})
        train_ratio = model_cfg.get("train_split", 0.70)
        val_ratio = model_cfg.get("val_split", 0.15)

        for ticker_label, group_df in combined_df.groupby("_ticker"):
            group_df = group_df.reset_index(drop=True)
            n = len(group_df)
            t_end = int(n * train_ratio)
            v_end = int(n * (train_ratio + val_ratio))
            train_dfs.append(group_df.iloc[:t_end])
            val_dfs.append(group_df.iloc[t_end:v_end])
            test_dfs.append(group_df.iloc[v_end:])

        train_combined = pd.concat(train_dfs, ignore_index=True)
        val_combined = pd.concat(val_dfs, ignore_index=True)
        test_combined = pd.concat(test_dfs, ignore_index=True)

        logger.info(
            f"Per-ticker splits — Train: {len(train_combined)}, "
            f"Val: {len(val_combined)}, Test: {len(test_combined)}"
        )

        from .forecasting.dataset import StockSequenceDataset
        seq_len = model_cfg.get("seq_len", 30)
        train_ds = StockSequenceDataset(train_combined, seq_len)
        val_ds = StockSequenceDataset(val_combined, seq_len)
        test_ds = StockSequenceDataset(test_combined, seq_len)

        history = self.trainer.train(train_ds, val_ds)

        # Step 7: Evaluate on combined test set
        logger.info("=" * 60)
        logger.info("STEP 7: Evaluating universal model")
        logger.info("=" * 60)

        predictions = self.trainer.predict(test_ds)

        actual_directions = []
        actual_returns = []
        for i in range(len(test_ds)):
            _, targets = test_ds[i]
            actual_directions.append(targets["direction"].item())
            actual_returns.append(targets["return"].item())

        actual_directions = np.array(actual_directions)
        actual_returns = np.array(actual_returns)

        overall_eval = self.evaluator.full_evaluation(
            actual_directions=actual_directions,
            predicted_directions=predictions["predicted_labels"],
            actual_returns=actual_returns,
            predicted_returns=predictions["returns"],
        )
        overall_eval["training_history"] = history

        logger.info(
            f"Universal Model — "
            f"Accuracy: {overall_eval['directional_accuracy']:.2%}, "
            f"Sharpe: {overall_eval['backtest']['strategy']['sharpe_ratio']:.2f}"
        )

        # Step 8: Per-ticker evaluation on individual test slices (from normalized data)
        logger.info("=" * 60)
        logger.info("STEP 8: Per-ticker evaluation")
        logger.info("=" * 60)

        all_results = {"_UNIVERSAL": overall_eval}

        for ticker_label, group_df in combined_df.groupby("_ticker"):
            try:
                group_df = group_df.reset_index(drop=True)
                n = len(group_df)
                test_start = int(n * 0.85)
                test_slice = group_df.iloc[test_start:].reset_index(drop=True)

                if len(test_slice) < (seq_len + 1):
                    continue

                test_ds_ticker = StockSequenceDataset(test_slice, seq_len)
                preds = self.trainer.predict(test_ds_ticker)

                act_dir, act_ret = [], []
                for i in range(len(test_ds_ticker)):
                    _, t = test_ds_ticker[i]
                    act_dir.append(t["direction"].item())
                    act_ret.append(t["return"].item())

                ticker_eval = self.evaluator.full_evaluation(
                    np.array(act_dir), preds["predicted_labels"],
                    np.array(act_ret), preds["returns"],
                )
                all_results[ticker_label] = ticker_eval

                logger.info(
                    f"  {ticker_label}: Accuracy={ticker_eval['directional_accuracy']:.2%}, "
                    f"Sharpe={ticker_eval['backtest']['strategy']['sharpe_ratio']:.2f}"
                )
            except Exception as e:
                logger.error(f"  {ticker_label} eval failed: {e}")
                continue

        # Save the fitted scaler for inference on new tickers
        self._save_scaler()

        self._save_results(all_results)
        return all_results

    def _save_scaler(self):
        """Save the fitted StandardScaler for use during inference."""
        import joblib
        scaler_path = os.path.join(
            self.config["model"].get("checkpoint_dir", "models"),
            "universal_scaler.joblib",
        )
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(self.feature_engineer.scaler, scaler_path)
        logger.info(f"Saved universal scaler to {scaler_path}")

    def _save_results(self, results: dict):
        """Save evaluation results summary."""
        os.makedirs("data/processed", exist_ok=True)

        summary = {}
        for ticker, result in results.items():
            summary[ticker] = {
                "directional_accuracy": result["directional_accuracy"],
                "f1_score": result["classification"]["f1"],
                "strategy_sharpe": result["backtest"]["strategy"]["sharpe_ratio"],
                "strategy_return": result["backtest"]["strategy"]["total_return"],
                "buyhold_return": result["backtest"]["buy_and_hold"]["total_return"],
                "excess_return": result["backtest"]["comparison"]["excess_return"],
            }

        summary_df = pd.DataFrame(summary).T
        summary_df.to_csv("data/processed/evaluation_summary.csv")
        logger.info("Saved evaluation summary to data/processed/evaluation_summary.csv")


class PredictionPipeline:
    """Live prediction pipeline: scrape -> analyze -> predict (universal model)."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.scraper = YahooRSSScraper(config_path)
        self.text_processor = TextPreprocessor(config_path)
        self.finbert = FinBERTSentiment(config_path)
        self.embedding_store = EmbeddingStore(config_path)
        self.stock_fetcher = StockDataFetcher(config_path)
        self.feature_engineer = FeatureEngineer(config_path)
        self.trainer = StockTrainer(config_path)

        # Load the universal scaler saved during training
        self._load_scaler()

    def _load_scaler(self):
        """Load the universal StandardScaler from disk."""
        import joblib
        scaler_path = os.path.join(
            self.config["model"].get("checkpoint_dir", "models"),
            "universal_scaler.joblib",
        )
        try:
            self.feature_engineer.scaler = joblib.load(scaler_path)
            self.feature_engineer._is_fitted = True
            logger.info(f"Loaded universal scaler from {scaler_path}")
        except FileNotFoundError:
            logger.warning(
                "No universal scaler found. Predictions will use per-ticker "
                "normalization. Run the training pipeline first for best results."
            )

    def predict(self, tickers: list[str] = None) -> dict:
        """
        Run prediction for given tickers using live news.

        Steps:
            1. Scrape latest news from Yahoo RSS
            2. Run FinBERT sentiment
            3. Find similar historical headlines via FAISS
            4. Fetch latest stock prices + compute indicators
            5. Run LSTM prediction

        Args:
            tickers: Tickers to predict. Defaults to config.

        Returns:
            Dict per ticker with predictions and analysis.
        """
        tickers = tickers or self.config["data"].get(
            "default_tickers", ["AAPL", "GOOGL", "MSFT"]
        )

        # Load saved embedding index
        try:
            self.embedding_store.load()
        except FileNotFoundError:
            logger.warning("No FAISS index found. Run training pipeline first.")

        results = {}

        for ticker in tickers:
            logger.info(f"Predicting for {ticker}...")

            try:
                result = self._predict_ticker(ticker)
                results[ticker] = result
                logger.info(
                    f"{ticker}: "
                    f"Direction={'UP' if result['predicted_direction'] == 1 else 'DOWN'}, "
                    f"Confidence={result['confidence']:.2%}, "
                    f"Sentiment={result['current_sentiment']:.3f}"
                )
            except Exception as e:
                logger.error(f"Prediction failed for {ticker}: {e}")
                continue

        return results

    def _predict_ticker(self, ticker: str) -> dict:
        """Generate prediction for a single ticker."""

        # Scrape latest news
        news_df = self.scraper.scrape_ticker(ticker)

        if news_df.empty:
            logger.warning(f"No news found for {ticker}")
            news_df = pd.DataFrame({"headline": ["No recent news available"]})

        # Clean text
        news_df = self.text_processor.process_dataframe(news_df)

        # Run sentiment
        news_df = self.finbert.analyze_dataframe(news_df)

        # Find similar historical headlines
        similar_news = []
        if self.embedding_store.size > 0:
            headlines = news_df["headline"].tolist()
            similar_results = self.embedding_store.search_text(
                headlines, self.finbert
            )
            for result_list in similar_results:
                similar_news.extend(result_list)

        # Get stock prices
        price_df = self.stock_fetcher.fetch(ticker)

        # Build features (skip normalization, apply universal scaler)
        daily_sentiment = self.finbert.get_daily_sentiment(news_df)
        feature_df = self.feature_engineer.prepare_final_dataset(
            price_df, daily_sentiment, ticker, normalize=False
        )

        # Apply the universal scaler (fit=False uses the saved scaler)
        if self.feature_engineer._is_fitted:
            feature_df = self.feature_engineer.normalize_features(
                feature_df, fit=False
            )
        else:
            feature_df = self.feature_engineer.normalize_features(
                feature_df, fit=True
            )

        # Load universal model and predict
        self.trainer.load_for_inference()
        from .forecasting.dataset import StockSequenceDataset
        predictions = self.trainer.predict(StockSequenceDataset(feature_df))

        latest_pred = predictions["predicted_labels"][-1] if len(predictions["predicted_labels"]) > 0 else 0
        latest_conf = predictions["directions"][-1] if len(predictions["directions"]) > 0 else 0.5

        return {
            "ticker": ticker,
            "predicted_direction": int(latest_pred),
            "predicted_direction_label": "UP" if latest_pred == 1 else "DOWN",
            "confidence": float(max(latest_conf, 1 - latest_conf)),
            "predicted_return": float(predictions["returns"][-1]) if len(predictions["returns"]) > 0 else 0.0,
            "current_sentiment": float(news_df["sentiment_score"].mean()) if "sentiment_score" in news_df.columns else 0.0,
            "news_count": len(news_df),
            "latest_headlines": news_df.head(5).to_dict("records"),
            "similar_historical": similar_news[:10],
        }


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Financial News Sentiment Stock Predictor"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        required=True,
        help="Pipeline mode: 'train' for full training, 'predict' for inference",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Stock tickers (e.g., AAPL GOOGL MSFT)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()

    if args.mode == "train":
        pipeline = TrainingPipeline(args.config)
        results = pipeline.run(args.tickers)

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE — UNIVERSAL MODEL RESULTS")
        print("=" * 60)

        if "_UNIVERSAL" in results:
            u = results["_UNIVERSAL"]
            print(f"\n*** UNIVERSAL MODEL (all stocks combined) ***")
            print(f"  Directional Accuracy: {u['directional_accuracy']:.2%}")
            print(f"  F1 Score:             {u['classification']['f1']:.2%}")
            print(f"  Strategy Sharpe:      {u['backtest']['strategy']['sharpe_ratio']:.2f}")
            print(f"  Strategy Return:      {u['backtest']['strategy']['total_return']:.2%}")

        print(f"\n--- Per-Ticker Breakdown ---")
        for ticker, result in results.items():
            if ticker == "_UNIVERSAL":
                continue
            print(f"\n{ticker}:")
            print(f"  Directional Accuracy: {result['directional_accuracy']:.2%}")
            print(f"  F1 Score:             {result['classification']['f1']:.2%}")
            print(f"  Strategy Sharpe:      {result['backtest']['strategy']['sharpe_ratio']:.2f}")
            print(f"  Strategy Return:      {result['backtest']['strategy']['total_return']:.2%}")
            print(f"  Buy&Hold Return:      {result['backtest']['buy_and_hold']['total_return']:.2%}")
            print(f"  Excess Return:        {result['backtest']['comparison']['excess_return']:.2%}")

    elif args.mode == "predict":
        pipeline = PredictionPipeline(args.config)
        results = pipeline.predict(args.tickers)

        print("\n" + "=" * 60)
        print("PREDICTIONS")
        print("=" * 60)
        for ticker, result in results.items():
            print(f"\n{ticker}:")
            print(f"  Direction:  {result['predicted_direction_label']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Sentiment:  {result['current_sentiment']:.3f}")
            print(f"  News Count: {result['news_count']}")


if __name__ == "__main__":
    main()
