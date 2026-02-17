"""
FinBERT Sentiment Analysis and Embedding Extraction.

Uses ProsusAI/finbert to:
1. Classify financial text as positive/negative/neutral with confidence scores
2. Extract 768-dim CLS token embeddings for vector similarity search
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yaml

logger = logging.getLogger(__name__)


class FinBERTSentiment:
    """Financial sentiment analysis using ProsusAI/finbert."""

    LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        sentiment_config = self.config.get("sentiment", {})
        self.model_name = sentiment_config.get("model_name", "ProsusAI/finbert")
        self.batch_size = sentiment_config.get("batch_size", 32)
        self.max_length = sentiment_config.get("max_length", 128)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._loaded = False

    def load_model(self):
        """Download and load the FinBERT model and tokenizer."""
        if self._loaded:
            return

        logger.info(f"Loading {self.model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True

        logger.info(f"Model loaded successfully on {self.device}")

    def predict_sentiment(self, texts: list[str]) -> list[dict]:
        """
        Predict sentiment for a list of texts.

        Args:
            texts: List of headline strings.

        Returns:
            List of dicts with keys:
                [label, positive, negative, neutral, score]
            where 'score' is a single float: positive - negative.
        """
        self.load_model()
        results = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            for prob in probs:
                label_idx = int(np.argmax(prob))
                results.append({
                    "label": self.LABEL_MAP[label_idx],
                    "sentiment_positive": float(prob[0]),
                    "sentiment_negative": float(prob[1]),
                    "sentiment_neutral": float(prob[2]),
                    "sentiment_score": float(prob[0] - prob[1]),
                })

        return results

    def extract_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Extract CLS token embeddings from FinBERT's last hidden layer.

        These 768-dim vectors capture the semantic meaning of headlines
        and are used for FAISS similarity search.

        Args:
            texts: List of headline strings.

        Returns:
            numpy array of shape (len(texts), 768).
        """
        self.load_model()
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # CLS token is the first token of the last hidden layer
                last_hidden = outputs.hidden_states[-1]
                cls_embeddings = last_hidden[:, 0, :].cpu().numpy()

            all_embeddings.append(cls_embeddings)

        embeddings = np.vstack(all_embeddings)
        logger.info(f"Extracted embeddings: shape {embeddings.shape}")
        return embeddings

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "headline",
    ) -> pd.DataFrame:
        """
        Run sentiment analysis on all headlines in a DataFrame.

        Adds columns: label, sentiment_positive, sentiment_negative,
        sentiment_neutral, sentiment_score.

        Args:
            df: DataFrame with a text column.
            text_column: Name of the column containing headlines.

        Returns:
            DataFrame with sentiment columns added.
        """
        df = df.copy()

        texts = df[text_column].tolist()
        logger.info(f"Analyzing sentiment for {len(texts)} headlines...")

        sentiments = self.predict_sentiment(texts)
        sentiment_df = pd.DataFrame(sentiments)

        for col in sentiment_df.columns:
            df[col] = sentiment_df[col].values

        logger.info(
            f"Sentiment distribution: "
            f"positive={sum(df['label'] == 'positive')}, "
            f"negative={sum(df['label'] == 'negative')}, "
            f"neutral={sum(df['label'] == 'neutral')}"
        )
        return df

    def get_daily_sentiment(
        self,
        df: pd.DataFrame,
        date_column: str = "date",
        ticker_column: Optional[str] = "ticker",
        aggregation: str = "mean",
    ) -> pd.DataFrame:
        """
        Aggregate sentiment scores to daily level per ticker.

        Args:
            df: DataFrame with sentiment columns and date column.
            date_column: Name of the date column.
            ticker_column: Name of the ticker column (optional).
            aggregation: 'mean' or 'median'.

        Returns:
            DataFrame with daily aggregated sentiment per ticker.
        """
        agg_func = aggregation if aggregation in ("mean", "median") else "mean"

        group_cols = [date_column]
        if ticker_column and ticker_column in df.columns:
            group_cols.append(ticker_column)

        sentiment_cols = [
            "sentiment_positive", "sentiment_negative",
            "sentiment_neutral", "sentiment_score",
        ]
        existing_cols = [c for c in sentiment_cols if c in df.columns]

        if not existing_cols:
            raise ValueError("No sentiment columns found. Run analyze_dataframe first.")

        daily = df.groupby(group_cols)[existing_cols].agg(agg_func).reset_index()

        # Add article count per day
        daily["article_count"] = (
            df.groupby(group_cols).size().reset_index(name="article_count")["article_count"]
        )

        logger.info(f"Aggregated to {len(daily)} daily sentiment records")
        return daily
