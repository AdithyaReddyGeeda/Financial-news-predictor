"""
Text Preprocessing for Financial News Headlines.

Cleans, normalizes, and deduplicates news headlines before
sentiment analysis and embedding generation.
"""

import re
import logging

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Clean and normalize financial news text for NLP processing."""

    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    HTML_PATTERN = re.compile(r"<[^>]+>")
    EXTRA_WHITESPACE = re.compile(r"\s+")
    SPECIAL_CHARS = re.compile(r"[^\w\s.,!?;:'\"-]")

    NOISE_PHRASES = [
        "breaking:", "breaking news:", "update:",
        "exclusive:", "just in:", "alert:",
    ]

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.min_length = self.config["preprocessing"].get("min_headline_length", 10)
        self.max_length = self.config["preprocessing"].get("max_headline_length", 512)

    def clean_text(self, text: str) -> str:
        """Clean a single text string: remove HTML, URLs, special chars, noise phrases."""
        if not isinstance(text, str) or not text.strip():
            return ""

        text = self.HTML_PATTERN.sub("", text)
        text = self.URL_PATTERN.sub("", text)
        text = self.SPECIAL_CHARS.sub(" ", text)

        text_lower = text.lower()
        for phrase in self.NOISE_PHRASES:
            if text_lower.startswith(phrase):
                text = text[len(phrase):]
                break

        text = self.EXTRA_WHITESPACE.sub(" ", text).strip()

        if len(text) > self.max_length:
            text = text[: self.max_length]

        return text

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "headline",
        keep_original: bool = True,
    ) -> pd.DataFrame:
        """
        Clean all headlines in a DataFrame.

        Args:
            df: DataFrame with a text column.
            text_column: Name of the column to clean.
            keep_original: If True, keeps original text in '{col}_raw'.

        Returns:
            DataFrame with cleaned text column.
        """
        df = df.copy()

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")

        if keep_original:
            df[f"{text_column}_raw"] = df[text_column]

        logger.info(f"Cleaning {len(df)} headlines...")
        df[text_column] = df[text_column].apply(self.clean_text)

        initial_count = len(df)
        df = df[df[text_column].str.len() >= self.min_length]
        removed = initial_count - len(df)

        if removed > 0:
            logger.info(f"Removed {removed} headlines below min length ({self.min_length})")

        df = df.reset_index(drop=True)
        logger.info(f"Preprocessing complete: {len(df)} headlines retained")
        return df

    def deduplicate(
        self,
        df: pd.DataFrame,
        text_column: str = "headline",
        threshold: float = 0.95,
    ) -> pd.DataFrame:
        """
        Remove duplicate and near-duplicate headlines.

        Uses exact match first, then word-level Jaccard similarity
        within a sliding window for near-duplicates.
        """
        initial_count = len(df)

        df = df.drop_duplicates(subset=[text_column]).reset_index(drop=True)
        exact_removed = initial_count - len(df)
        logger.info(f"Removed {exact_removed} exact duplicates")

        if threshold < 1.0 and len(df) > 1:
            normalized = df[text_column].str.lower().str.strip()
            to_drop = set()

            for i in range(len(normalized)):
                if i in to_drop:
                    continue
                for j in range(i + 1, min(i + 50, len(normalized))):
                    if j in to_drop:
                        continue
                    sim = self._word_jaccard(normalized.iloc[i], normalized.iloc[j])
                    if sim >= threshold:
                        to_drop.add(j)

            if to_drop:
                df = df.drop(index=list(to_drop)).reset_index(drop=True)
                logger.info(
                    f"Removed {len(to_drop)} near-duplicates (threshold={threshold})"
                )

        total_removed = initial_count - len(df)
        logger.info(f"Deduplication: {total_removed} removed, {len(df)} remaining")
        return df

    @staticmethod
    def _word_jaccard(a: str, b: str) -> float:
        """Word-level Jaccard similarity between two strings."""
        if not a or not b:
            return 0.0
        set_a = set(a.split())
        set_b = set(b.split())
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0
