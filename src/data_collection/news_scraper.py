"""
Multi-Source Financial News Scraper.

Scrapes financial news headlines from:
  - Yahoo Finance RSS
  - Google News RSS
  - Reddit (r/wallstreetbets, r/stocks, r/investing, r/CryptoCurrency)
  - Seeking Alpha RSS
No API keys required — completely free.
"""

import hashlib
import logging
import time
from datetime import datetime
from typing import Optional
from urllib.parse import quote_plus

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yaml

logger = logging.getLogger(__name__)


class YahooRSSScraper:
    """Scrape financial news from Yahoo Finance RSS feeds."""

    YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"
    GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}"
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.max_articles = self.config["news"].get("max_articles_per_ticker", 500)
        self._seen_hashes: set[str] = set()

    def _hash_headline(self, headline: str) -> str:
        """Generate a hash for deduplication."""
        normalized = headline.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def _fetch_rss(self, url: str, source_label: str) -> feedparser.FeedParserDict:
        """Fetch an RSS URL with requests, then parse with feedparser."""
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": self.USER_AGENT},
                timeout=15,
            )
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
            logger.info(f"{source_label}: HTTP {resp.status_code}, {len(feed.entries)} entries")
            return feed
        except requests.RequestException as e:
            logger.error(f"{source_label} request failed: {e}")
            return feedparser.FeedParserDict(entries=[])

    def scrape_yahoo_rss(self, ticker: str) -> list[dict]:
        """
        Scrape news from Yahoo Finance RSS for a given ticker.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL').

        Returns:
            List of dicts with keys: [headline, date, source, ticker, url]
        """
        url = f"{self.YAHOO_RSS_URL}?s={quote_plus(ticker)}&region=US&lang=en-US"

        logger.info(f"Scraping Yahoo RSS for {ticker}")
        feed = self._fetch_rss(url, f"Yahoo RSS [{ticker}]")

        articles = []
        for entry in feed.entries:
            headline = entry.get("title", "").strip()
            if not headline:
                continue

            h = self._hash_headline(headline)
            if h in self._seen_hashes:
                continue
            self._seen_hashes.add(h)

            try:
                date = datetime(*entry.published_parsed[:6]) if entry.get("published_parsed") else None
            except (TypeError, ValueError):
                date = None

            articles.append({
                "headline": headline,
                "date": date,
                "source": "yahoo_finance",
                "ticker": ticker,
                "url": entry.get("link", ""),
            })

        logger.info(f"Found {len(articles)} articles for {ticker} from Yahoo RSS")
        return articles

    def scrape_google_news(self, query: str, ticker: str = "UNKNOWN") -> list[dict]:
        """
        Scrape news from Google News RSS as a fallback/supplement.

        Args:
            query: Search query (e.g., 'Apple stock' or 'AAPL').
            ticker: Ticker to tag results with.

        Returns:
            List of article dicts.
        """
        encoded_query = quote_plus(f"{query} stock finance")
        url = self.GOOGLE_NEWS_RSS.format(query=encoded_query)

        logger.info(f"Scraping Google News RSS for '{query}' → {url}")
        feed = self._fetch_rss(url, f"Google News [{query}]")

        articles = []
        for entry in feed.entries:
            headline = entry.get("title", "").strip()
            headline = BeautifulSoup(headline, "html.parser").get_text()

            if not headline:
                continue

            h = self._hash_headline(headline)
            if h in self._seen_hashes:
                continue
            self._seen_hashes.add(h)

            try:
                date = datetime(*entry.published_parsed[:6]) if entry.get("published_parsed") else None
            except (TypeError, ValueError):
                date = None

            articles.append({
                "headline": headline,
                "date": date,
                "source": "google_news",
                "ticker": ticker,
                "url": entry.get("link", ""),
            })

        logger.info(f"Found {len(articles)} articles for '{query}' from Google News")
        return articles

    def scrape_reddit(self, query: str, ticker: str = "UNKNOWN") -> list[dict]:
        """
        Scrape posts from financial subreddits using Reddit's public JSON API.

        Args:
            query: Search query (company name or ticker).
            ticker: Ticker to tag results with.

        Returns:
            List of article dicts.
        """
        subreddits = ["wallstreetbets", "stocks", "investing", "CryptoCurrency"]
        articles = []

        for sub in subreddits:
            url = (
                f"https://www.reddit.com/r/{sub}/search.json"
                f"?q={quote_plus(query)}&sort=new&restrict_sr=1&limit=15"
            )

            try:
                resp = requests.get(
                    url,
                    headers={"User-Agent": self.USER_AGENT},
                    timeout=10,
                )
                if resp.status_code != 200:
                    continue

                data = resp.json()
                posts = data.get("data", {}).get("children", [])

                for post in posts:
                    p = post.get("data", {})
                    headline = p.get("title", "").strip()
                    if not headline:
                        continue

                    h = self._hash_headline(headline)
                    if h in self._seen_hashes:
                        continue
                    self._seen_hashes.add(h)

                    created = p.get("created_utc")
                    date = datetime.utcfromtimestamp(created) if created else None

                    articles.append({
                        "headline": headline,
                        "date": date,
                        "source": f"reddit_r/{sub}",
                        "ticker": ticker,
                        "url": f"https://reddit.com{p.get('permalink', '')}",
                    })

            except Exception as e:
                logger.error(f"Reddit r/{sub} scrape failed for '{query}': {e}")
                continue

            time.sleep(0.3)

        logger.info(f"Found {len(articles)} posts for '{query}' from Reddit")
        return articles

    def scrape_seeking_alpha(self, ticker: str) -> list[dict]:
        """
        Scrape news headlines from Seeking Alpha's public RSS/Atom feed.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL').

        Returns:
            List of article dicts.
        """
        clean_ticker = ticker.split(".")[0].replace("-", "").upper()
        urls = [
            f"https://seekingalpha.com/api/sa/combined/{clean_ticker}.xml",
            f"https://seekingalpha.com/feed/tag/{clean_ticker.lower()}",
        ]

        articles = []

        for url in urls:
            try:
                feed = self._fetch_rss(url, f"Seeking Alpha [{ticker}]")

                for entry in feed.entries:
                    headline = entry.get("title", "").strip()
                    headline = BeautifulSoup(headline, "html.parser").get_text()
                    if not headline:
                        continue

                    h = self._hash_headline(headline)
                    if h in self._seen_hashes:
                        continue
                    self._seen_hashes.add(h)

                    try:
                        date = datetime(*entry.published_parsed[:6]) if entry.get("published_parsed") else None
                    except (TypeError, ValueError):
                        date = None

                    articles.append({
                        "headline": headline,
                        "date": date,
                        "source": "seeking_alpha",
                        "ticker": ticker,
                        "url": entry.get("link", ""),
                    })

                if articles:
                    break

            except Exception as e:
                logger.error(f"Seeking Alpha scrape failed for {ticker}: {e}")
                continue

        logger.info(f"Found {len(articles)} articles for {ticker} from Seeking Alpha")
        return articles

    def scrape_ticker(self, ticker: str, company_name: Optional[str] = None) -> pd.DataFrame:
        """
        Scrape all available news for a ticker from all sources.

        Sources: Yahoo Finance RSS, Google News RSS, Reddit, Seeking Alpha.

        Args:
            ticker: Stock ticker symbol.
            company_name: Full company name for broader search. Auto-resolved if None.

        Returns:
            DataFrame with columns: [headline, date, source, ticker, url]
        """
        all_articles = []

        # Yahoo Finance RSS
        all_articles.extend(self.scrape_yahoo_rss(ticker))

        # Google News RSS
        search_query = company_name or ticker
        all_articles.extend(self.scrape_google_news(search_query, ticker))

        # Reddit
        reddit_query = company_name or ticker
        all_articles.extend(self.scrape_reddit(reddit_query, ticker))

        # Seeking Alpha
        all_articles.extend(self.scrape_seeking_alpha(ticker))

        time.sleep(0.5)

        if not all_articles:
            logger.warning(f"No articles found for {ticker}")
            return pd.DataFrame(columns=["headline", "date", "source", "ticker", "url"])

        df = pd.DataFrame(all_articles)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date", ascending=False).reset_index(drop=True)

        if len(df) > self.max_articles:
            df = df.head(self.max_articles)

        return df

    def scrape_multiple(
        self,
        tickers: list[str],
        company_names: Optional[dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Scrape news for multiple tickers.

        Args:
            tickers: List of ticker symbols.
            company_names: Optional dict mapping ticker -> company name.

        Returns:
            Combined DataFrame of all articles.
        """
        company_names = company_names or {}
        all_dfs = []

        for ticker in tickers:
            name = company_names.get(ticker)
            df = self.scrape_ticker(ticker, name)
            all_dfs.append(df)
            time.sleep(1)

        if not all_dfs:
            return pd.DataFrame(columns=["headline", "date", "source", "ticker", "url"])

        combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Total articles scraped: {len(combined)} for {len(tickers)} tickers")
        return combined

    def clear_cache(self):
        """Reset the deduplication cache."""
        self._seen_hashes.clear()
