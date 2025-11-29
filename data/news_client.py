"""
News API Client
================
뉴스 데이터 제공 - Tavily 대안/보조

Features Generated:
- headline_sentiment: 헤드라인 센티먼트 → QM Overlay
- news_volume_zscore: 비정상 뉴스량 → 이벤트 리스크
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

import aiohttp

from core.interfaces import INewsProvider
from core.utils import get_env, get_logger, retry_async

logger = get_logger(__name__)


class NewsClient(INewsProvider):
    """
    News API Client (supports NewsAPI.org and GNews)
    
    Fallback/supplement to Tavily for news data.
    """
    
    NEWSAPI_URL = "https://newsapi.org/v2"
    GNEWS_URL = "https://gnews.io/api/v4"
    
    def __init__(
        self,
        newsapi_key: Optional[str] = None,
        gnews_key: Optional[str] = None
    ):
        """
        Initialize News client
        
        Args:
            newsapi_key: NewsAPI.org key (defaults to NEWS_API_KEY env var)
            gnews_key: GNews key (defaults to GNEWS_API_KEY env var)
        """
        self._newsapi_key = newsapi_key or get_env("NEWS_API_KEY", default=None)
        self._gnews_key = gnews_key or get_env("GNEWS_API_KEY", default=None)
        
        if not self._newsapi_key and not self._gnews_key:
            logger.warning("No news API keys configured")
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._connected = False
    
    @property
    def name(self) -> str:
        return "news"
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    async def connect(self) -> bool:
        """Initialize HTTP session"""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        self._connected = True
        logger.info("News client connected")
        return True
    
    async def disconnect(self) -> None:
        """Close HTTP session"""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        logger.info("News client disconnected")
    
    async def get_prices(self, *args, **kwargs) -> Any:
        """Not implemented"""
        raise NotImplementedError("NewsClient is for news data")
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Not implemented"""
        raise NotImplementedError("NewsClient is for news data")
    
    @retry_async(max_retries=2, delay=1.0, exceptions=(aiohttp.ClientError,))
    async def _newsapi_request(
        self,
        endpoint: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make NewsAPI request"""
        if not self._session:
            await self.connect()
        
        if not self._newsapi_key:
            raise ValueError("NewsAPI key not configured")
        
        url = f"{self.NEWSAPI_URL}{endpoint}"
        params["apiKey"] = self._newsapi_key
        
        async with self._session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                text = await response.text()
                logger.error(f"NewsAPI error {response.status}: {text}")
                raise aiohttp.ClientError(f"API error: {response.status}")
    
    @retry_async(max_retries=2, delay=1.0, exceptions=(aiohttp.ClientError,))
    async def _gnews_request(
        self,
        endpoint: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make GNews request"""
        if not self._session:
            await self.connect()
        
        if not self._gnews_key:
            raise ValueError("GNews key not configured")
        
        url = f"{self.GNEWS_URL}{endpoint}"
        params["token"] = self._gnews_key
        
        async with self._session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                text = await response.text()
                logger.error(f"GNews error {response.status}: {text}")
                raise aiohttp.ClientError(f"API error: {response.status}")
    
    async def get_news(
        self,
        query: str,
        symbols: Optional[Sequence[str]] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search for news articles
        
        Tries NewsAPI first, falls back to GNews.
        """
        # Build query
        if symbols:
            query = f"{query} {' '.join(symbols[:3])}"
        
        articles = []
        
        # Try NewsAPI
        if self._newsapi_key:
            try:
                newsapi_articles = await self._search_newsapi(
                    query, from_date, to_date, max_results
                )
                articles.extend(newsapi_articles)
            except Exception as e:
                logger.warning(f"NewsAPI failed: {e}")
        
        # Try GNews if needed
        if len(articles) < max_results // 2 and self._gnews_key:
            try:
                gnews_articles = await self._search_gnews(
                    query, from_date, to_date, max_results - len(articles)
                )
                articles.extend(gnews_articles)
            except Exception as e:
                logger.warning(f"GNews failed: {e}")
        
        return articles[:max_results]
    
    async def _search_newsapi(
        self,
        query: str,
        from_date: Optional[datetime],
        to_date: Optional[datetime],
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Search NewsAPI"""
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(max_results, 100),
        }
        
        if from_date:
            params["from"] = from_date.strftime("%Y-%m-%dT%H:%M:%S")
        if to_date:
            params["to"] = to_date.strftime("%Y-%m-%dT%H:%M:%S")
        
        result = await self._newsapi_request("/everything", params)
        
        articles = []
        for item in result.get("articles", []):
            articles.append({
                "title": item.get("title", ""),
                "content": item.get("description", "") or item.get("content", ""),
                "url": item.get("url", ""),
                "source": item.get("source", {}).get("name", ""),
                "published_date": item.get("publishedAt"),
            })
        
        return articles
    
    async def _search_gnews(
        self,
        query: str,
        from_date: Optional[datetime],
        to_date: Optional[datetime],
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Search GNews"""
        params = {
            "q": query,
            "lang": "en",
            "max": min(max_results, 10),  # GNews limit
        }
        
        if from_date:
            params["from"] = from_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        if to_date:
            params["to"] = to_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        result = await self._gnews_request("/search", params)
        
        articles = []
        for item in result.get("articles", []):
            articles.append({
                "title": item.get("title", ""),
                "content": item.get("description", ""),
                "url": item.get("url", ""),
                "source": item.get("source", {}).get("name", ""),
                "published_date": item.get("publishedAt"),
            })
        
        return articles
    
    async def get_sentiment_score(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> float:
        """
        Calculate sentiment score from news headlines
        """
        from_date = datetime.now() - timedelta(hours=lookback_hours)
        
        articles = await self.get_news(
            query=f"{symbol} stock",
            from_date=from_date,
            max_results=20
        )
        
        if not articles:
            return 0.0
        
        # Simple keyword-based sentiment
        positive_words = {"beat", "surge", "gain", "profit", "growth", "upgrade", "record"}
        negative_words = {"miss", "fall", "loss", "decline", "downgrade", "warning", "concern"}
        
        pos_count = 0
        neg_count = 0
        
        for article in articles:
            text = article.get("title", "").lower()
            
            pos_count += sum(1 for w in positive_words if w in text)
            neg_count += sum(1 for w in negative_words if w in text)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        return (pos_count - neg_count) / total
    
    async def get_event_risk_flag(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> bool:
        """
        Check for high-risk events
        """
        risk_keywords = [
            "bankruptcy", "investigation", "fraud",
            "SEC", "lawsuit", "recall", "resignation"
        ]
        
        from_date = datetime.now() - timedelta(hours=lookback_hours)
        
        articles = await self.get_news(
            query=f"{symbol} stock {' '.join(risk_keywords[:3])}",
            from_date=from_date,
            max_results=5
        )
        
        for article in articles:
            text = (article.get("title", "") + " " + article.get("content", "")).lower()
            
            for keyword in risk_keywords:
                if keyword.lower() in text:
                    logger.warning(f"Event risk detected for {symbol}: {keyword}")
                    return True
        
        return False
