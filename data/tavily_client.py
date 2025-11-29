"""
Tavily AI Search Client
========================
AI 검색 및 뉴스 이벤트 데이터 - 알파/리스크에 직접 활용

Features Generated (MUST be used in alpha/risk):
- news_count_24h: 24시간 뉴스 수 → 이상 변동성 예측
- news_sentiment: 센티먼트 점수 → QM Overlay 보정
- event_risk_flag: 고위험 이벤트 플래그 → CB 트리거/리스크 스케일링

고위험 이벤트 키워드:
- bankruptcy, SEC investigation, fraud
- CEO resignation, earnings miss
- product recall, lawsuit
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

import aiohttp

from core.interfaces import INewsProvider
from core.utils import get_env, get_logger, retry_async

logger = get_logger(__name__)


class TavilyClient(INewsProvider):
    """
    Tavily AI Search Client
    
    Provides AI-powered search and news analysis for:
    - Event risk detection
    - Sentiment scoring
    - News volume anomaly detection
    
    All features are designed to directly impact alpha/risk decisions.
    """
    
    BASE_URL = "https://api.tavily.com"
    
    # High-risk event keywords (trigger risk scaling)
    HIGH_RISK_KEYWORDS = [
        "bankruptcy",
        "SEC investigation",
        "fraud",
        "accounting irregularities",
        "CEO resignation",
        "CFO departure",
        "earnings miss",
        "guidance cut",
        "product recall",
        "data breach",
        "antitrust",
        "class action lawsuit",
        "FDA rejection",
        "delisting",
        "default",
    ]
    
    # Sentiment keywords
    POSITIVE_KEYWORDS = [
        "beat expectations", "upgrade", "outperform",
        "record revenue", "strong growth", "acquisition",
        "dividend increase", "buyback", "expansion",
    ]
    
    NEGATIVE_KEYWORDS = [
        "miss expectations", "downgrade", "underperform",
        "revenue decline", "layoffs", "restructuring",
        "warning", "concern", "investigation",
    ]
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Tavily client
        
        Args:
            api_key: Tavily API key (defaults to TAVILY_API_KEY env var)
        """
        self._api_key = api_key or get_env("TAVILY_API_KEY", required=True)
        self._session: Optional[aiohttp.ClientSession] = None
        self._connected = False
        
        # Cache for recent searches
        self._cache: Dict[str, Dict] = {}
        self._cache_ttl = 1800  # 30 minutes
    
    @property
    def name(self) -> str:
        return "tavily"
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    async def connect(self) -> bool:
        """Initialize HTTP session"""
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers={"Content-Type": "application/json"}
            )
        self._connected = True
        logger.info("Tavily client connected")
        return True
    
    async def disconnect(self) -> None:
        """Close HTTP session"""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        self._cache.clear()
        logger.info("Tavily client disconnected")
    
    async def get_prices(
        self,
        symbols: Sequence[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> Any:
        """Not implemented - Tavily is for news/search"""
        raise NotImplementedError("TavilyClient is for news/search")
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Not implemented"""
        raise NotImplementedError("TavilyClient is for news/search")
    
    @retry_async(max_retries=2, delay=1.0, exceptions=(aiohttp.ClientError,))
    async def _search(
        self,
        query: str,
        search_depth: str = "advanced",
        max_results: int = 10,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute Tavily search"""
        if not self._session:
            await self.connect()
        
        url = f"{self.BASE_URL}/search"
        
        payload = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_answer": True,
        }
        
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains
        
        async with self._session.post(url, json=payload) as response:
            if response.status == 200:
                return await response.json()
            else:
                text = await response.text()
                logger.error(f"Tavily API error {response.status}: {text}")
                raise aiohttp.ClientError(f"API error: {response.status}")
    
    async def get_news(
        self,
        query: str,
        symbols: Optional[Sequence[str]] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for news articles
        
        Args:
            query: Search query
            symbols: Optional list of symbols to include in query
            from_date: Not directly supported, but affects query
            to_date: Not directly supported
            max_results: Maximum results to return
        
        Returns:
            List of news articles with title, content, url, score
        """
        # Build query
        if symbols:
            symbol_str = " OR ".join(symbols[:5])  # Limit symbols
            query = f"({query}) AND ({symbol_str})"
        
        # Add recency if from_date is recent
        if from_date and (datetime.now() - from_date).days < 7:
            query = f"{query} recent news"
        
        result = await self._search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            exclude_domains=["reddit.com", "twitter.com", "facebook.com"]
        )
        
        articles = []
        for item in result.get("results", []):
            articles.append({
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "url": item.get("url", ""),
                "score": item.get("score", 0),
                "published_date": item.get("published_date"),
            })
        
        return articles
    
    async def get_sentiment_score(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> float:
        """
        Calculate sentiment score for a symbol
        
        This score DIRECTLY affects QM overlay adjustment.
        
        Args:
            symbol: Stock symbol
            lookback_hours: Hours to look back
        
        Returns:
            Sentiment score (-1.0 to 1.0)
            - Positive: bullish news
            - Negative: bearish news
            - Near 0: neutral or mixed
        """
        # Check cache
        cache_key = f"sentiment_{symbol}_{lookback_hours}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        # Search for recent news
        query = f"{symbol} stock news"
        articles = await self.get_news(query, max_results=10)
        
        if not articles:
            return 0.0
        
        # Calculate sentiment from content
        positive_count = 0
        negative_count = 0
        total_weight = 0
        
        for article in articles:
            text = (article.get("title", "") + " " + article.get("content", "")).lower()
            score = article.get("score", 0.5)
            
            # Count keyword matches
            pos_matches = sum(1 for kw in self.POSITIVE_KEYWORDS if kw.lower() in text)
            neg_matches = sum(1 for kw in self.NEGATIVE_KEYWORDS if kw.lower() in text)
            
            positive_count += pos_matches * score
            negative_count += neg_matches * score
            total_weight += score
        
        if total_weight == 0:
            sentiment = 0.0
        else:
            # Normalize to [-1, 1]
            net_sentiment = (positive_count - negative_count) / total_weight
            sentiment = max(-1.0, min(1.0, net_sentiment * 0.3))  # Scale down
        
        # Cache
        self._set_cached(cache_key, sentiment)
        
        logger.debug(f"Sentiment for {symbol}: {sentiment:.3f}")
        return sentiment
    
    async def get_event_risk_flag(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> bool:
        """
        Check for high-risk events
        
        This flag DIRECTLY triggers risk scaling in AARM.
        When True: Reduce position size by cb_reduction_factor
        
        Args:
            symbol: Stock symbol
            lookback_hours: Hours to look back
        
        Returns:
            True if high-risk event detected
        """
        # Check cache
        cache_key = f"event_risk_{symbol}_{lookback_hours}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        # Search for risk events
        query = f"{symbol} {' OR '.join(self.HIGH_RISK_KEYWORDS[:5])}"
        articles = await self.get_news(query, max_results=5)
        
        risk_detected = False
        risk_reasons = []
        
        for article in articles:
            text = (article.get("title", "") + " " + article.get("content", "")).lower()
            
            for keyword in self.HIGH_RISK_KEYWORDS:
                if keyword.lower() in text:
                    risk_detected = True
                    risk_reasons.append(keyword)
        
        if risk_detected:
            logger.warning(f"Event risk for {symbol}: {risk_reasons}")
        
        # Cache
        self._set_cached(cache_key, risk_detected)
        
        return risk_detected
    
    async def get_news_volume(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get news volume metrics
        
        High news volume often precedes volatility - used in vol forecasting.
        
        Returns:
            Dict with:
            - count: Number of articles
            - unique_sources: Number of unique domains
            - is_abnormal: True if volume is unusually high
        """
        query = f"{symbol} stock"
        articles = await self.get_news(query, max_results=20)
        
        count = len(articles)
        unique_sources = len(set(a.get("url", "").split("/")[2] for a in articles if a.get("url")))
        
        # Abnormal if > 10 articles from > 5 sources in 24h
        is_abnormal = count > 10 and unique_sources > 5
        
        return {
            "count": count,
            "unique_sources": unique_sources,
            "is_abnormal": is_abnormal,
        }
    
    async def get_market_risk_summary(self) -> Dict[str, Any]:
        """
        Get overall market risk summary
        
        Used for global risk scaling across all positions.
        
        Returns:
            Dict with:
            - global_risk_level: 0-1 scale
            - risk_factors: List of identified risks
            - recommended_scale: Suggested position scale
        """
        # Search for macro risk events
        queries = [
            "stock market crash risk",
            "recession warning economy",
            "geopolitical risk market",
            "federal reserve policy concern",
        ]
        
        risk_factors = []
        total_risk = 0
        
        for query in queries:
            try:
                articles = await self.get_news(query, max_results=3)
                
                for article in articles:
                    text = (article.get("title", "") + " " + article.get("content", "")).lower()
                    
                    # Check for severe risk language
                    severe_keywords = ["crash", "crisis", "collapse", "plunge", "panic"]
                    if any(kw in text for kw in severe_keywords):
                        risk_factors.append(query.split()[0])
                        total_risk += 0.2
                        
            except Exception as e:
                logger.warning(f"Failed to search '{query}': {e}")
                continue
        
        global_risk_level = min(1.0, total_risk)
        
        # Recommended scale: 1.0 at 0 risk, 0.5 at max risk
        recommended_scale = 1.0 - (global_risk_level * 0.5)
        
        return {
            "global_risk_level": global_risk_level,
            "risk_factors": list(set(risk_factors)),
            "recommended_scale": recommended_scale,
        }
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        if key in self._cache:
            entry = self._cache[key]
            if datetime.now().timestamp() - entry["time"] < self._cache_ttl:
                return entry["value"]
        return None
    
    def _set_cached(self, key: str, value: Any) -> None:
        """Set cached value"""
        self._cache[key] = {
            "value": value,
            "time": datetime.now().timestamp()
        }
