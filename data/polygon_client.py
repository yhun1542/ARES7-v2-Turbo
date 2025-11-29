"""
Polygon.io Data Client
=======================
가격 데이터 제공 - QM Overlay 및 Regime Filter에 사용

Features Generated:
- daily_ohlcv: 일봉 데이터 → 모멘텀 계산
- returns: 수익률 시리즈 → 변동성/드로다운
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

import aiohttp
import pandas as pd

from core.interfaces import IDataProvider
from core.utils import get_env, get_logger, retry_async

logger = get_logger(__name__)


class PolygonClient(IDataProvider):
    """
    Polygon.io REST API Client
    
    Primary data source for:
    - Daily OHLCV prices
    - Price momentum calculation
    - Volatility estimation
    """
    
    BASE_URL = "https://api.polygon.io"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Polygon client
        
        Args:
            api_key: API key (defaults to POLYGON_API_KEY env var)
        """
        self._api_key = api_key or get_env("POLYGON_API_KEY", required=True)
        self._session: Optional[aiohttp.ClientSession] = None
        self._connected = False
        
        # Rate limiting
        self._requests_per_minute = get_env("POLYGON_RPM", default=5, cast_type=int)
        self._last_request_time = 0.0
        self._request_interval = 60.0 / self._requests_per_minute
    
    @property
    def name(self) -> str:
        return "polygon"
    
    @property
    def is_connected(self) -> bool:
        return self._connected and self._session is not None
    
    async def connect(self) -> bool:
        """Initialize HTTP session"""
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self._api_key}"}
            )
        self._connected = True
        logger.info("Polygon client connected")
        return True
    
    async def disconnect(self) -> None:
        """Close HTTP session"""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        logger.info("Polygon client disconnected")
    
    async def _rate_limit(self) -> None:
        """Enforce rate limiting"""
        import time
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._request_interval:
            await asyncio.sleep(self._request_interval - elapsed)
        self._last_request_time = time.time()
    
    @retry_async(max_retries=3, delay=2.0, exceptions=(aiohttp.ClientError,))
    async def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make API request with rate limiting"""
        if not self._session:
            await self.connect()
        
        await self._rate_limit()
        
        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        params["apiKey"] = self._api_key
        
        async with self._session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            elif response.status == 429:
                logger.warning("Rate limited, waiting...")
                await asyncio.sleep(60)
                raise aiohttp.ClientError("Rate limited")
            else:
                text = await response.text()
                logger.error(f"Polygon API error {response.status}: {text}")
                raise aiohttp.ClientError(f"API error: {response.status}")
    
    async def get_prices(
        self,
        symbols: Sequence[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for multiple symbols
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date
            end_date: End date
            interval: Bar interval ("1d", "1h", etc.)
        
        Returns:
            DataFrame with MultiIndex (date, symbol)
            Columns: open, high, low, close, volume, vwap
        """
        # Map interval to Polygon timespan
        timespan_map = {
            "1d": "day",
            "1h": "hour",
            "5m": "minute",
        }
        timespan = timespan_map.get(interval, "day")
        multiplier = 1 if interval == "1d" else (1 if interval == "1h" else 5)
        
        all_data = []
        
        for symbol in symbols:
            try:
                data = await self._get_symbol_prices(
                    symbol,
                    start_date,
                    end_date,
                    timespan,
                    multiplier
                )
                if not data.empty:
                    data["symbol"] = symbol
                    all_data.append(data)
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.concat(all_data, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["date", "symbol"]).sort_index()
        
        logger.info(f"Fetched prices for {len(symbols)} symbols: {len(df)} rows")
        return df
    
    async def _get_symbol_prices(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timespan: str,
        multiplier: int
    ) -> pd.DataFrame:
        """Fetch prices for a single symbol"""
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_str}/{end_str}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000
        }
        
        result = await self._request(endpoint, params)
        
        if result.get("status") != "OK" or not result.get("results"):
            return pd.DataFrame()
        
        df = pd.DataFrame(result["results"])
        df = df.rename(columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
            "n": "transactions"
        })
        
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["date", "open", "high", "low", "close", "volume", "vwap"]]
        
        return df
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol"""
        endpoint = f"/v2/last/trade/{symbol}"
        
        try:
            result = await self._request(endpoint)
            if result.get("status") == "OK" and result.get("results"):
                return float(result["results"]["p"])
        except Exception as e:
            logger.warning(f"Failed to get latest price for {symbol}: {e}")
        
        return None
    
    async def get_daily_returns(
        self,
        symbols: Sequence[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Calculate daily returns from prices
        
        Returns:
            DataFrame with date index, symbol columns
        """
        prices = await self.get_prices(symbols, start_date, end_date, "1d")
        
        if prices.empty:
            return pd.DataFrame()
        
        # Pivot to get close prices per symbol
        close = prices["close"].unstack("symbol")
        returns = close.pct_change().dropna()
        
        return returns
    
    async def get_momentum_factors(
        self,
        symbols: Sequence[str],
        as_of_date: datetime,
        lookback_6m: int = 126,
        lookback_12m: int = 252,
        skip_recent: int = 21
    ) -> pd.DataFrame:
        """
        Calculate momentum factors for QM Overlay
        
        Returns:
            DataFrame with symbol index, columns: ret_6m, ret_12m_skip_1m
        """
        # Need enough history
        start_date = as_of_date - timedelta(days=lookback_12m + skip_recent + 30)
        
        prices = await self.get_prices(symbols, start_date, as_of_date, "1d")
        
        if prices.empty:
            return pd.DataFrame()
        
        close = prices["close"].unstack("symbol")
        
        factors = []
        
        for symbol in close.columns:
            series = close[symbol].dropna()
            if len(series) < lookback_6m + skip_recent:
                continue
            
            # 6-month return (skip most recent month)
            if len(series) >= lookback_6m + skip_recent:
                ret_6m = series.iloc[-(skip_recent + 1)] / series.iloc[-(lookback_6m + skip_recent)] - 1
            else:
                ret_6m = float("nan")
            
            # 12-month return (skip most recent month)
            if len(series) >= lookback_12m + skip_recent:
                ret_12m_skip = series.iloc[-(skip_recent + 1)] / series.iloc[-(lookback_12m + skip_recent)] - 1
            else:
                ret_12m_skip = float("nan")
            
            factors.append({
                "symbol": symbol,
                "ret_6m": ret_6m,
                "ret_12m_skip_1m": ret_12m_skip
            })
        
        df = pd.DataFrame(factors).set_index("symbol")
        logger.info(f"Calculated momentum for {len(df)} symbols")
        
        return df
    
    async def get_spy_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get SPY data for regime filter
        
        Returns:
            DataFrame with columns: close, ma200, ret_6m, ret_12m
        """
        # Need extra history for MA200
        extended_start = start_date - timedelta(days=250)
        
        prices = await self.get_prices(["SPY"], extended_start, end_date, "1d")
        
        if prices.empty:
            return pd.DataFrame()
        
        df = prices.xs("SPY", level="symbol").copy()
        
        # Calculate indicators
        df["ma200"] = df["close"].rolling(200).mean()
        df["ret_6m"] = df["close"].pct_change(126)
        df["ret_12m"] = df["close"].pct_change(252)
        
        # Filter to requested range
        df = df.loc[start_date:end_date]
        
        return df


# =============================================================================
# Fallback: yfinance
# =============================================================================

class YFinanceFallback(IDataProvider):
    """
    yfinance fallback for when Polygon is unavailable
    """
    
    def __init__(self):
        self._connected = False
    
    @property
    def name(self) -> str:
        return "yfinance"
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    async def connect(self) -> bool:
        try:
            import yfinance as yf
            self._yf = yf
            self._connected = True
            return True
        except ImportError:
            logger.error("yfinance not installed")
            return False
    
    async def disconnect(self) -> None:
        self._connected = False
    
    async def get_prices(
        self,
        symbols: Sequence[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch prices using yfinance"""
        import yfinance as yf
        
        # Map interval
        interval_map = {"1d": "1d", "1h": "1h", "5m": "5m"}
        yf_interval = interval_map.get(interval, "1d")
        
        all_data = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=yf_interval
                )
                
                if not df.empty:
                    df = df.reset_index()
                    df.columns = [c.lower() for c in df.columns]
                    df["symbol"] = symbol
                    df = df.rename(columns={"date": "date"})
                    all_data.append(df)
                    
            except Exception as e:
                logger.warning(f"yfinance failed for {symbol}: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.concat(all_data, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.set_index(["date", "symbol"]).sort_index()
        
        return df
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price"""
        import yfinance as yf
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get("regularMarketPrice") or info.get("currentPrice")
        except Exception:
            return None
