"""
FRED (Federal Reserve Economic Data) Client
=============================================
거시경제 데이터 제공 - Regime Filter 및 Risk Scaling에 사용

Features Generated:
- vix: 변동성 지수 → regime_filter, turbo_aarm
- term_spread: 10Y-2Y 스프레드 → regime_filter
- credit_spread: 하이일드 스프레드 → risk_scaling
- consumer_sentiment: 소비자 심리 → regime_filter
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

import aiohttp
import pandas as pd

from core.interfaces import IMacroProvider
from core.utils import get_env, get_logger, retry_async

logger = get_logger(__name__)


class FREDClient(IMacroProvider):
    """
    FRED (Federal Reserve Economic Data) Client
    
    Provides macro indicators for:
    - Regime detection (VIX, term spread)
    - Risk scaling (credit spread)
    - Economic context (unemployment, industrial production)
    """
    
    BASE_URL = "https://api.stlouisfed.org/fred"
    
    # Commonly used series
    SERIES_MAPPING = {
        "vix": "VIXCLS",
        "term_spread": "T10Y2Y",
        "credit_spread": "BAMLH0A0HYM2",
        "consumer_sentiment": "UMCSENT",
        "industrial_production": "INDPRO",
        "unemployment_rate": "UNRATE",
        "fed_funds_rate": "DFF",
        "sp500": "SP500",
        "real_gdp": "GDPC1",
        "cpi": "CPIAUCSL",
        "pce": "PCE",
        "t10y": "DGS10",
        "t2y": "DGS2",
        "baa_spread": "BAMLC0A4CBBB",
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED client
        
        Args:
            api_key: FRED API key (defaults to FRED_API_KEY env var)
        """
        self._api_key = api_key or get_env("FRED_API_KEY", required=True)
        self._session: Optional[aiohttp.ClientSession] = None
        self._connected = False
        
        # Cache for series data
        self._cache: Dict[str, pd.Series] = {}
    
    @property
    def name(self) -> str:
        return "fred"
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    async def connect(self) -> bool:
        """Initialize HTTP session"""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        self._connected = True
        logger.info("FRED client connected")
        return True
    
    async def disconnect(self) -> None:
        """Close HTTP session"""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        self._cache.clear()
        logger.info("FRED client disconnected")
    
    @retry_async(max_retries=3, delay=2.0, exceptions=(aiohttp.ClientError,))
    async def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make FRED API request"""
        if not self._session:
            await self.connect()
        
        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        params["api_key"] = self._api_key
        params["file_type"] = "json"
        
        async with self._session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                text = await response.text()
                logger.error(f"FRED API error {response.status}: {text}")
                raise aiohttp.ClientError(f"API error: {response.status}")
    
    async def get_prices(
        self,
        symbols: Sequence[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Not implemented - use get_macro_series instead"""
        raise NotImplementedError("FREDClient is for macro data - use get_macro_series")
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Not implemented - use get_latest_value instead"""
        raise NotImplementedError("Use get_latest_value for FRED data")
    
    async def get_macro_series(
        self,
        series_ids: Sequence[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch multiple FRED series
        
        Args:
            series_ids: FRED series IDs or aliases (e.g., ["vix", "term_spread"])
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with date index, series columns
        """
        all_series = []
        
        for series_id in series_ids:
            # Map alias to FRED ID
            fred_id = self.SERIES_MAPPING.get(series_id.lower(), series_id.upper())
            
            try:
                series = await self._fetch_series(fred_id, start_date, end_date)
                if series is not None and not series.empty:
                    series.name = series_id.lower()
                    all_series.append(series)
            except Exception as e:
                logger.warning(f"Failed to fetch {series_id}: {e}")
                continue
        
        if not all_series:
            return pd.DataFrame()
        
        df = pd.concat(all_series, axis=1)
        df = df.sort_index()
        
        logger.info(f"FRED data: {len(df)} rows, {len(df.columns)} series")
        return df
    
    async def _fetch_series(
        self,
        series_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.Series]:
        """Fetch a single FRED series"""
        # Check cache
        cache_key = f"{series_id}_{start_date.date()}_{end_date.date()}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        endpoint = "/series/observations"
        params = {
            "series_id": series_id,
            "observation_start": start_date.strftime("%Y-%m-%d"),
            "observation_end": end_date.strftime("%Y-%m-%d"),
            "sort_order": "asc",
        }
        
        result = await self._request(endpoint, params)
        
        if not result.get("observations"):
            return None
        
        data = []
        for obs in result["observations"]:
            date = pd.to_datetime(obs["date"])
            value = obs["value"]
            if value != ".":  # FRED uses "." for missing
                data.append({"date": date, "value": float(value)})
        
        if not data:
            return None
        
        series = pd.DataFrame(data).set_index("date")["value"]
        
        # Cache
        self._cache[cache_key] = series
        
        return series
    
    async def get_latest_value(self, series_id: str) -> Optional[float]:
        """Get latest value for a series"""
        fred_id = self.SERIES_MAPPING.get(series_id.lower(), series_id.upper())
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        series = await self._fetch_series(fred_id, start_date, end_date)
        
        if series is not None and not series.empty:
            return float(series.iloc[-1])
        
        return None
    
    async def get_vix(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.Series:
        """
        Get VIX series for regime filter
        
        Returns:
            Series with date index
        """
        df = await self.get_macro_series(["vix"], start_date, end_date)
        if df.empty:
            return pd.Series()
        return df["vix"]
    
    async def get_regime_indicators(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get all regime-relevant indicators
        
        Returns:
            DataFrame with: vix, term_spread, credit_spread
        """
        series_ids = ["vix", "term_spread", "credit_spread", "fed_funds_rate"]
        
        df = await self.get_macro_series(series_ids, start_date, end_date)
        
        # Forward fill to handle different frequencies
        df = df.ffill()
        
        return df
    
    async def calculate_regime_signals(
        self,
        as_of_date: datetime,
        lookback_days: int = 252
    ) -> Dict[str, Any]:
        """
        Calculate regime signals from macro data
        
        Returns:
            Dict with:
            - vix_level: Current VIX
            - vix_high: VIX > 25
            - vix_extreme: VIX > 30
            - term_spread_positive: 10Y > 2Y
            - credit_stress: Credit spread > historical 75th percentile
        """
        start_date = as_of_date - timedelta(days=lookback_days + 30)
        
        df = await self.get_regime_indicators(start_date, as_of_date)
        
        if df.empty:
            return {
                "vix_level": 20.0,
                "vix_high": False,
                "vix_extreme": False,
                "term_spread_positive": True,
                "credit_stress": False,
            }
        
        # Get latest values
        latest = df.iloc[-1]
        
        vix = latest.get("vix", 20.0)
        term_spread = latest.get("term_spread", 1.0)
        credit_spread = latest.get("credit_spread", 3.0)
        
        # Calculate percentiles
        if "credit_spread" in df.columns and len(df) > 20:
            credit_75pct = df["credit_spread"].quantile(0.75)
        else:
            credit_75pct = 4.0
        
        signals = {
            "vix_level": float(vix),
            "vix_high": vix > 25,
            "vix_extreme": vix > 30,
            "term_spread_positive": term_spread > 0,
            "term_spread_inverted": term_spread < -0.2,
            "credit_stress": credit_spread > credit_75pct,
            "credit_spread_level": float(credit_spread),
        }
        
        logger.debug(f"FRED regime signals: {signals}")
        return signals
