"""
SF1 (Sharadar) Fundamental Data Client
=======================================
재무 데이터 제공 - QM Overlay의 Quality Factor 계산에 사용

Features Generated:
- roe: Return on Equity
- roic: Return on Invested Capital
- grossmargin: Gross Margin
- currentratio: Current Ratio
- de_ratio: Debt/Equity Ratio (inverted)

Point-in-Time (PIT) 준수:
- 90일 지연으로 lookahead bias 방지
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

import aiohttp
import pandas as pd

from core.interfaces import IFundamentalProvider
from core.utils import get_env, get_logger, retry_async, safe_divide

logger = get_logger(__name__)


class SF1Client(IFundamentalProvider):
    """
    Sharadar SF1 Fundamental Data Client (via Nasdaq Data Link)
    
    Provides Point-in-Time (PIT) compliant fundamental data for:
    - Quality factor calculation (ROE, ROIC, Gross Margin, etc.)
    - Financial health screening
    """
    
    BASE_URL = "https://data.nasdaq.com/api/v3"
    DATASET = "SHARADAR/SF1"
    
    # Mapping of factor names to SF1 columns
    FACTOR_MAPPING = {
        "roe": ("netinc", "equity"),  # netinc / equity
        "roic": ("ebit", ("debt", "equity")),  # ebit / (debt + equity)
        "grossmargin": "gp",  # Direct field: gross profit / revenue
        "currentratio": "currentratio",  # Direct field
        "de_ratio": ("debt", "equity"),  # debt / equity
        "roa": ("netinc", "assets"),  # netinc / assets
        "fcf_yield": ("fcf", "marketcap"),  # fcf / marketcap
        "pe": "pe",  # Direct field
        "pb": "pb",  # Direct field
        "ps": "ps",  # Direct field
        "ev_ebitda": "evebitda",  # Direct field
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize SF1 client
        
        Args:
            api_key: API key (defaults to NASDAQ_DATA_LINK_API_KEY env var)
        """
        self._api_key = api_key or get_env("NASDAQ_DATA_LINK_API_KEY", required=True)
        self._session: Optional[aiohttp.ClientSession] = None
        self._connected = False
        
        # Cache for fundamental data
        self._cache: Dict[str, pd.DataFrame] = {}
    
    @property
    def name(self) -> str:
        return "sf1"
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    async def connect(self) -> bool:
        """Initialize HTTP session"""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        self._connected = True
        logger.info("SF1 client connected")
        return True
    
    async def disconnect(self) -> None:
        """Close HTTP session"""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        self._cache.clear()
        logger.info("SF1 client disconnected")
    
    @retry_async(max_retries=3, delay=2.0, exceptions=(aiohttp.ClientError,))
    async def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make API request"""
        if not self._session:
            await self.connect()
        
        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        params["api_key"] = self._api_key
        
        async with self._session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                text = await response.text()
                logger.error(f"SF1 API error {response.status}: {text}")
                raise aiohttp.ClientError(f"API error: {response.status}")
    
    async def get_prices(
        self,
        symbols: Sequence[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Not implemented for fundamental provider"""
        raise NotImplementedError("SF1Client is for fundamental data only")
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Not implemented for fundamental provider"""
        raise NotImplementedError("SF1Client is for fundamental data only")
    
    async def get_fundamentals(
        self,
        symbols: Sequence[str],
        metrics: Sequence[str],
        as_of_date: datetime,
        pit_delay_days: int = 90
    ) -> pd.DataFrame:
        """
        Get Point-in-Time fundamental data
        
        Args:
            symbols: List of ticker symbols
            metrics: List of metrics (e.g., ["roe", "roic", "grossmargin"])
            as_of_date: As-of date for PIT calculation
            pit_delay_days: Days to delay (default 90 for quarterly reports)
        
        Returns:
            DataFrame with symbol index, metric columns
        """
        # Calculate PIT cutoff date
        pit_date = as_of_date - timedelta(days=pit_delay_days)
        
        # Fetch raw SF1 data
        raw_data = await self._fetch_sf1_data(symbols, pit_date)
        
        if raw_data.empty:
            logger.warning("No SF1 data available")
            return pd.DataFrame()
        
        # Calculate requested metrics
        result = self._calculate_metrics(raw_data, metrics)
        
        logger.info(f"SF1 fundamentals: {len(result)} symbols, {len(metrics)} metrics")
        return result
    
    async def _fetch_sf1_data(
        self,
        symbols: Sequence[str],
        pit_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch raw SF1 data from Nasdaq Data Link
        """
        # Build filter for symbols
        symbols_str = ",".join(symbols[:100])  # API limit
        
        endpoint = f"/datatables/{self.DATASET}"
        params = {
            "ticker": symbols_str,
            "dimension": "ARQ",  # Annual Restated Quarterly
            "datekey.lte": pit_date.strftime("%Y-%m-%d"),
            "qopts.columns": "ticker,datekey,calendardate,reportperiod," + \
                            "revenue,gp,netinc,ebit,equity,debt,assets," + \
                            "currentratio,marketcap,fcf,pe,pb,ps,evebitda",
        }
        
        try:
            result = await self._request(endpoint, params)
            
            if not result.get("datatable", {}).get("data"):
                return pd.DataFrame()
            
            columns = [c["name"] for c in result["datatable"]["columns"]]
            data = result["datatable"]["data"]
            
            df = pd.DataFrame(data, columns=columns)
            df["datekey"] = pd.to_datetime(df["datekey"])
            
            # Keep only most recent report per ticker
            df = df.sort_values("datekey").groupby("ticker").last().reset_index()
            df = df.set_index("ticker")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch SF1 data: {e}")
            return pd.DataFrame()
    
    def _calculate_metrics(
        self,
        raw_data: pd.DataFrame,
        metrics: Sequence[str]
    ) -> pd.DataFrame:
        """
        Calculate fundamental metrics from raw SF1 data
        """
        result = pd.DataFrame(index=raw_data.index)
        
        for metric in metrics:
            if metric not in self.FACTOR_MAPPING:
                logger.warning(f"Unknown metric: {metric}")
                continue
            
            mapping = self.FACTOR_MAPPING[metric]
            
            if isinstance(mapping, str):
                # Direct field
                if mapping in raw_data.columns:
                    result[metric] = raw_data[mapping]
                else:
                    result[metric] = float("nan")
            
            elif isinstance(mapping, tuple) and len(mapping) == 2:
                numerator_col, denominator_col = mapping
                
                # Get numerator
                if numerator_col in raw_data.columns:
                    numerator = raw_data[numerator_col]
                else:
                    numerator = float("nan")
                
                # Get denominator (can be tuple for sum)
                if isinstance(denominator_col, tuple):
                    # Sum of columns
                    denom_values = []
                    for col in denominator_col:
                        if col in raw_data.columns:
                            denom_values.append(raw_data[col].fillna(0))
                    if denom_values:
                        denominator = sum(denom_values)
                    else:
                        denominator = float("nan")
                elif denominator_col in raw_data.columns:
                    denominator = raw_data[denominator_col]
                else:
                    denominator = float("nan")
                
                # Calculate ratio
                result[metric] = safe_divide(numerator, denominator, float("nan"))
        
        return result
    
    async def get_quality_scores(
        self,
        symbols: Sequence[str],
        as_of_date: datetime,
        pit_delay_days: int = 90
    ) -> pd.DataFrame:
        """
        Calculate composite quality scores for QM Overlay
        
        Quality factors:
        - ROE (higher is better)
        - ROIC (higher is better)
        - Gross Margin (higher is better)
        - Current Ratio (higher is better, but clip extremes)
        - D/E Ratio (lower is better - inverted)
        
        Returns:
            DataFrame with symbol index, columns: quality_score, quality_rank
        """
        metrics = ["roe", "roic", "grossmargin", "currentratio", "de_ratio"]
        
        fundamentals = await self.get_fundamentals(
            symbols, metrics, as_of_date, pit_delay_days
        )
        
        if fundamentals.empty:
            return pd.DataFrame()
        
        # Standardize each metric (z-score)
        standardized = pd.DataFrame(index=fundamentals.index)
        
        for metric in metrics:
            if metric not in fundamentals.columns:
                continue
            
            series = fundamentals[metric].dropna()
            
            # Winsorize extremes
            lower = series.quantile(0.01)
            upper = series.quantile(0.99)
            series = series.clip(lower, upper)
            
            # Standardize
            mean = series.mean()
            std = series.std()
            if std > 0:
                z_score = (series - mean) / std
            else:
                z_score = series * 0
            
            # Invert D/E (lower is better)
            if metric == "de_ratio":
                z_score = -z_score
            
            standardized[metric] = z_score
        
        # Composite score (equal weight)
        standardized["quality_score"] = standardized.mean(axis=1)
        standardized["quality_rank"] = standardized["quality_score"].rank(pct=True)
        
        logger.info(f"Quality scores calculated for {len(standardized)} symbols")
        
        return standardized[["quality_score", "quality_rank"]]


# =============================================================================
# Helper Functions
# =============================================================================

async def get_quality_momentum_scores(
    symbols: Sequence[str],
    as_of_date: datetime,
    polygon_client: "PolygonClient",
    sf1_client: SF1Client,
    quality_weight: float = 0.6,
    momentum_weight: float = 0.4,
    pit_delay_days: int = 90
) -> pd.DataFrame:
    """
    Calculate combined Quality-Momentum scores
    
    Args:
        symbols: List of symbols
        as_of_date: As-of date
        polygon_client: Polygon client for momentum
        sf1_client: SF1 client for quality
        quality_weight: Weight for quality factor
        momentum_weight: Weight for momentum factor
        pit_delay_days: PIT delay days
    
    Returns:
        DataFrame with columns: quality_rank, momentum_rank, qm_score, qm_rank
    """
    # Get quality scores
    quality = await sf1_client.get_quality_scores(
        symbols, as_of_date, pit_delay_days
    )
    
    # Get momentum factors
    momentum = await polygon_client.get_momentum_factors(symbols, as_of_date)
    
    # Merge
    if quality.empty and momentum.empty:
        return pd.DataFrame()
    
    result = pd.DataFrame(index=list(set(quality.index) | set(momentum.index)))
    
    if not quality.empty:
        result["quality_rank"] = quality["quality_rank"]
    else:
        result["quality_rank"] = 0.5
    
    if not momentum.empty:
        # Average of 6m and 12m momentum ranks
        momentum["momentum_score"] = (
            momentum["ret_6m"].rank(pct=True) + 
            momentum["ret_12m_skip_1m"].rank(pct=True)
        ) / 2
        result["momentum_rank"] = momentum["momentum_score"]
    else:
        result["momentum_rank"] = 0.5
    
    # Combined QM score
    result["qm_score"] = (
        quality_weight * result["quality_rank"].fillna(0.5) +
        momentum_weight * result["momentum_rank"].fillna(0.5)
    )
    result["qm_rank"] = result["qm_score"].rank(pct=True)
    
    return result
