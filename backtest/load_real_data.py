#!/usr/bin/env python3
"""
Real Data Loader for ARES-Ultimate
===================================
Polygon.io, FRED, Sharadar SF1 등 실제 데이터를 로드하는 모듈

더미 데이터 대신 실제 API를 통해 데이터를 가져옵니다.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from core.utils import get_logger
from data.polygon_client import PolygonClient, YFinanceFallback
from data.fred_client import FREDClient
from data.sf1_client import SF1Client

logger = get_logger(__name__)

# Load environment variables
load_dotenv()


class RealDataLoader:
    """
    Real Data Loader
    
    실제 데이터 소스에서 백테스트용 데이터를 로드합니다.
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        use_cache: bool = True
    ):
        """
        Initialize data loader
        
        Args:
            cache_dir: Cache directory path
            use_cache: Whether to use cached data
        """
        self.cache_dir = Path(cache_dir or os.getenv("DATA_CACHE_DIR", "./data_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        
        # Initialize clients
        self.polygon_client = PolygonClient()
        self.yfinance_client = YFinanceFallback()
        self.fred_client = FREDClient()
        self.sf1_client = SF1Client()
        
        self._connected = False
    
    async def connect(self) -> None:
        """Connect to all data sources"""
        if self._connected:
            return
        
        logger.info("Connecting to data sources...")
        
        await self.polygon_client.connect()
        await self.yfinance_client.connect()
        await self.fred_client.connect()
        await self.sf1_client.connect()
        
        self._connected = True
        logger.info("✅ All data sources connected")
    
    async def disconnect(self) -> None:
        """Disconnect from all data sources"""
        if not self._connected:
            return
        
        await self.polygon_client.disconnect()
        await self.yfinance_client.disconnect()
        await self.fred_client.disconnect()
        await self.sf1_client.disconnect()
        
        self._connected = False
        logger.info("Disconnected from data sources")
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{key}.parquet"
    
    def _load_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """Load data from cache"""
        if not self.use_cache:
            return None
        
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            logger.info(f"Loading from cache: {key}")
            return pd.read_parquet(cache_path)
        
        return None
    
    def _save_to_cache(self, key: str, df: pd.DataFrame) -> None:
        """Save data to cache"""
        if not self.use_cache:
            return
        
        cache_path = self._get_cache_path(key)
        df.to_parquet(cache_path)
        logger.info(f"Saved to cache: {key}")
    
    async def load_sp100_universe(self) -> List[str]:
        """
        Load S&P 100 universe
        
        Returns:
            List of ticker symbols
        """
        # S&P 100 tickers (OEX components)
        # This is a static list - in production, fetch from a data provider
        sp100_tickers = [
            "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
            "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C",
            "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
            "CVX", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FDX", "GD",
            "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC",
            "INTU", "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA",
            "MCD", "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT",
            "NEE", "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL",
            "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TSLA",
            "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT", "XOM"
        ]
        
        logger.info(f"Loaded S&P 100 universe: {len(sp100_tickers)} symbols")
        return sp100_tickers
    
    async def load_prices(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Load price data
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with date index, symbol columns (close prices)
        """
        cache_key = f"prices_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        
        # Try cache first
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached
        
        logger.info(f"Loading prices for {len(symbols)} symbols from {start_date} to {end_date}")
        
        try:
            # Try Polygon first
            df = await self.polygon_client.get_prices(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                interval="1d"
            )
            
            # Pivot to wide format (date x symbol)
            prices = df.pivot_table(
                index="date",
                columns="symbol",
                values="close"
            )
            
        except Exception as e:
            logger.warning(f"Polygon failed: {e}, falling back to yfinance")
            
            # Fallback to yfinance
            df = await self.yfinance_client.get_prices(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                interval="1d"
            )
            
            prices = df.pivot_table(
                index="date",
                columns="symbol",
                values="close"
            )
        
        # Forward fill missing values (up to 5 days)
        prices = prices.fillna(method="ffill", limit=5)
        
        # Drop symbols with too many missing values
        missing_pct = prices.isna().sum() / len(prices)
        valid_symbols = missing_pct[missing_pct < 0.2].index
        prices = prices[valid_symbols]
        
        logger.info(f"✅ Loaded prices: {len(prices)} days, {len(prices.columns)} symbols")
        
        # Save to cache
        self._save_to_cache(cache_key, prices)
        
        return prices
    
    async def load_spx_vix(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Load SPX and VIX data
        
        Args:
            start_date: Start date
            end_date: End date
        
        Returns:
            (spx_series, vix_series)
        """
        cache_key = f"spx_vix_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        
        # Try cache
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached["spx"], cached["vix"]
        
        logger.info(f"Loading SPX and VIX from {start_date} to {end_date}")
        
        try:
            # Load SPY as proxy for SPX
            spy_df = await self.polygon_client.get_prices(
                symbols=["SPY"],
                start_date=start_date,
                end_date=end_date,
                interval="1d"
            )
            spx = spy_df.set_index("date")["close"] * 10  # SPY * 10 ≈ SPX
            spx.name = "spx"
            
        except Exception as e:
            logger.warning(f"Polygon SPY failed: {e}, using yfinance")
            spy_df = await self.yfinance_client.get_prices(
                symbols=["SPY"],
                start_date=start_date,
                end_date=end_date,
                interval="1d"
            )
            spx = spy_df.set_index("date")["close"] * 10
            spx.name = "spx"
        
        # Load VIX from FRED
        try:
            vix = await self.fred_client.get_vix(
                start_date=start_date,
                end_date=end_date
            )
        except Exception as e:
            logger.warning(f"FRED VIX failed: {e}, using synthetic")
            # Fallback: synthetic VIX (mean 20, std 8)
            vix = pd.Series(
                20 + np.random.randn(len(spx)) * 8,
                index=spx.index,
                name="vix"
            ).clip(10, 80)
        
        # Align indices
        common_idx = spx.index.intersection(vix.index)
        spx = spx.loc[common_idx]
        vix = vix.loc[common_idx]
        
        logger.info(f"✅ Loaded SPX and VIX: {len(spx)} days")
        
        # Save to cache
        df = pd.DataFrame({"spx": spx, "vix": vix})
        self._save_to_cache(cache_key, df)
        
        return spx, vix
    
    async def load_fundamentals(
        self,
        symbols: List[str],
        as_of_date: datetime,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load fundamental data (SF1)
        
        Args:
            symbols: List of ticker symbols
            as_of_date: As-of date
            metrics: List of metrics (default: quality metrics)
        
        Returns:
            DataFrame with symbol index, metric columns
        """
        if metrics is None:
            metrics = ["roe", "roic", "grossmargin", "currentratio", "de"]
        
        cache_key = f"fundamentals_{as_of_date.strftime('%Y%m%d')}"
        
        # Try cache
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached
        
        logger.info(f"Loading fundamentals for {len(symbols)} symbols as of {as_of_date}")
        
        try:
            df = await self.sf1_client.get_fundamentals(
                symbols=symbols,
                metrics=metrics,
                as_of_date=as_of_date,
                pit_delay_days=90
            )
            
            logger.info(f"✅ Loaded fundamentals: {len(df)} symbols, {len(df.columns)} metrics")
            
            # Save to cache
            self._save_to_cache(cache_key, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load fundamentals: {e}")
            # Return empty DataFrame
            return pd.DataFrame(index=symbols, columns=metrics)


async def load_backtest_data(
    start_date: datetime,
    end_date: datetime,
    universe: str = "SP100",
    use_cache: bool = True
) -> Dict[str, any]:
    """
    Load all data needed for backtest
    
    Args:
        start_date: Start date
        end_date: End date
        universe: Universe name ("SP100", "SP500", etc.)
        use_cache: Whether to use cached data
    
    Returns:
        Dictionary with keys: prices, spx, vix, fundamentals, symbols
    """
    loader = RealDataLoader(use_cache=use_cache)
    
    try:
        await loader.connect()
        
        # Load universe
        if universe == "SP100":
            symbols = await loader.load_sp100_universe()
        else:
            raise ValueError(f"Unknown universe: {universe}")
        
        # Load prices
        prices = await loader.load_prices(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        # Load SPX and VIX
        spx, vix = await loader.load_spx_vix(
            start_date=start_date,
            end_date=end_date
        )
        
        # Load fundamentals (optional, can be None)
        fundamentals = None
        try:
            fundamentals = await loader.load_fundamentals(
                symbols=list(prices.columns),
                as_of_date=end_date
            )
        except Exception as e:
            logger.warning(f"Fundamentals not available: {e}")
        
        return {
            "prices": prices,
            "spx": spx,
            "vix": vix,
            "fundamentals": fundamentals,
            "symbols": list(prices.columns)
        }
        
    finally:
        await loader.disconnect()


if __name__ == "__main__":
    # Test data loading
    async def test():
        data = await load_backtest_data(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2024, 1, 1),
            universe="SP100"
        )
        
        print(f"Prices shape: {data['prices'].shape}")
        print(f"SPX length: {len(data['spx'])}")
        print(f"VIX length: {len(data['vix'])}")
        print(f"Symbols: {len(data['symbols'])}")
    
    asyncio.run(test())
