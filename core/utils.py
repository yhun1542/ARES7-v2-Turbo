"""
ARES-Ultimate Core Utilities
=============================
공통 유틸리티 함수: 로깅, 타임존, NaN 처리, 설정 로더 등
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Union

import numpy as np
import pandas as pd
import structlog
import yaml

# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T")

# =============================================================================
# Timezone Utilities
# =============================================================================

# Standard Market Timezones
TZ_UTC = timezone.utc
TZ_NYC = "America/New_York"
TZ_SEOUL = "Asia/Seoul"


def to_utc(dt: datetime) -> datetime:
    """Convert datetime to UTC"""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=TZ_UTC)
    return dt.astimezone(TZ_UTC)


def to_market_time(dt: datetime, market: str = "US") -> datetime:
    """
    Convert datetime to market timezone
    
    Args:
        dt: datetime to convert
        market: "US" for NYSE/NASDAQ, "KR" for KRX
    
    Returns:
        Localized datetime
    """
    import pytz
    
    tz_map = {
        "US": TZ_NYC,
        "KR": TZ_SEOUL,
    }
    
    tz = pytz.timezone(tz_map.get(market, TZ_NYC))
    
    if dt.tzinfo is None:
        return tz.localize(dt)
    return dt.astimezone(tz)


def now_utc() -> datetime:
    """Get current UTC time"""
    return datetime.now(TZ_UTC)


def now_market(market: str = "US") -> datetime:
    """Get current market time"""
    return to_market_time(now_utc(), market)


def trading_day(dt: Optional[datetime] = None, market: str = "US") -> pd.Timestamp:
    """
    Get trading day for given datetime
    
    - Weekend → Friday
    - Holiday → Previous trading day (simplified, no holiday calendar)
    """
    if dt is None:
        dt = now_market(market)
    
    ts = pd.Timestamp(dt)
    
    # Weekend adjustment
    if ts.dayofweek == 5:  # Saturday
        ts -= pd.Timedelta(days=1)
    elif ts.dayofweek == 6:  # Sunday
        ts -= pd.Timedelta(days=2)
    
    return ts


# =============================================================================
# NaN Handling
# =============================================================================

def safe_divide(
    numerator: Union[float, np.ndarray, pd.Series],
    denominator: Union[float, np.ndarray, pd.Series],
    fill_value: float = 0.0
) -> Union[float, np.ndarray, pd.Series]:
    """
    Safe division handling zero/NaN denominators
    """
    if isinstance(numerator, (pd.Series, pd.DataFrame)):
        result = numerator / denominator
        return result.fillna(fill_value).replace([np.inf, -np.inf], fill_value)
    elif isinstance(numerator, np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = numerator / denominator
            result = np.where(np.isfinite(result), result, fill_value)
            return result
    else:
        if denominator == 0 or np.isnan(denominator):
            return fill_value
        return numerator / denominator


def fillna_forward(series: pd.Series, limit: int = 5) -> pd.Series:
    """Forward fill with limit"""
    return series.ffill(limit=limit)


def fillna_mean(df: pd.DataFrame, axis: int = 0) -> pd.DataFrame:
    """Fill NaN with column/row means"""
    return df.fillna(df.mean(axis=axis))


def clip_outliers(
    series: pd.Series,
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99
) -> pd.Series:
    """Clip outliers to percentile bounds"""
    lower = series.quantile(lower_percentile)
    upper = series.quantile(upper_percentile)
    return series.clip(lower, upper)


def winsorize(
    series: pd.Series,
    limits: tuple = (0.05, 0.05)
) -> pd.Series:
    """Winsorize series (replace extreme values)"""
    from scipy.stats import mstats
    return pd.Series(
        mstats.winsorize(series.values, limits=limits),
        index=series.index
    )


# =============================================================================
# Configuration Loading
# =============================================================================

@lru_cache(maxsize=10)
def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file with caching
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def get_env(
    key: str,
    default: Optional[T] = None,
    required: bool = False,
    cast_type: type = str
) -> T:
    """
    Get environment variable with type casting
    
    Args:
        key: Environment variable name
        default: Default value if not found
        required: Raise error if not found
        cast_type: Type to cast value to
    
    Returns:
        Environment variable value
    """
    value = os.environ.get(key)
    
    if value is None:
        if required:
            raise EnvironmentError(f"Required environment variable not set: {key}")
        return default
    
    if cast_type == bool:
        return value.lower() in ("true", "1", "yes", "on")
    
    return cast_type(value)


def get_api_key(provider: str) -> str:
    """
    Get API key from environment
    
    Args:
        provider: Provider name (e.g., "polygon", "fred")
    
    Returns:
        API key
    
    Raises:
        EnvironmentError if key not found
    """
    env_map = {
        "polygon": "POLYGON_API_KEY",
        "fred": "FRED_API_KEY",
        "sf1": "NASDAQ_DATA_LINK_API_KEY",
        "nasdaq": "NASDAQ_DATA_LINK_API_KEY",
        "sec": "SEC_API_KEY",
        "tavily": "TAVILY_API_KEY",
        "news": "NEWS_API_KEY",
        "gnews": "GNEWS_API_KEY",
        "alphavantage": "ALPHA_VANTAGE_API_KEY",
        "ibkr": "IB_CLIENT_ID",  # Not a key, but connection param
        "kis": "KIS_APP_KEY",
    }
    
    env_name = env_map.get(provider.lower())
    if not env_name:
        raise ValueError(f"Unknown provider: {provider}")
    
    return get_env(env_name, required=True)


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False
) -> structlog.BoundLogger:
    """
    Setup structured logging
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for file logging
        json_format: Use JSON format (for production)
    
    Returns:
        Configured logger
    """
    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(file_handler)
    
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger()


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a named logger"""
    return structlog.get_logger(name)


# =============================================================================
# Performance Metrics Calculation
# =============================================================================

def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate simple returns from price series"""
    return prices.pct_change().dropna()


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """Calculate log returns from price series"""
    return np.log(prices / prices.shift(1)).dropna()


def calculate_sharpe(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio
    
    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year
    
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    ann_return = excess_returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    
    return safe_divide(ann_return, ann_vol, 0.0)


def calculate_sortino(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    ann_return = excess_returns.mean() * periods_per_year
    
    downside_returns = returns[returns < 0]
    if len(downside_returns) < 2:
        return calculate_sharpe(returns, risk_free_rate, periods_per_year)
    
    downside_vol = downside_returns.std() * np.sqrt(periods_per_year)
    
    return safe_divide(ann_return, downside_vol, 0.0)


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown
    
    Returns:
        Maximum drawdown (negative value, e.g., -0.15 for 15% drawdown)
    """
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    return drawdowns.min()


def calculate_calmar(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown)
    """
    ann_return = returns.mean() * periods_per_year
    max_dd = abs(calculate_max_drawdown(returns))
    
    return safe_divide(ann_return, max_dd, 0.0)


def calculate_var(
    returns: pd.Series,
    confidence: float = 0.95
) -> float:
    """
    Calculate Value at Risk (Historical)
    
    Returns:
        VaR at specified confidence (negative value)
    """
    return returns.quantile(1 - confidence)


def calculate_cvar(
    returns: pd.Series,
    confidence: float = 0.95
) -> float:
    """
    Calculate Conditional VaR (Expected Shortfall)
    
    Returns:
        CVaR at specified confidence (negative value)
    """
    var = calculate_var(returns, confidence)
    return returns[returns <= var].mean()


# =============================================================================
# Data Validation
# =============================================================================

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list,
    name: str = "DataFrame"
) -> None:
    """Validate DataFrame has required columns"""
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def validate_date_range(
    start_date: datetime,
    end_date: datetime
) -> None:
    """Validate date range"""
    if start_date >= end_date:
        raise ValueError(f"start_date ({start_date}) must be before end_date ({end_date})")


# =============================================================================
# Retry Decorator
# =============================================================================

import asyncio
from functools import wraps


def retry_async(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Async retry decorator with exponential backoff
    
    Args:
        max_retries: Maximum number of retries
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier
        exceptions: Exceptions to catch
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        raise last_exception
            
            raise last_exception
        
        return wrapper
    return decorator


def retry_sync(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Sync retry decorator with exponential backoff
    """
    import time
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        raise last_exception
            
            raise last_exception
        
        return wrapper
    return decorator
