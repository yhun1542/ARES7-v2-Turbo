"""
ARES-Ultimate Core Package
===========================
핵심 인터페이스, 모델, 유틸리티 익스포트
"""

from core.interfaces import (
    # Enums
    Side,
    OrderType,
    TimeInForce,
    OrderStatus,
    Regime,
    SignalType,
    # Domain Objects
    Signal,
    Order,
    Fill,
    Position,
    PortfolioState,
    RiskMetrics,
    # Interfaces
    IDataProvider,
    IFundamentalProvider,
    IMacroProvider,
    INewsProvider,
    IStrategyEngine,
    IBroker,
    IRiskManager,
)

from core.utils import (
    # Timezone
    to_utc,
    to_market_time,
    now_utc,
    now_market,
    trading_day,
    # NaN handling
    safe_divide,
    fillna_forward,
    fillna_mean,
    clip_outliers,
    winsorize,
    # Config
    load_config,
    get_env,
    get_api_key,
    # Logging
    setup_logging,
    get_logger,
    # Metrics
    calculate_returns,
    calculate_sharpe,
    calculate_sortino,
    calculate_max_drawdown,
    calculate_calmar,
    calculate_var,
    calculate_cvar,
    # Decorators
    retry_async,
    retry_sync,
)

__version__ = "1.0.0"

__all__ = [
    # Enums
    "Side",
    "OrderType",
    "TimeInForce",
    "OrderStatus",
    "Regime",
    "SignalType",
    # Domain Objects
    "Signal",
    "Order",
    "Fill",
    "Position",
    "PortfolioState",
    "RiskMetrics",
    # Interfaces
    "IDataProvider",
    "IFundamentalProvider",
    "IMacroProvider",
    "INewsProvider",
    "IStrategyEngine",
    "IBroker",
    "IRiskManager",
    # Utilities
    "to_utc",
    "to_market_time",
    "now_utc",
    "now_market",
    "trading_day",
    "safe_divide",
    "fillna_forward",
    "fillna_mean",
    "clip_outliers",
    "winsorize",
    "load_config",
    "get_env",
    "get_api_key",
    "setup_logging",
    "get_logger",
    "calculate_returns",
    "calculate_sharpe",
    "calculate_sortino",
    "calculate_max_drawdown",
    "calculate_calmar",
    "calculate_var",
    "calculate_cvar",
    "retry_async",
    "retry_sync",
]
