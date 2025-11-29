"""
Backtest Metrics
=================
Sharpe, Calmar, MDD, WinRate 등 성능 지표 계산
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from core.utils import safe_divide


@dataclass
class PerformanceMetrics:
    """성능 지표 컨테이너"""
    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    cumulative_return: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    downside_volatility: float = 0.0
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    
    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    
    # VaR/CVaR
    var_95: float = 0.0
    cvar_95: float = 0.0
    
    # Trade metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    
    # Period info
    n_days: int = 0
    n_years: float = 0.0
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "win_rate": self.win_rate,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "n_days": self.n_days,
        }
    
    def summary(self) -> str:
        """Generate summary string"""
        return (
            f"Performance Summary ({self.start_date} to {self.end_date})\n"
            f"{'=' * 50}\n"
            f"Total Return:      {self.total_return:>10.2%}\n"
            f"Annualized Return: {self.annualized_return:>10.2%}\n"
            f"Volatility:        {self.volatility:>10.2%}\n"
            f"Max Drawdown:      {self.max_drawdown:>10.2%}\n"
            f"Sharpe Ratio:      {self.sharpe_ratio:>10.2f}\n"
            f"Sortino Ratio:     {self.sortino_ratio:>10.2f}\n"
            f"Calmar Ratio:      {self.calmar_ratio:>10.2f}\n"
            f"Win Rate:          {self.win_rate:>10.2%}\n"
            f"VaR (95%):         {self.var_95:>10.2%}\n"
            f"CVaR (95%):        {self.cvar_95:>10.2%}\n"
        )


def calculate_returns_metrics(returns: pd.Series) -> Dict[str, float]:
    """Calculate return-based metrics"""
    if len(returns) < 2:
        return {}
    
    n_days = len(returns)
    n_years = n_days / 252
    
    # Cumulative return
    cum_return = (1 + returns).prod() - 1
    
    # Annualized return
    ann_return = (1 + cum_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Volatility
    volatility = returns.std() * np.sqrt(252)
    
    # Downside volatility
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 1 else volatility
    
    return {
        "total_return": cum_return,
        "annualized_return": ann_return,
        "volatility": volatility,
        "downside_volatility": downside_vol,
        "n_days": n_days,
        "n_years": n_years,
    }


def calculate_drawdown_metrics(returns: pd.Series) -> Dict[str, float]:
    """Calculate drawdown-based metrics"""
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    
    max_dd = drawdowns.min()
    avg_dd = drawdowns[drawdowns < 0].mean() if len(drawdowns[drawdowns < 0]) > 0 else 0
    
    # Max drawdown duration
    dd_duration = 0
    current_duration = 0
    max_duration = 0
    
    for dd in drawdowns:
        if dd < 0:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    
    return {
        "max_drawdown": max_dd,
        "avg_drawdown": avg_dd,
        "max_drawdown_duration": max_duration,
        "drawdown_series": drawdowns,
    }


def calculate_risk_adjusted_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> Dict[str, float]:
    """Calculate risk-adjusted return metrics"""
    metrics = calculate_returns_metrics(returns)
    dd_metrics = calculate_drawdown_metrics(returns)
    
    ann_return = metrics.get("annualized_return", 0)
    volatility = metrics.get("volatility", 1)
    downside_vol = metrics.get("downside_volatility", 1)
    max_dd = dd_metrics.get("max_drawdown", -1)
    
    # Sharpe ratio
    sharpe = safe_divide(ann_return - risk_free_rate, volatility, 0)
    
    # Sortino ratio
    sortino = safe_divide(ann_return - risk_free_rate, downside_vol, 0)
    
    # Calmar ratio
    calmar = safe_divide(ann_return, abs(max_dd), 0)
    
    # Omega ratio
    threshold = 0
    gains = returns[returns > threshold].sum()
    losses = abs(returns[returns < threshold].sum())
    omega = safe_divide(gains, losses, 1)
    
    return {
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "omega_ratio": omega,
    }


def calculate_var_cvar(returns: pd.Series, confidence: float = 0.95) -> Dict[str, float]:
    """Calculate VaR and CVaR"""
    alpha = 1 - confidence
    
    # Historical VaR
    var = returns.quantile(alpha)
    
    # Historical CVaR (Expected Shortfall)
    cvar = returns[returns <= var].mean()
    
    return {
        f"var_{int(confidence*100)}": var,
        f"cvar_{int(confidence*100)}": cvar,
    }


def calculate_trade_metrics(returns: pd.Series) -> Dict[str, float]:
    """Calculate trade-based metrics"""
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    n_trades = len(returns)
    n_wins = len(wins)
    n_losses = len(losses)
    
    # Win rate
    win_rate = n_wins / n_trades if n_trades > 0 else 0
    
    # Average win/loss
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    
    # Profit factor
    total_wins = wins.sum() if len(wins) > 0 else 0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 0
    profit_factor = safe_divide(total_wins, total_losses, 0)
    
    return {
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "n_trades": n_trades,
    }


def calculate_all_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0
) -> PerformanceMetrics:
    """
    Calculate all performance metrics
    
    Args:
        returns: Daily return series
        risk_free_rate: Annual risk-free rate
    
    Returns:
        PerformanceMetrics object
    """
    if len(returns) < 2:
        return PerformanceMetrics()
    
    # Get all metrics
    ret_metrics = calculate_returns_metrics(returns)
    dd_metrics = calculate_drawdown_metrics(returns)
    ra_metrics = calculate_risk_adjusted_metrics(returns, risk_free_rate)
    var_metrics = calculate_var_cvar(returns)
    trade_metrics = calculate_trade_metrics(returns)
    
    # Build PerformanceMetrics
    metrics = PerformanceMetrics(
        # Returns
        total_return=ret_metrics.get("total_return", 0),
        annualized_return=ret_metrics.get("annualized_return", 0),
        cumulative_return=ret_metrics.get("total_return", 0),
        
        # Risk
        volatility=ret_metrics.get("volatility", 0),
        downside_volatility=ret_metrics.get("downside_volatility", 0),
        max_drawdown=dd_metrics.get("max_drawdown", 0),
        avg_drawdown=dd_metrics.get("avg_drawdown", 0),
        max_drawdown_duration=dd_metrics.get("max_drawdown_duration", 0),
        
        # Risk-adjusted
        sharpe_ratio=ra_metrics.get("sharpe_ratio", 0),
        sortino_ratio=ra_metrics.get("sortino_ratio", 0),
        calmar_ratio=ra_metrics.get("calmar_ratio", 0),
        omega_ratio=ra_metrics.get("omega_ratio", 0),
        
        # VaR
        var_95=var_metrics.get("var_95", 0),
        cvar_95=var_metrics.get("cvar_95", 0),
        
        # Trade
        win_rate=trade_metrics.get("win_rate", 0),
        profit_factor=trade_metrics.get("profit_factor", 0),
        avg_win=trade_metrics.get("avg_win", 0),
        avg_loss=trade_metrics.get("avg_loss", 0),
        
        # Period
        n_days=ret_metrics.get("n_days", 0),
        n_years=ret_metrics.get("n_years", 0),
        start_date=str(returns.index[0]) if len(returns) > 0 else None,
        end_date=str(returns.index[-1]) if len(returns) > 0 else None,
    )
    
    return metrics


def compare_strategies(
    returns_dict: Dict[str, pd.Series],
    benchmark: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Compare multiple strategies
    
    Args:
        returns_dict: Dict of strategy name -> returns
        benchmark: Optional benchmark returns
    
    Returns:
        DataFrame comparing metrics
    """
    if benchmark is not None:
        returns_dict["Benchmark"] = benchmark
    
    results = []
    
    for name, returns in returns_dict.items():
        metrics = calculate_all_metrics(returns)
        results.append({
            "Strategy": name,
            "Total Return": f"{metrics.total_return:.2%}",
            "Ann. Return": f"{metrics.annualized_return:.2%}",
            "Volatility": f"{metrics.volatility:.2%}",
            "Max DD": f"{metrics.max_drawdown:.2%}",
            "Sharpe": f"{metrics.sharpe_ratio:.2f}",
            "Sortino": f"{metrics.sortino_ratio:.2f}",
            "Calmar": f"{metrics.calmar_ratio:.2f}",
        })
    
    return pd.DataFrame(results).set_index("Strategy")
