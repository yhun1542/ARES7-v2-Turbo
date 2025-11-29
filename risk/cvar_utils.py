"""
CVaR Utilities
===============
CVaR (Conditional Value at Risk) 및 리스크 계산 유틸리티

CVaR은 VaR을 초과하는 손실의 기대값으로,
테일 리스크를 더 정확하게 측정.

향후 Conservative Profile 튜닝에 사용.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from core.utils import get_logger, safe_divide

logger = get_logger(__name__)


class CVaRCalculator:
    """
    CVaR (Conditional Value at Risk) Calculator
    
    Methods:
    - Historical CVaR
    - Parametric CVaR (Normal, Student-t)
    - Cornish-Fisher adjusted VaR/CVaR
    - EVT (Extreme Value Theory) tail estimation
    """
    
    def __init__(
        self,
        confidence: float = 0.95,
        method: str = "historical"
    ):
        """
        Initialize CVaR calculator
        
        Args:
            confidence: Confidence level (e.g., 0.95 for 95%)
            method: "historical", "parametric", "cornish_fisher", "evt"
        """
        self.confidence = confidence
        self.method = method
        self.alpha = 1 - confidence
    
    def calculate_var(
        self,
        returns: pd.Series,
        method: Optional[str] = None
    ) -> float:
        """
        Calculate Value at Risk
        
        Args:
            returns: Return series
            method: Override default method
        
        Returns:
            VaR (negative value, e.g., -0.02 for 2% loss)
        """
        method = method or self.method
        
        if len(returns) < 20:
            return returns.quantile(self.alpha)
        
        if method == "historical":
            return self._var_historical(returns)
        elif method == "parametric":
            return self._var_parametric(returns)
        elif method == "cornish_fisher":
            return self._var_cornish_fisher(returns)
        elif method == "evt":
            return self._var_evt(returns)
        else:
            return self._var_historical(returns)
    
    def calculate_cvar(
        self,
        returns: pd.Series,
        method: Optional[str] = None
    ) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall)
        
        Args:
            returns: Return series
            method: Override default method
        
        Returns:
            CVaR (negative value)
        """
        method = method or self.method
        
        if len(returns) < 20:
            var = returns.quantile(self.alpha)
            return returns[returns <= var].mean()
        
        if method == "historical":
            return self._cvar_historical(returns)
        elif method == "parametric":
            return self._cvar_parametric(returns)
        elif method == "cornish_fisher":
            return self._cvar_cornish_fisher(returns)
        elif method == "evt":
            return self._cvar_evt(returns)
        else:
            return self._cvar_historical(returns)
    
    def _var_historical(self, returns: pd.Series) -> float:
        """Historical VaR (empirical quantile)"""
        return float(returns.quantile(self.alpha))
    
    def _cvar_historical(self, returns: pd.Series) -> float:
        """Historical CVaR (mean of tail)"""
        var = self._var_historical(returns)
        tail = returns[returns <= var]
        return float(tail.mean()) if len(tail) > 0 else var
    
    def _var_parametric(self, returns: pd.Series) -> float:
        """Parametric VaR (Normal distribution)"""
        mu = returns.mean()
        sigma = returns.std()
        z = stats.norm.ppf(self.alpha)
        return float(mu + sigma * z)
    
    def _cvar_parametric(self, returns: pd.Series) -> float:
        """Parametric CVaR (Normal distribution)"""
        mu = returns.mean()
        sigma = returns.std()
        z = stats.norm.ppf(self.alpha)
        # ES for normal: mu - sigma * phi(z) / alpha
        phi_z = stats.norm.pdf(z)
        return float(mu - sigma * phi_z / self.alpha)
    
    def _var_cornish_fisher(self, returns: pd.Series) -> float:
        """
        Cornish-Fisher adjusted VaR
        
        Accounts for skewness and kurtosis.
        """
        mu = returns.mean()
        sigma = returns.std()
        skew = returns.skew()
        kurt = returns.kurtosis()  # Excess kurtosis
        
        z = stats.norm.ppf(self.alpha)
        
        # Cornish-Fisher expansion
        z_cf = (
            z +
            (z**2 - 1) * skew / 6 +
            (z**3 - 3*z) * kurt / 24 -
            (2*z**3 - 5*z) * skew**2 / 36
        )
        
        return float(mu + sigma * z_cf)
    
    def _cvar_cornish_fisher(self, returns: pd.Series) -> float:
        """Cornish-Fisher adjusted CVaR (approximation)"""
        var_cf = self._var_cornish_fisher(returns)
        
        # Use historical approach for tail with CF-adjusted threshold
        tail = returns[returns <= var_cf]
        if len(tail) > 0:
            return float(tail.mean())
        return var_cf
    
    def _var_evt(self, returns: pd.Series) -> float:
        """
        EVT (Extreme Value Theory) VaR
        
        Uses Generalized Pareto Distribution for tail.
        """
        # Use 10% threshold for POT (Peaks Over Threshold)
        threshold = returns.quantile(0.10)
        exceedances = threshold - returns[returns < threshold]
        
        if len(exceedances) < 20:
            return self._var_historical(returns)
        
        try:
            # Fit GPD
            shape, loc, scale = stats.genpareto.fit(exceedances)
            
            n = len(returns)
            n_u = len(exceedances)
            
            # VaR from GPD
            var_evt = threshold - scale / shape * (
                (n * self.alpha / n_u) ** (-shape) - 1
            )
            
            return float(var_evt)
            
        except Exception as e:
            logger.warning(f"EVT fitting failed: {e}")
            return self._var_historical(returns)
    
    def _cvar_evt(self, returns: pd.Series) -> float:
        """EVT CVaR"""
        var_evt = self._var_evt(returns)
        
        # Simple approach: mean of returns below VaR
        tail = returns[returns <= var_evt]
        if len(tail) > 0:
            return float(tail.mean())
        return var_evt
    
    def calculate_all(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate all risk metrics
        
        Returns:
            Dict with VaR, CVaR by different methods
        """
        results = {
            "var_historical": self._var_historical(returns),
            "cvar_historical": self._cvar_historical(returns),
            "var_parametric": self._var_parametric(returns),
            "cvar_parametric": self._cvar_parametric(returns),
        }
        
        if len(returns) >= 50:
            results["var_cornish_fisher"] = self._var_cornish_fisher(returns)
            results["cvar_cornish_fisher"] = self._cvar_cornish_fisher(returns)
        
        if len(returns) >= 100:
            results["var_evt"] = self._var_evt(returns)
            results["cvar_evt"] = self._cvar_evt(returns)
        
        return results


def calculate_rolling_cvar(
    returns: pd.Series,
    window: int = 60,
    confidence: float = 0.95
) -> pd.Series:
    """
    Calculate rolling CVaR
    
    Args:
        returns: Return series
        window: Rolling window size
        confidence: Confidence level
    
    Returns:
        Rolling CVaR series
    """
    calc = CVaRCalculator(confidence=confidence)
    
    rolling_cvar = returns.rolling(window).apply(
        lambda x: calc.calculate_cvar(pd.Series(x)),
        raw=False
    )
    
    return rolling_cvar


def calculate_drawdown_series(returns: pd.Series) -> pd.DataFrame:
    """
    Calculate drawdown-related series
    
    Returns:
        DataFrame with columns: cumulative, peak, drawdown, dd_duration
    """
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.expanding().max()
    drawdown = (cum_returns - peak) / peak
    
    # Drawdown duration
    dd_duration = pd.Series(0, index=returns.index)
    count = 0
    for i in range(len(drawdown)):
        if drawdown.iloc[i] < 0:
            count += 1
        else:
            count = 0
        dd_duration.iloc[i] = count
    
    return pd.DataFrame({
        "cumulative": cum_returns,
        "peak": peak,
        "drawdown": drawdown,
        "dd_duration": dd_duration
    })


def calculate_tail_ratio(returns: pd.Series, percentile: float = 0.05) -> float:
    """
    Calculate tail ratio (right tail / left tail)
    
    Values > 1 indicate positive skew (good)
    Values < 1 indicate negative skew (bad)
    """
    left_tail = abs(returns.quantile(percentile))
    right_tail = returns.quantile(1 - percentile)
    
    return safe_divide(right_tail, left_tail, 1.0)


def calculate_gain_loss_ratio(returns: pd.Series) -> float:
    """
    Calculate gain/loss ratio
    
    Average gain / Average loss
    """
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    
    if len(gains) == 0 or len(losses) == 0:
        return 1.0
    
    avg_gain = gains.mean()
    avg_loss = abs(losses.mean())
    
    return safe_divide(avg_gain, avg_loss, 1.0)


def calculate_omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0
) -> float:
    """
    Calculate Omega ratio
    
    Sum of returns above threshold / Sum of returns below threshold
    """
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = abs(excess[excess < 0].sum())
    
    return safe_divide(gains, losses, 1.0)
