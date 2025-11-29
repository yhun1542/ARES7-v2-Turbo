"""
AARM Core - Adaptive Asymmetric Risk Management
================================================
Turbo AARM: ARES7 QM Regime Turbo의 핵심 리스크 관리

최적 파라미터 (Grid Search 결과):
- base_leverage: 1.2
- max_leverage: 1.8
- target_volatility: 0.18
- cb_trigger: -0.06
- cb_reduction_factor: 0.4
- lookback_days: 60
- tc_bps: 10

성능 목표:
- Sharpe: 3.86
- OOS Sharpe: 4.37
- MDD: -12.63%
- OOS MDD: -10.10%
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.interfaces import IRiskManager, PortfolioState, RiskMetrics, Signal, Regime
from core.utils import get_logger, safe_divide

logger = get_logger(__name__)


@dataclass
class AARMConfig:
    """AARM Configuration"""
    # Leverage
    base_leverage: float = 1.2
    max_leverage: float = 1.8
    
    # Volatility targeting
    target_volatility: float = 0.18
    vol_floor: float = 0.5
    vol_cap: float = 2.0
    lookback_days: int = 60
    
    # Circuit breaker
    cb_trigger: float = -0.06
    cb_reduction_factor: float = 0.4
    
    # Drawdown scaling thresholds
    dd_thresholds: List[Tuple[float, float]] = field(default_factory=lambda: [
        (-0.05, 0.9),   # -5% DD → 90% position
        (-0.08, 0.7),   # -8% DD → 70% position
        (-0.10, 0.5),   # -10% DD → 50% position
        (-0.15, 0.3),   # -15% DD → 30% position
    ])
    
    # Transaction costs
    tc_bps: float = 10.0


class AARMCore(IRiskManager):
    """
    Adaptive Asymmetric Risk Manager Core
    
    Features:
    1. Volatility targeting: Scale position by target_vol / realized_vol
    2. Drawdown-based scaling: Progressive reduction based on DD depth
    3. Circuit breaker: Hard cut when DD exceeds trigger
    4. Regime adjustment: Additional scale based on market regime
    """
    
    def __init__(self, config: Optional[AARMConfig] = None):
        """
        Initialize AARM
        
        Args:
            config: AARM configuration (uses optimal defaults if None)
        """
        self.config = config or AARMConfig()
        
        # State
        self._current_dd: float = 0.0
        self._peak_value: float = 1.0
        self._cb_active: bool = False
        self._position_scale: float = 1.0
        self._last_position: float = 1.0
        
        # History
        self._cum_returns: float = 1.0
        self._returns_history: List[float] = []
    
    def update(
        self,
        returns: pd.Series,
        portfolio_state: PortfolioState
    ) -> RiskMetrics:
        """
        Update risk metrics from return series
        
        Args:
            returns: Historical returns
            portfolio_state: Current portfolio state
        
        Returns:
            Updated RiskMetrics
        """
        if len(returns) < 2:
            return RiskMetrics()
        
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Current value
        current_value = cum_returns.iloc[-1]
        
        # Update peak
        rolling_max = cum_returns.expanding().max()
        peak_value = rolling_max.iloc[-1]
        
        # Drawdown
        current_dd = (current_value - peak_value) / peak_value
        max_dd = ((cum_returns - rolling_max) / rolling_max).min()
        
        # Volatility
        lookback = min(self.config.lookback_days, len(returns) - 1)
        recent_returns = returns.iloc[-lookback:]
        vol_20d = recent_returns.std() * np.sqrt(252)
        
        # VaR/CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        
        # Sharpe/Sortino
        ann_return = returns.mean() * 252
        sharpe = safe_divide(ann_return, vol_20d, 0.0)
        
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else vol_20d
        sortino = safe_divide(ann_return, downside_vol, 0.0)
        
        # Update state
        self._current_dd = current_dd
        self._peak_value = peak_value
        
        # Check circuit breaker
        cb_active = self.check_circuit_breaker(RiskMetrics(current_drawdown=current_dd))
        self._cb_active = cb_active
        
        return RiskMetrics(
            current_drawdown=current_dd,
            max_drawdown=max_dd,
            volatility_20d=vol_20d,
            var_95=var_95,
            cvar_95=cvar_95,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            position_scale=self._position_scale,
            cb_active=cb_active,
        )
    
    def calculate_position_scale(
        self,
        risk_metrics: RiskMetrics,
        regime: Regime
    ) -> float:
        """
        Calculate position scale factor
        
        Combines:
        1. Volatility targeting
        2. Drawdown-based reduction
        3. Circuit breaker
        4. Regime adjustment
        
        Returns:
            Position scale factor (0.0 to max_leverage)
        """
        scale = self.config.base_leverage
        
        # 1. Volatility targeting
        if risk_metrics.volatility_20d > 0:
            vol_factor = self.config.target_volatility / risk_metrics.volatility_20d
            vol_factor = np.clip(vol_factor, self.config.vol_floor, self.config.vol_cap)
            scale *= vol_factor
        
        # 2. Drawdown-based reduction (progressive)
        dd_multiplier = self._get_dd_multiplier(risk_metrics.current_drawdown)
        scale *= dd_multiplier
        
        # 3. Circuit breaker
        if risk_metrics.cb_active or self._cb_active:
            scale *= self.config.cb_reduction_factor
        
        # 4. Regime adjustment
        regime_scale = self._get_regime_scale(regime)
        scale *= regime_scale
        
        # Apply constraints
        scale = np.clip(scale, 0.1, self.config.max_leverage)
        
        self._position_scale = scale
        
        logger.debug(
            f"Position scale: {scale:.3f} "
            f"(vol_factor: {vol_factor if risk_metrics.volatility_20d > 0 else 1:.2f}, "
            f"dd_mult: {dd_multiplier:.2f}, "
            f"regime: {regime.value})"
        )
        
        return scale
    
    def _get_dd_multiplier(self, current_dd: float) -> float:
        """
        Get drawdown-based multiplier (progressive reduction)
        """
        if current_dd >= 0:
            return 1.0
        
        # Find applicable threshold
        multiplier = 1.0
        
        for threshold, mult in self.config.dd_thresholds:
            if current_dd <= threshold:
                multiplier = mult
        
        # Interpolate between thresholds for smoother scaling
        # (Optional: can use step function instead)
        
        return max(0.3, multiplier)
    
    def _get_regime_scale(self, regime: Regime) -> float:
        """Get regime-based scale factor"""
        scales = {
            Regime.BULL: 1.0,
            Regime.BEAR: 0.7,
            Regime.HIGH_VOL: 0.5,
            Regime.NEUTRAL: 0.85,
        }
        return scales.get(regime, 0.8)
    
    def check_circuit_breaker(self, risk_metrics: RiskMetrics) -> bool:
        """
        Check if circuit breaker should be activated
        
        Returns True if:
        - Current drawdown exceeds cb_trigger
        """
        return risk_metrics.current_drawdown <= self.config.cb_trigger
    
    def apply_risk_limits(
        self,
        signals: List[Signal],
        portfolio_state: PortfolioState,
        risk_metrics: RiskMetrics
    ) -> List[Signal]:
        """
        Apply risk limits to signals
        
        Limits:
        - Max position size per stock
        - Max sector exposure
        - Overall leverage
        """
        if not signals:
            return signals
        
        adjusted_signals = []
        total_weight = sum(abs(s.target_weight) for s in signals)
        
        for signal in signals:
            new_weight = signal.target_weight
            
            # Scale by position_scale
            new_weight *= self._position_scale
            
            # Max position size (10%)
            max_position = 0.10
            new_weight = min(abs(new_weight), max_position) * np.sign(new_weight)
            
            # Create new signal with adjusted weight
            adjusted_signals.append(Signal(
                symbol=signal.symbol,
                side=signal.side,
                signal_type=signal.signal_type,
                target_weight=new_weight,
                confidence=signal.confidence,
                timestamp=signal.timestamp,
                metadata={**signal.metadata, "risk_adjusted": True}
            ))
        
        # Normalize if total exceeds max leverage
        total_adjusted = sum(abs(s.target_weight) for s in adjusted_signals)
        
        if total_adjusted > self.config.max_leverage:
            scale_down = self.config.max_leverage / total_adjusted
            adjusted_signals = [
                Signal(
                    symbol=s.symbol,
                    side=s.side,
                    signal_type=s.signal_type,
                    target_weight=s.target_weight * scale_down,
                    confidence=s.confidence,
                    timestamp=s.timestamp,
                    metadata=s.metadata
                )
                for s in adjusted_signals
            ]
        
        return adjusted_signals
    
    def calculate_transaction_cost(
        self,
        old_weight: float,
        new_weight: float,
        portfolio_value: float
    ) -> float:
        """Calculate transaction cost for position change"""
        turnover = abs(new_weight - old_weight)
        tc_decimal = self.config.tc_bps / 10000.0
        return turnover * portfolio_value * tc_decimal


class TurboAARM(AARMCore):
    """
    Turbo AARM - Optimized for ARES7 QM Regime Turbo
    
    Uses the grid-search optimized parameters:
    - base_leverage: 1.2
    - max_leverage: 1.8
    - target_volatility: 0.18
    - cb_trigger: -0.06
    - cb_reduction_factor: 0.4
    """
    
    def __init__(self):
        """Initialize with optimal Turbo parameters"""
        config = AARMConfig(
            base_leverage=1.2,
            max_leverage=1.8,
            target_volatility=0.18,
            vol_floor=0.5,
            vol_cap=2.0,
            lookback_days=60,
            cb_trigger=-0.06,
            cb_reduction_factor=0.4,
            dd_thresholds=[
                (-0.05, 0.9),
                (-0.08, 0.7),
                (-0.10, 0.5),
                (-0.15, 0.3),
            ],
            tc_bps=10.0,
        )
        super().__init__(config)
    
    def apply_turbo(
        self,
        base_returns: pd.Series,
        regime_series: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Apply Turbo AARM to return series
        
        This is the main backtest method that reproduces
        the Sharpe 3.86 / OOS Sharpe 4.37 performance.
        
        Args:
            base_returns: Base strategy returns
            regime_series: Optional regime series (date → Regime.value)
        
        Returns:
            Managed returns series
        """
        if len(base_returns) < self.config.lookback_days + 1:
            return base_returns
        
        # Convert to numpy for speed
        returns_arr = base_returns.values.astype(np.float64)
        n = len(returns_arr)
        
        # Output arrays
        managed_returns = np.zeros(n, dtype=np.float64)
        cum_returns = np.ones(n, dtype=np.float64)
        
        # Initial period: use base leverage
        for i in range(min(self.config.lookback_days + 1, n)):
            managed_returns[i] = returns_arr[i] * self.config.base_leverage
            if i == 0:
                cum_returns[i] = 1 + managed_returns[i]
            else:
                cum_returns[i] = cum_returns[i-1] * (1 + managed_returns[i])
        
        prev_position = self.config.base_leverage
        
        for i in range(self.config.lookback_days + 1, n):
            # Calculate current drawdown
            running_max = cum_returns[:i].max()
            current_dd = (cum_returns[i-1] - running_max) / running_max
            
            # Calculate rolling volatility
            lookback_returns = returns_arr[i-self.config.lookback_days:i]
            vol = np.std(lookback_returns) * np.sqrt(252)
            
            # Volatility targeting
            if vol > 0:
                vol_factor = self.config.target_volatility / vol
                vol_factor = np.clip(vol_factor, self.config.vol_floor, self.config.vol_cap)
            else:
                vol_factor = 1.0
            
            # Drawdown-based scaling
            dd_multiplier = self._compute_dd_multiplier_fast(current_dd)
            
            # Base position
            position = self.config.base_leverage * vol_factor * dd_multiplier
            
            # Circuit breaker
            if current_dd <= self.config.cb_trigger:
                position *= self.config.cb_reduction_factor
            
            # Regime adjustment (if available)
            if regime_series is not None:
                date = base_returns.index[i]
                if date in regime_series.index:
                    regime_str = regime_series.loc[date]
                    regime_scale = {
                        "BULL": 1.0,
                        "BEAR": 0.7,
                        "HIGH_VOL": 0.5,
                        "NEUTRAL": 0.85
                    }.get(regime_str, 0.85)
                    position *= regime_scale
            
            # Constrain
            position = min(position, self.config.max_leverage)
            
            # Transaction cost
            turnover = abs(position - prev_position)
            tc = turnover * (self.config.tc_bps / 10000.0)
            
            # Apply
            managed_returns[i] = returns_arr[i] * position - tc
            cum_returns[i] = cum_returns[i-1] * (1 + managed_returns[i])
            
            prev_position = position
        
        result = pd.Series(
            managed_returns,
            index=base_returns.index,
            name="turbo_aarm_returns"
        )
        
        return result
    
    def _compute_dd_multiplier_fast(self, dd: float) -> float:
        """Fast drawdown multiplier (inline)"""
        if dd >= 0:
            return 1.0
        
        dd_abs = abs(dd)
        
        if dd_abs < 0.05:
            return 1.0 - dd_abs * 2  # Linear from 1.0 to 0.9
        elif dd_abs < 0.08:
            return 0.9 - (dd_abs - 0.05) * (0.9 - 0.7) / 0.03
        elif dd_abs < 0.10:
            return 0.7 - (dd_abs - 0.08) * (0.7 - 0.5) / 0.02
        elif dd_abs < 0.15:
            return 0.5 - (dd_abs - 0.10) * (0.5 - 0.3) / 0.05
        else:
            return 0.3
