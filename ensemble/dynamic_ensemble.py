"""
Dynamic Ensemble
=================
레짐별 엔진 가중합

레짐에 따른 가중치:
- BULL: QM 100%
- BEAR: QM 60%, Defensive 40%
- HIGH_VOL: QM 50%, Defensive 50%
- NEUTRAL: QM 30%, Defensive 70%

확장 가능: LowVol 엔진 추가 지원
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.interfaces import Regime
from core.utils import get_logger

logger = get_logger(__name__)


@dataclass
class RegimeWeights:
    """레짐별 엔진 가중치"""
    bull: Dict[str, float] = field(default_factory=lambda: {"qm": 1.0, "defensive": 0.0})
    bear: Dict[str, float] = field(default_factory=lambda: {"qm": 0.6, "defensive": 0.4})
    high_vol: Dict[str, float] = field(default_factory=lambda: {"qm": 0.5, "defensive": 0.5})
    neutral: Dict[str, float] = field(default_factory=lambda: {"qm": 0.3, "defensive": 0.7})


@dataclass
class EngineReturns:
    """엔진별 수익률"""
    qm: pd.Series
    defensive: pd.Series
    lowvol: Optional[pd.Series] = None


class DynamicEnsemble:
    """
    Dynamic Ensemble - Regime-Based Engine Blending
    
    ARES7 QM Regime Turbo의 앙상블 레이어.
    시장 레짐에 따라 QM/Defensive 엔진의 가중치를 동적 조정.
    """
    
    def __init__(
        self,
        weights: Optional[RegimeWeights] = None,
        include_lowvol: bool = False
    ):
        """
        Initialize Dynamic Ensemble
        
        Args:
            weights: Custom regime weights (uses defaults if None)
            include_lowvol: Whether to include LowVol engine
        """
        self.weights = weights or RegimeWeights()
        self.include_lowvol = include_lowvol
    
    def get_weights(self, regime: Regime) -> Dict[str, float]:
        """Get engine weights for given regime"""
        if regime == Regime.BULL:
            return self.weights.bull.copy()
        elif regime == Regime.BEAR:
            return self.weights.bear.copy()
        elif regime == Regime.HIGH_VOL:
            return self.weights.high_vol.copy()
        else:
            return self.weights.neutral.copy()
    
    def blend_returns(
        self,
        engine_returns: EngineReturns,
        regime_series: pd.Series
    ) -> pd.Series:
        """
        Blend engine returns based on regime
        
        Args:
            engine_returns: EngineReturns with qm, defensive, (optional) lowvol
            regime_series: Series with date index, Regime.value strings
        
        Returns:
            Blended return series
        """
        # Align all series
        df = pd.DataFrame({
            "qm": engine_returns.qm,
            "defensive": engine_returns.defensive,
        })
        
        if self.include_lowvol and engine_returns.lowvol is not None:
            df["lowvol"] = engine_returns.lowvol
        
        df = df.dropna()
        
        # Reindex regime to match returns
        regime_aligned = regime_series.reindex(df.index, method="ffill")
        
        # Blend
        blended = []
        
        for date, row in df.iterrows():
            regime_str = regime_aligned.loc[date] if date in regime_aligned.index else "NEUTRAL"
            
            try:
                regime = Regime(regime_str)
            except ValueError:
                regime = Regime.NEUTRAL
            
            weights = self.get_weights(regime)
            
            # Calculate weighted return
            ret = 0.0
            total_weight = 0.0
            
            for engine, weight in weights.items():
                if engine in row.index and not np.isnan(row[engine]):
                    ret += weight * row[engine]
                    total_weight += weight
            
            if total_weight > 0:
                ret /= total_weight
            
            blended.append((date, ret))
        
        result = pd.Series(
            data=[r for _, r in blended],
            index=[d for d, _ in blended],
            name="ensemble_returns"
        )
        
        logger.info(f"Blended {len(result)} days of returns")
        return result
    
    def blend_signals(
        self,
        qm_signals: Dict[str, float],
        defensive_signals: Dict[str, float],
        regime: Regime,
        lowvol_signals: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Blend engine signals (target weights) based on regime
        
        Args:
            qm_signals: QM engine target weights {symbol: weight}
            defensive_signals: Defensive engine target weights
            regime: Current regime
            lowvol_signals: Optional LowVol signals
        
        Returns:
            Blended target weights {symbol: weight}
        """
        weights = self.get_weights(regime)
        
        w_qm = weights.get("qm", 0.5)
        w_def = weights.get("defensive", 0.5)
        w_lv = weights.get("lowvol", 0.0) if lowvol_signals else 0.0
        
        # Normalize
        total = w_qm + w_def + w_lv
        if total > 0:
            w_qm /= total
            w_def /= total
            w_lv /= total
        
        # Collect all symbols
        all_symbols = set(qm_signals.keys()) | set(defensive_signals.keys())
        if lowvol_signals:
            all_symbols |= set(lowvol_signals.keys())
        
        # Blend
        blended = {}
        
        for symbol in all_symbols:
            target = 0.0
            target += w_qm * qm_signals.get(symbol, 0.0)
            target += w_def * defensive_signals.get(symbol, 0.0)
            if lowvol_signals:
                target += w_lv * lowvol_signals.get(symbol, 0.0)
            
            if abs(target) > 1e-6:
                blended[symbol] = target
        
        return blended
    
    def generate_backtest_returns(
        self,
        prices: pd.DataFrame,
        qm_weights: pd.DataFrame,
        defensive_weights: pd.DataFrame,
        regime_series: pd.Series,
        transaction_cost_bps: float = 10.0
    ) -> pd.Series:
        """
        Generate ensemble returns for backtesting
        
        Args:
            prices: Price DataFrame (date index, symbol columns)
            qm_weights: QM weight DataFrame (date index, symbol columns)
            defensive_weights: Defensive weight DataFrame
            regime_series: Regime series
            transaction_cost_bps: Transaction cost in bps
        
        Returns:
            Ensemble return series
        """
        # Calculate returns from prices
        returns = prices.pct_change()
        
        # Align weights to regime
        regime_aligned = regime_series.reindex(returns.index, method="ffill")
        
        ensemble_returns = []
        prev_weights = {}
        
        for date in returns.index:
            if date not in qm_weights.index or date not in defensive_weights.index:
                ensemble_returns.append(0.0)
                continue
            
            # Get regime
            regime_str = regime_aligned.loc[date] if date in regime_aligned.index else "NEUTRAL"
            try:
                regime = Regime(regime_str)
            except ValueError:
                regime = Regime.NEUTRAL
            
            # Blend weights
            qm_w = qm_weights.loc[date].to_dict()
            def_w = defensive_weights.loc[date].to_dict()
            blended_weights = self.blend_signals(qm_w, def_w, regime)
            
            # Calculate return
            daily_return = 0.0
            for symbol, weight in blended_weights.items():
                if symbol in returns.columns:
                    ret = returns.loc[date, symbol]
                    if not np.isnan(ret):
                        daily_return += weight * ret
            
            # Transaction cost
            turnover = sum(
                abs(blended_weights.get(s, 0) - prev_weights.get(s, 0))
                for s in set(blended_weights.keys()) | set(prev_weights.keys())
            )
            tc = turnover * transaction_cost_bps / 10000.0
            
            ensemble_returns.append(daily_return - tc)
            prev_weights = blended_weights.copy()
        
        return pd.Series(ensemble_returns, index=returns.index, name="ensemble_returns")


class ThreeEngineEnsemble(DynamicEnsemble):
    """
    Three-Engine Ensemble: QM + Defensive + LowVol
    
    Extended version for future use.
    """
    
    def __init__(self):
        """Initialize with three-engine weights"""
        weights = RegimeWeights(
            bull={"qm": 0.8, "defensive": 0.1, "lowvol": 0.1},
            bear={"qm": 0.4, "defensive": 0.4, "lowvol": 0.2},
            high_vol={"qm": 0.3, "defensive": 0.3, "lowvol": 0.4},
            neutral={"qm": 0.3, "defensive": 0.4, "lowvol": 0.3},
        )
        super().__init__(weights=weights, include_lowvol=True)
