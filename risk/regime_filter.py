"""
Regime Filter
==============
시장 레짐 판별 - BULL/BEAR/HIGH_VOL/NEUTRAL

레짐에 따라:
- BULL: QM 100% 가중
- BEAR: QM 60%, Defensive 40%
- HIGH_VOL: QM 50%, Defensive 50%
- NEUTRAL: QM 30%, Defensive 70%

판별 기준:
- SPX > MA200: 상승 추세
- SPX 6M/12M Return > 0: 모멘텀 양수
- VIX < 25: 저변동성
- VIX >= 30: 고변동성
- FRED macro signals: 추가 확인
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.interfaces import Regime
from core.utils import get_logger

logger = get_logger(__name__)


@dataclass
class RegimeConditions:
    """레짐 판별 조건"""
    spx_above_ma200: bool = False
    ret_6m_positive: bool = False
    ret_12m_positive: bool = False
    vix_below_25: bool = False
    vix_above_30: bool = False
    term_spread_positive: bool = True
    credit_stress: bool = False
    event_risk_flag: bool = False


@dataclass
class RegimeResult:
    """레짐 판별 결과"""
    regime: Regime
    conditions: RegimeConditions
    confidence: float
    timestamp: datetime
    details: Dict[str, Any]


class RegimeFilter:
    """
    Market Regime Filter
    
    ARES7 QM Regime Turbo의 핵심 컴포넌트.
    SPX, VIX, FRED macro 데이터를 사용해 시장 레짐을 판별.
    """
    
    def __init__(
        self,
        ma_period: int = 200,
        vix_low_threshold: float = 25.0,
        vix_high_threshold: float = 30.0,
        momentum_lookback_6m: int = 126,
        momentum_lookback_12m: int = 252,
    ):
        """
        Initialize Regime Filter
        
        Args:
            ma_period: Moving average period for trend
            vix_low_threshold: VIX below this = low vol
            vix_high_threshold: VIX above this = high vol
            momentum_lookback_6m: 6-month lookback
            momentum_lookback_12m: 12-month lookback
        """
        self.ma_period = ma_period
        self.vix_low_threshold = vix_low_threshold
        self.vix_high_threshold = vix_high_threshold
        self.momentum_lookback_6m = momentum_lookback_6m
        self.momentum_lookback_12m = momentum_lookback_12m
        
        # Historical regimes
        self._regime_history: List[RegimeResult] = []
        
        # Current state
        self._current_regime: Regime = Regime.NEUTRAL
        self._last_update: Optional[datetime] = None
    
    @property
    def current_regime(self) -> Regime:
        """Get current regime"""
        return self._current_regime
    
    def detect_regime(
        self,
        spx_prices: pd.Series,
        vix_level: float,
        as_of_date: datetime,
        macro_signals: Optional[Dict[str, Any]] = None,
        event_risk: bool = False,
    ) -> RegimeResult:
        """
        Detect market regime
        
        Args:
            spx_prices: SPX price series (date index)
            vix_level: Current VIX level
            as_of_date: As-of date
            macro_signals: Optional FRED macro signals
            event_risk: Event risk flag from Tavily/news
        
        Returns:
            RegimeResult with regime and conditions
        """
        conditions = RegimeConditions()
        details = {}
        
        # 1. SPX vs MA200
        if len(spx_prices) >= self.ma_period:
            ma200 = spx_prices.rolling(self.ma_period).mean()
            current_price = spx_prices.iloc[-1]
            ma200_value = ma200.iloc[-1]
            
            conditions.spx_above_ma200 = current_price > ma200_value
            details["spx_current"] = float(current_price)
            details["spx_ma200"] = float(ma200_value)
            details["spx_vs_ma200_pct"] = float((current_price / ma200_value - 1) * 100)
        
        # 2. 6-month momentum
        if len(spx_prices) >= self.momentum_lookback_6m:
            ret_6m = spx_prices.iloc[-1] / spx_prices.iloc[-self.momentum_lookback_6m] - 1
            conditions.ret_6m_positive = ret_6m > 0
            details["ret_6m"] = float(ret_6m)
        
        # 3. 12-month momentum
        if len(spx_prices) >= self.momentum_lookback_12m:
            ret_12m = spx_prices.iloc[-1] / spx_prices.iloc[-self.momentum_lookback_12m] - 1
            conditions.ret_12m_positive = ret_12m > 0
            details["ret_12m"] = float(ret_12m)
        
        # 4. VIX levels
        conditions.vix_below_25 = vix_level < self.vix_low_threshold
        conditions.vix_above_30 = vix_level >= self.vix_high_threshold
        details["vix_level"] = float(vix_level)
        
        # 5. Macro signals (from FRED)
        if macro_signals:
            conditions.term_spread_positive = macro_signals.get("term_spread_positive", True)
            conditions.credit_stress = macro_signals.get("credit_stress", False)
            details["macro_signals"] = macro_signals
        
        # 6. Event risk (from Tavily/News)
        conditions.event_risk_flag = event_risk
        details["event_risk"] = event_risk
        
        # Determine regime
        regime = self._classify_regime(conditions)
        confidence = self._calculate_confidence(conditions, regime)
        
        result = RegimeResult(
            regime=regime,
            conditions=conditions,
            confidence=confidence,
            timestamp=as_of_date,
            details=details
        )
        
        # Update state
        self._current_regime = regime
        self._last_update = as_of_date
        self._regime_history.append(result)
        
        # Keep history limited
        if len(self._regime_history) > 1000:
            self._regime_history = self._regime_history[-500:]
        
        logger.info(
            f"Regime detected: {regime.value} (confidence: {confidence:.2f})",
            extra={"vix": vix_level, "spx_vs_ma": details.get("spx_vs_ma200_pct")}
        )
        
        return result
    
    def _classify_regime(self, conditions: RegimeConditions) -> Regime:
        """
        Classify regime based on conditions
        
        Priority:
        1. HIGH_VOL if VIX >= 30 OR event_risk
        2. BULL if all bull conditions met
        3. BEAR if bear conditions met
        4. NEUTRAL otherwise
        """
        # Check HIGH_VOL first (highest priority)
        if conditions.vix_above_30 or conditions.event_risk_flag:
            return Regime.HIGH_VOL
        
        # Check credit stress
        if conditions.credit_stress:
            return Regime.HIGH_VOL
        
        # Check BULL conditions (all must be true)
        bull_conditions = [
            conditions.spx_above_ma200,
            conditions.ret_6m_positive,
            conditions.ret_12m_positive,
            conditions.vix_below_25,
        ]
        
        if all(bull_conditions):
            return Regime.BULL
        
        # Check BEAR conditions
        bear_conditions = [
            not conditions.spx_above_ma200,
            not conditions.ret_6m_positive,
        ]
        
        if all(bear_conditions):
            return Regime.BEAR
        
        # Check partial BEAR (SPX below MA200 and negative 12m)
        if not conditions.spx_above_ma200 and not conditions.ret_12m_positive:
            return Regime.BEAR
        
        # Default to NEUTRAL
        return Regime.NEUTRAL
    
    def _calculate_confidence(
        self,
        conditions: RegimeConditions,
        regime: Regime
    ) -> float:
        """
        Calculate confidence level (0-1)
        
        Higher confidence when more conditions align.
        """
        if regime == Regime.BULL:
            # Count how many bull conditions are met
            score = sum([
                conditions.spx_above_ma200,
                conditions.ret_6m_positive,
                conditions.ret_12m_positive,
                conditions.vix_below_25,
                conditions.term_spread_positive,
                not conditions.credit_stress,
            ])
            return score / 6.0
        
        elif regime == Regime.BEAR:
            score = sum([
                not conditions.spx_above_ma200,
                not conditions.ret_6m_positive,
                not conditions.ret_12m_positive,
                not conditions.term_spread_positive,
                conditions.credit_stress,
            ])
            return score / 5.0
        
        elif regime == Regime.HIGH_VOL:
            score = sum([
                conditions.vix_above_30,
                conditions.credit_stress,
                conditions.event_risk_flag,
            ])
            return min(1.0, score / 2.0)
        
        else:  # NEUTRAL
            return 0.5
    
    def get_regime_series(
        self,
        spx_prices: pd.Series,
        vix_series: pd.Series,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.Series:
        """
        Generate regime series for backtesting
        
        Args:
            spx_prices: SPX price series
            vix_series: VIX series
            start_date: Start date (optional)
            end_date: End date (optional)
        
        Returns:
            Series with date index, regime values
        """
        # Align series
        df = pd.DataFrame({
            "spx": spx_prices,
            "vix": vix_series
        }).dropna()
        
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        regimes = []
        
        for date in df.index:
            # Get data up to this date
            spx_to_date = spx_prices.loc[:date]
            vix_level = df.loc[date, "vix"]
            
            if len(spx_to_date) < self.ma_period:
                regimes.append(Regime.NEUTRAL.value)
                continue
            
            result = self.detect_regime(
                spx_prices=spx_to_date,
                vix_level=vix_level,
                as_of_date=date
            )
            
            regimes.append(result.regime.value)
        
        return pd.Series(regimes, index=df.index, name="regime")
    
    def get_regime_weights(self, regime: Regime) -> Dict[str, float]:
        """
        Get engine weights for regime
        
        Returns weights for QM and Defensive engines.
        """
        weights = {
            Regime.BULL: {"qm": 1.0, "defensive": 0.0},
            Regime.BEAR: {"qm": 0.6, "defensive": 0.4},
            Regime.HIGH_VOL: {"qm": 0.5, "defensive": 0.5},
            Regime.NEUTRAL: {"qm": 0.3, "defensive": 0.7},
        }
        
        return weights.get(regime, {"qm": 0.5, "defensive": 0.5})
    
    def get_position_scale(self, regime: Regime) -> float:
        """
        Get position scale factor for regime
        
        Used by AARM for additional regime-based scaling.
        """
        scales = {
            Regime.BULL: 1.0,
            Regime.BEAR: 0.7,
            Regime.HIGH_VOL: 0.5,
            Regime.NEUTRAL: 0.8,
        }
        
        return scales.get(regime, 0.8)
    
    def get_regime_stats(self) -> Dict[str, Any]:
        """Get regime statistics from history"""
        if not self._regime_history:
            return {}
        
        regimes = [r.regime.value for r in self._regime_history]
        
        from collections import Counter
        counts = Counter(regimes)
        total = len(regimes)
        
        return {
            "total_observations": total,
            "current_regime": self._current_regime.value,
            "regime_distribution": {k: v / total for k, v in counts.items()},
            "last_update": self._last_update.isoformat() if self._last_update else None,
        }
