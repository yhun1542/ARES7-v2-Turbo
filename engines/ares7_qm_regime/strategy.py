"""
ARES7 QM Regime Strategy
=========================
Quality-Momentum Overlay with Regime Filter

ARES7 QM Regime Turbo의 핵심 알파 엔진.
IStrategyEngine 인터페이스 구현.

주요 기능:
1. QM Overlay: Quality + Momentum 스코어 기반 종목 선정
2. Regime Filter: BULL에서만 overlay 적용
3. Turbo AARM 연동: 리스크 관리
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from core.interfaces import (
    IStrategyEngine,
    Signal,
    SignalType,
    Side,
    PortfolioState,
    RiskMetrics,
    Regime,
)
from core.utils import get_logger, load_config

logger = get_logger(__name__)


class ARES7QMRegimeStrategy(IStrategyEngine):
    """
    ARES7 QM Regime Strategy Engine
    
    Quality-Momentum strategy with regime-based overlay.
    
    Configuration:
    - universe: SP100
    - pit_delay_days: 90
    - qm_overlay: top_frac, bottom_frac, overlay_strength
    - quality_weight: 0.6, momentum_weight: 0.4
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize strategy
        
        Args:
            config_path: Path to YAML config (uses default if None)
        """
        self._name = "ARES7_QM_REGIME"
        self._config: Dict[str, Any] = {}
        self._initialized = False
        
        # Data
        self._prices: Optional[pd.DataFrame] = None
        self._fundamentals: Optional[pd.DataFrame] = None
        self._qm_scores: Optional[pd.DataFrame] = None
        self._universe: List[str] = []
        
        # State
        self._current_regime: Regime = Regime.NEUTRAL
        self._target_weights: Dict[str, float] = {}
        
        if config_path:
            self.initialize(load_config(config_path))
    
    @property
    def name(self) -> str:
        return self._name
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize strategy with configuration"""
        self._config = config
        
        # Load QM overlay config
        qm_config = config.get("qm_overlay", {})
        self._top_frac = qm_config.get("top_frac", 0.20)
        self._bottom_frac = qm_config.get("bottom_frac", 0.20)
        self._overlay_strength = qm_config.get("overlay_strength", 0.04)
        self._quality_weight = qm_config.get("quality_weight", 0.6)
        self._momentum_weight = qm_config.get("momentum_weight", 0.4)
        
        # PIT settings
        data_config = config.get("data", {})
        self._pit_delay_days = data_config.get("pit_delay_days", 90)
        
        # Universe
        universe_config = config.get("universe", {})
        self._universe_base = universe_config.get("base", "SP100")
        
        self._initialized = True
        logger.info(f"ARES7 QM Regime strategy initialized with config: {config.get('strategy', {})}")
    
    def update_data(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Update strategy data
        
        Args:
            prices: Price DataFrame with date index, symbol columns
            fundamentals: Fundamental DataFrame with symbol index
        """
        self._prices = prices
        self._fundamentals = fundamentals
        
        if prices is not None:
            self._universe = list(prices.columns)
        
        logger.debug(f"Data updated: {len(self._universe)} symbols")
    
    def set_regime(self, regime: Regime) -> None:
        """Set current market regime"""
        self._current_regime = regime
    
    def generate_signals(
        self,
        as_of: datetime,
        portfolio_state: PortfolioState,
        risk_metrics: RiskMetrics
    ) -> List[Signal]:
        """
        Generate trading signals
        
        Args:
            as_of: Signal generation timestamp
            portfolio_state: Current portfolio state
            risk_metrics: Current risk metrics
        
        Returns:
            List of Signal objects
        """
        if not self._initialized:
            logger.warning("Strategy not initialized")
            return []
        
        if self._prices is None or self._prices.empty:
            logger.warning("No price data available")
            return []
        
        # 1. Calculate QM scores
        qm_scores = self._calculate_qm_scores(as_of)
        
        if qm_scores.empty:
            logger.warning("No QM scores calculated")
            return []
        
        # 2. Apply regime filter
        regime = risk_metrics.regime if hasattr(risk_metrics, 'regime') else self._current_regime
        
        # 3. Generate target weights
        target_weights = self._generate_target_weights(qm_scores, regime)
        
        # 4. Convert to signals
        signals = self._weights_to_signals(target_weights, as_of)
        
        self._target_weights = target_weights
        
        logger.info(f"Generated {len(signals)} signals (regime: {regime.value})")
        return signals
    
    def _calculate_qm_scores(self, as_of: datetime) -> pd.DataFrame:
        """
        Calculate Quality-Momentum scores
        
        Quality factors (from fundamentals):
        - ROE
        - ROIC
        - Gross Margin
        - Current Ratio
        - D/E (inverted)
        
        Momentum factors (from prices):
        - 6-month return
        - 12-month return (skip 1 month)
        """
        scores = pd.DataFrame(index=self._universe)
        
        # Quality scores
        if self._fundamentals is not None and not self._fundamentals.empty:
            quality_score = self._calculate_quality_score()
            scores["quality"] = quality_score
        else:
            scores["quality"] = 0.5
        
        # Momentum scores
        momentum_score = self._calculate_momentum_score(as_of)
        scores["momentum"] = momentum_score
        
        # Combined QM score
        scores["qm_score"] = (
            self._quality_weight * scores["quality"].fillna(0.5) +
            self._momentum_weight * scores["momentum"].fillna(0.5)
        )
        
        # Rank
        scores["qm_rank"] = scores["qm_score"].rank(pct=True)
        
        return scores.dropna(subset=["qm_score"])
    
    def _calculate_quality_score(self) -> pd.Series:
        """Calculate quality score from fundamentals"""
        if self._fundamentals is None:
            return pd.Series()
        
        # Standardize each metric
        standardized = pd.DataFrame(index=self._fundamentals.index)
        
        metrics = ["roe", "roic", "grossmargin", "currentratio", "de_ratio"]
        
        for metric in metrics:
            if metric not in self._fundamentals.columns:
                continue
            
            series = self._fundamentals[metric].dropna()
            
            # Winsorize
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
            
            # Invert D/E
            if metric == "de_ratio":
                z_score = -z_score
            
            standardized[metric] = z_score
        
        # Average
        quality_score = standardized.mean(axis=1)
        
        # Convert to rank percentile
        return quality_score.rank(pct=True)
    
    def _calculate_momentum_score(self, as_of: datetime) -> pd.Series:
        """Calculate momentum score from prices"""
        if self._prices is None or self._prices.empty:
            return pd.Series()
        
        # Get prices up to as_of
        prices_to_date = self._prices.loc[:as_of]
        
        if len(prices_to_date) < 252 + 21:  # Need enough history
            return pd.Series(0.5, index=self._universe)
        
        scores = pd.DataFrame(index=self._universe)
        
        # 6-month return (skip recent month)
        lookback_6m = 126
        skip = 21
        
        try:
            ret_6m = (
                prices_to_date.iloc[-(skip + 1)] /
                prices_to_date.iloc[-(lookback_6m + skip)] - 1
            )
            scores["ret_6m"] = ret_6m.rank(pct=True)
        except Exception:
            scores["ret_6m"] = 0.5
        
        # 12-month return (skip recent month)
        lookback_12m = 252
        
        try:
            ret_12m = (
                prices_to_date.iloc[-(skip + 1)] /
                prices_to_date.iloc[-(lookback_12m + skip)] - 1
            )
            scores["ret_12m"] = ret_12m.rank(pct=True)
        except Exception:
            scores["ret_12m"] = 0.5
        
        # Average
        momentum_score = (scores["ret_6m"] + scores["ret_12m"]) / 2
        
        return momentum_score
    
    def _generate_target_weights(
        self,
        qm_scores: pd.DataFrame,
        regime: Regime
    ) -> Dict[str, float]:
        """
        Generate target portfolio weights based on QM scores and regime
        
        In BULL regime: Apply QM overlay
        In other regimes: Equal weight (defensive)
        """
        n_stocks = len(qm_scores)
        if n_stocks == 0:
            return {}
        
        # Base equal weight
        base_weight = 1.0 / n_stocks
        
        weights = {}
        
        if regime == Regime.BULL:
            # Apply QM overlay
            for symbol in qm_scores.index:
                rank = qm_scores.loc[symbol, "qm_rank"]
                
                # Top tier: positive overlay
                if rank >= (1 - self._top_frac):
                    overlay = self._overlay_strength
                # Bottom tier: negative overlay
                elif rank <= self._bottom_frac:
                    overlay = -self._overlay_strength
                else:
                    overlay = 0.0
                
                weights[symbol] = max(0, base_weight + overlay)
        else:
            # Equal weight in non-BULL regimes
            for symbol in qm_scores.index:
                weights[symbol] = base_weight
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {s: w / total for s, w in weights.items()}
        
        return weights
    
    def _weights_to_signals(
        self,
        weights: Dict[str, float],
        timestamp: datetime
    ) -> List[Signal]:
        """Convert target weights to Signal objects"""
        signals = []
        
        for symbol, weight in weights.items():
            if weight <= 0:
                continue
            
            signal = Signal(
                symbol=symbol,
                side=Side.BUY,
                signal_type=SignalType.REBALANCE,
                target_weight=weight,
                confidence=1.0,
                timestamp=timestamp,
                metadata={
                    "strategy": self._name,
                    "regime": self._current_regime.value,
                }
            )
            signals.append(signal)
        
        return signals
    
    def get_qm_returns(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.Series:
        """
        Generate QM strategy returns for backtesting
        
        This simulates the return stream that would be fed to
        the ensemble/AARM pipeline.
        """
        if self._prices is None:
            return pd.Series()
        
        # Filter date range
        prices = self._prices.loc[start_date:end_date]
        returns = prices.pct_change()
        
        # Generate daily returns based on daily rebalancing
        strategy_returns = []
        
        for date in returns.index[1:]:
            # Get QM scores as of previous day
            qm_scores = self._calculate_qm_scores(date - timedelta(days=1))
            
            if qm_scores.empty:
                strategy_returns.append(0.0)
                continue
            
            # Generate weights (assume BULL for now)
            weights = self._generate_target_weights(qm_scores, Regime.BULL)
            
            # Calculate return
            daily_ret = 0.0
            for symbol, weight in weights.items():
                if symbol in returns.columns:
                    r = returns.loc[date, symbol]
                    if not np.isnan(r):
                        daily_ret += weight * r
            
            strategy_returns.append(daily_ret)
        
        return pd.Series(strategy_returns, index=returns.index[1:], name="qm_returns")


class DefensiveStrategy(IStrategyEngine):
    """
    Defensive Strategy - Low Volatility / Equal Weight
    
    Used as complement to QM in BEAR/HIGH_VOL regimes.
    """
    
    def __init__(self):
        self._name = "DEFENSIVE"
        self._prices: Optional[pd.DataFrame] = None
        self._universe: List[str] = []
    
    @property
    def name(self) -> str:
        return self._name
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize defensive strategy"""
        pass
    
    def update_data(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None
    ) -> None:
        """Update data"""
        self._prices = prices
        if prices is not None:
            self._universe = list(prices.columns)
    
    def generate_signals(
        self,
        as_of: datetime,
        portfolio_state: PortfolioState,
        risk_metrics: RiskMetrics
    ) -> List[Signal]:
        """Generate defensive signals (equal weight)"""
        if not self._universe:
            return []
        
        weight = 1.0 / len(self._universe)
        
        signals = [
            Signal(
                symbol=symbol,
                side=Side.BUY,
                signal_type=SignalType.REBALANCE,
                target_weight=weight,
                confidence=1.0,
                timestamp=as_of,
                metadata={"strategy": self._name}
            )
            for symbol in self._universe
        ]
        
        return signals
    
    def get_defensive_returns(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.Series:
        """Generate defensive (equal weight) returns"""
        if self._prices is None:
            return pd.Series()
        
        prices = self._prices.loc[start_date:end_date]
        returns = prices.pct_change()
        
        # Equal weight
        strategy_returns = returns.mean(axis=1)
        
        return strategy_returns.rename("defensive_returns")
