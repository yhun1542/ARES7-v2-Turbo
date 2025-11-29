"""
Backtest Runner
================
E2E 백테스트 실행 모듈

ARES7 QM Regime Turbo 전략의 전체 백테스트 파이프라인.

성능 목표:
- In-Sample Sharpe: 3.86
- Out-of-Sample Sharpe: 4.37
- In-Sample MDD: -12.63%
- Out-of-Sample MDD: -10.10%
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.metrics import (
    PerformanceMetrics,
    calculate_all_metrics,
    compare_strategies,
)
from core.interfaces import Regime
from core.utils import get_logger, load_config
from ensemble.turbo_aarm import TurboAARMEnsemble, BacktestResult
from risk.regime_filter import RegimeFilter
from risk.aarm_core import TurboAARM

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """백테스트 설정"""
    start_date: datetime = datetime(2016, 3, 1)
    end_date: datetime = datetime(2025, 11, 18)
    initial_capital: float = 1_000_000
    train_ratio: float = 0.7
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    rebalance_frequency: str = "daily"


@dataclass
class BacktestOutput:
    """백테스트 출력"""
    returns: pd.Series
    cumulative_returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    metrics_full: PerformanceMetrics
    metrics_train: PerformanceMetrics
    metrics_test: PerformanceMetrics
    regime_series: pd.Series
    config: BacktestConfig
    

class BacktestRunner:
    """
    Backtest Runner
    
    전체 백테스트 파이프라인 실행.
    
    Pipeline:
    1. 데이터 로드 (prices, fundamentals, macro)
    2. 전략 수익률 생성 (QM, Defensive)
    3. 레짐 필터 적용
    4. 앙상블 블렌딩
    5. Turbo AARM 적용
    6. 성능 지표 계산
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[BacktestConfig] = None
    ):
        """
        Initialize backtest runner
        
        Args:
            config_path: Path to strategy YAML config
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
        
        # Load strategy config if provided
        self.strategy_config: Dict[str, Any] = {}
        if config_path:
            self.strategy_config = load_config(config_path)
        
        # Components
        self.regime_filter = RegimeFilter()
        self.turbo_aarm = TurboAARM()
        self.ensemble = TurboAARMEnsemble()
        
        # Data
        self._prices: Optional[pd.DataFrame] = None
        self._spx: Optional[pd.Series] = None
        self._vix: Optional[pd.Series] = None
    
    def load_data(
        self,
        prices: pd.DataFrame,
        spx: pd.Series,
        vix: pd.Series,
        fundamentals: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Load data for backtest
        
        Args:
            prices: Price DataFrame (date index, symbol columns)
            spx: SPX price series
            vix: VIX series
            fundamentals: Optional fundamentals DataFrame
        """
        self._prices = prices
        self._spx = spx
        self._vix = vix
        self._fundamentals = fundamentals
        
        logger.info(f"Data loaded: {len(prices)} days, {len(prices.columns)} symbols")
    
    def run(self) -> BacktestOutput:
        """
        Run full backtest
        
        Returns:
            BacktestOutput with all results
        """
        logger.info("Starting backtest...")
        
        if self._prices is None or self._spx is None or self._vix is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # 1. Filter date range
        prices = self._prices.loc[self.config.start_date:self.config.end_date]
        spx = self._spx.loc[self.config.start_date:self.config.end_date]
        vix = self._vix.loc[self.config.start_date:self.config.end_date]
        
        logger.info(f"Date range: {prices.index[0]} to {prices.index[-1]}")
        
        # 2. Generate regime series
        logger.info("Generating regime series...")
        regime_series = self.regime_filter.get_regime_series(
            spx_prices=spx,
            vix_series=vix
        )
        
        # 3. Generate strategy returns
        logger.info("Generating strategy returns...")
        qm_returns = self._generate_qm_returns(prices)
        defensive_returns = self._generate_defensive_returns(prices)
        
        # 4. Run ensemble + AARM pipeline
        logger.info("Running ensemble + AARM pipeline...")
        result = self.ensemble.run_backtest(
            qm_returns=qm_returns,
            defensive_returns=defensive_returns,
            spx_prices=spx,
            vix_series=vix,
            train_ratio=self.config.train_ratio,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
        )
        
        # 5. Calculate detailed metrics
        logger.info("Calculating detailed metrics...")
        
        split_idx = int(len(result.returns) * self.config.train_ratio)
        train_returns = result.returns.iloc[:split_idx]
        test_returns = result.returns.iloc[split_idx:]
        
        metrics_full = calculate_all_metrics(result.returns)
        metrics_train = calculate_all_metrics(train_returns)
        metrics_test = calculate_all_metrics(test_returns)
        
        # 6. Generate cumulative returns
        cumulative_returns = (1 + result.returns).cumprod()
        
        # 7. Generate positions (simplified)
        positions = pd.DataFrame(index=result.returns.index)
        positions["cash"] = 1 - result.returns.abs()
        positions["invested"] = result.returns.abs()
        
        # 8. Generate trades log (simplified)
        trades = pd.DataFrame({
            "date": result.returns.index,
            "return": result.returns.values,
            "regime": regime_series.reindex(result.returns.index, method="ffill").values
        })
        
        logger.info("Backtest completed!")
        logger.info(f"Full Sharpe: {metrics_full.sharpe_ratio:.2f}, MDD: {metrics_full.max_drawdown:.2%}")
        logger.info(f"Train Sharpe: {metrics_train.sharpe_ratio:.2f}, MDD: {metrics_train.max_drawdown:.2%}")
        logger.info(f"Test (OOS) Sharpe: {metrics_test.sharpe_ratio:.2f}, MDD: {metrics_test.max_drawdown:.2%}")
        
        return BacktestOutput(
            returns=result.returns,
            cumulative_returns=cumulative_returns,
            positions=positions,
            trades=trades,
            metrics_full=metrics_full,
            metrics_train=metrics_train,
            metrics_test=metrics_test,
            regime_series=result.regime_series,
            config=self.config,
        )
    
    def _generate_qm_returns(self, prices: pd.DataFrame) -> pd.Series:
        """
        Generate QM strategy returns
        
        Simplified implementation for backtest.
        """
        returns = prices.pct_change()
        
        # Calculate momentum factor
        momentum_6m = prices.pct_change(126)
        momentum_12m = prices.pct_change(252)
        
        # Simple QM: overweight top momentum, underweight bottom
        qm_returns = []
        
        for date in returns.index[253:]:  # Need 12m history
            try:
                # Get momentum as of previous day
                prev_date = returns.index[returns.index.get_loc(date) - 1]
                
                mom_6m = momentum_6m.loc[prev_date].dropna()
                mom_12m = momentum_12m.loc[prev_date].dropna()
                
                # Combined momentum rank
                combined_rank = (mom_6m.rank(pct=True) + mom_12m.rank(pct=True)) / 2
                
                # Top 20% get extra weight
                top_20 = combined_rank >= 0.8
                bottom_20 = combined_rank <= 0.2
                
                # Base equal weight
                n_stocks = len(combined_rank)
                base_weight = 1.0 / n_stocks if n_stocks > 0 else 0
                
                # Apply overlay
                weights = pd.Series(base_weight, index=combined_rank.index)
                overlay = 0.04  # 4% overlay
                weights[top_20] += overlay
                weights[bottom_20] -= overlay
                weights = weights.clip(lower=0)
                weights = weights / weights.sum()  # Normalize
                
                # Calculate return
                daily_returns = returns.loc[date]
                qm_ret = (weights * daily_returns).sum()
                
                qm_returns.append(qm_ret)
                
            except Exception:
                qm_returns.append(0.0)
        
        return pd.Series(
            qm_returns,
            index=returns.index[253:],
            name="qm_returns"
        )
    
    def _generate_defensive_returns(self, prices: pd.DataFrame) -> pd.Series:
        """Generate defensive (equal weight) returns"""
        returns = prices.pct_change()
        
        # Equal weight
        defensive_returns = returns.mean(axis=1)
        
        return defensive_returns.iloc[253:].rename("defensive_returns")


def run_full_backtest(
    prices: pd.DataFrame,
    spx: pd.Series,
    vix: pd.Series,
    config: Optional[BacktestConfig] = None,
    strategy_config_path: Optional[str] = None
) -> BacktestOutput:
    """
    Convenience function to run full backtest
    
    Args:
        prices: Price DataFrame
        spx: SPX series
        vix: VIX series
        config: Backtest config
        strategy_config_path: Path to strategy YAML
    
    Returns:
        BacktestOutput
    """
    runner = BacktestRunner(
        config_path=strategy_config_path,
        config=config
    )
    
    runner.load_data(prices=prices, spx=spx, vix=vix)
    
    return runner.run()


def generate_backtest_report(output: BacktestOutput, save_path: Optional[str] = None) -> str:
    """
    Generate backtest report
    
    Args:
        output: BacktestOutput
        save_path: Optional path to save report
    
    Returns:
        Report as string
    """
    report = []
    report.append("=" * 60)
    report.append("ARES-Ultimate Backtest Report")
    report.append("=" * 60)
    report.append("")
    
    # Configuration
    report.append("Configuration")
    report.append("-" * 40)
    report.append(f"Start Date:       {output.config.start_date.date()}")
    report.append(f"End Date:         {output.config.end_date.date()}")
    report.append(f"Initial Capital:  ${output.config.initial_capital:,.0f}")
    report.append(f"Train Ratio:      {output.config.train_ratio:.0%}")
    report.append(f"Transaction Cost: {output.config.transaction_cost_bps} bps")
    report.append("")
    
    # Full Period Metrics
    report.append("Full Period Performance")
    report.append("-" * 40)
    report.append(output.metrics_full.summary())
    
    # Train Period Metrics
    report.append("In-Sample (Train) Performance")
    report.append("-" * 40)
    report.append(output.metrics_train.summary())
    
    # Test Period Metrics
    report.append("Out-of-Sample (Test) Performance")
    report.append("-" * 40)
    report.append(output.metrics_test.summary())
    
    # Regime Distribution
    report.append("Regime Distribution")
    report.append("-" * 40)
    regime_counts = output.regime_series.value_counts()
    for regime, count in regime_counts.items():
        pct = count / len(output.regime_series)
        report.append(f"{regime:12s}: {count:5d} days ({pct:6.2%})")
    report.append("")
    
    # Performance vs Targets
    report.append("Performance vs Targets")
    report.append("-" * 40)
    targets = {
        "In-Sample Sharpe": (output.metrics_train.sharpe_ratio, 3.86),
        "OOS Sharpe": (output.metrics_test.sharpe_ratio, 4.37),
        "In-Sample MDD": (output.metrics_train.max_drawdown, -0.1263),
        "OOS MDD": (output.metrics_test.max_drawdown, -0.1010),
    }
    
    for name, (actual, target) in targets.items():
        if "MDD" in name:
            status = "✓" if actual >= target else "✗"
            report.append(f"{name:20s}: {actual:>8.2%} (target: {target:>8.2%}) {status}")
        else:
            status = "✓" if actual >= target * 0.9 else "✗"  # 10% tolerance
            report.append(f"{name:20s}: {actual:>8.2f} (target: {target:>8.2f}) {status}")
    
    report.append("")
    report.append("=" * 60)
    
    report_str = "\n".join(report)
    
    if save_path:
        with open(save_path, "w") as f:
            f.write(report_str)
        logger.info(f"Report saved to {save_path}")
    
    return report_str


# =============================================================================
# Example usage with synthetic data
# =============================================================================

def run_synthetic_backtest():
    """Run backtest with synthetic data for testing"""
    import numpy as np
    
    np.random.seed(42)
    
    # Generate synthetic data
    dates = pd.date_range("2016-03-01", "2025-11-18", freq="B")
    n_days = len(dates)
    n_stocks = 100
    
    # Random walk prices
    returns = np.random.randn(n_days, n_stocks) * 0.02
    prices = pd.DataFrame(
        np.exp(np.cumsum(returns, axis=0)) * 100,
        index=dates,
        columns=[f"STOCK{i:03d}" for i in range(n_stocks)]
    )
    
    # SPX (slight upward bias)
    spx_returns = np.random.randn(n_days) * 0.01 + 0.0003
    spx = pd.Series(
        np.exp(np.cumsum(spx_returns)) * 4000,
        index=dates,
        name="SPX"
    )
    
    # VIX (mean reverting)
    vix_base = 20
    vix = pd.Series(
        vix_base + np.cumsum(np.random.randn(n_days) * 0.5),
        index=dates,
        name="VIX"
    ).clip(10, 80)
    
    # Run backtest
    config = BacktestConfig(
        start_date=datetime(2016, 3, 1),
        end_date=datetime(2025, 11, 18),
        initial_capital=1_000_000,
        train_ratio=0.7,
    )
    
    output = run_full_backtest(
        prices=prices,
        spx=spx,
        vix=vix,
        config=config
    )
    
    # Generate report
    report = generate_backtest_report(output)
    print(report)
    
    return output


if __name__ == "__main__":
    run_synthetic_backtest()
