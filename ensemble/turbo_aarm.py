"""
Turbo AARM Ensemble
====================
ARES7 QM Regime Turbo 전체 파이프라인

Ensemble + AARM을 결합하여:
1. Regime 판별
2. 엔진별 수익률/시그널 블렌딩
3. Turbo AARM 적용

성능 재현 목표:
- Sharpe: 3.86
- OOS Sharpe: 4.37
- MDD: -12.63%
- OOS MDD: -10.10%
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.interfaces import Regime
from core.utils import get_logger, calculate_sharpe, calculate_max_drawdown
from ensemble.dynamic_ensemble import DynamicEnsemble, EngineReturns, RegimeWeights
from risk.aarm_core import TurboAARM
from risk.regime_filter import RegimeFilter

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """백테스트 결과"""
    returns: pd.Series
    metrics_full: Dict[str, float]
    metrics_train: Dict[str, float]
    metrics_test: Dict[str, float]
    regime_series: pd.Series
    config: Dict[str, Any]


class TurboAARMEnsemble:
    """
    Turbo AARM Ensemble Pipeline
    
    Full pipeline:
    1. RegimeFilter: SPX/VIX/FRED → Regime
    2. DynamicEnsemble: QM + Defensive → Blended returns
    3. TurboAARM: Vol targeting + DD scaling + CB → Managed returns
    """
    
    def __init__(
        self,
        regime_filter: Optional[RegimeFilter] = None,
        ensemble: Optional[DynamicEnsemble] = None,
        turbo_aarm: Optional[TurboAARM] = None,
    ):
        """
        Initialize Turbo AARM Ensemble
        
        Args:
            regime_filter: Regime filter (uses default if None)
            ensemble: Dynamic ensemble (uses default if None)
            turbo_aarm: Turbo AARM (uses default if None)
        """
        self.regime_filter = regime_filter or RegimeFilter()
        self.ensemble = ensemble or DynamicEnsemble()
        self.turbo_aarm = turbo_aarm or TurboAARM()
    
    def run_backtest(
        self,
        qm_returns: pd.Series,
        defensive_returns: pd.Series,
        spx_prices: pd.Series,
        vix_series: pd.Series,
        train_ratio: float = 0.7,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> BacktestResult:
        """
        Run full Turbo AARM backtest
        
        Args:
            qm_returns: QM engine returns
            defensive_returns: Defensive engine returns
            spx_prices: SPX prices for regime detection
            vix_series: VIX series
            train_ratio: Train/test split ratio
            start_date: Optional start date filter
            end_date: Optional end date filter
        
        Returns:
            BacktestResult with returns and metrics
        """
        logger.info("Starting Turbo AARM backtest...")
        
        # 1. Generate regime series
        logger.info("Step 1: Generating regime series...")
        regime_series = self.regime_filter.get_regime_series(
            spx_prices=spx_prices,
            vix_series=vix_series,
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info(f"Regime distribution: {regime_series.value_counts().to_dict()}")
        
        # 2. Blend engine returns
        logger.info("Step 2: Blending engine returns...")
        engine_returns = EngineReturns(
            qm=qm_returns,
            defensive=defensive_returns
        )
        
        ensemble_returns = self.ensemble.blend_returns(
            engine_returns=engine_returns,
            regime_series=regime_series
        )
        
        # 3. Apply Turbo AARM
        logger.info("Step 3: Applying Turbo AARM...")
        managed_returns = self.turbo_aarm.apply_turbo(
            base_returns=ensemble_returns,
            regime_series=regime_series
        )
        
        # 4. Calculate metrics
        logger.info("Step 4: Calculating metrics...")
        
        # Train/Test split
        split_idx = int(len(managed_returns) * train_ratio)
        train_returns = managed_returns.iloc[:split_idx]
        test_returns = managed_returns.iloc[split_idx:]
        
        metrics_full = self._calculate_metrics(managed_returns)
        metrics_train = self._calculate_metrics(train_returns)
        metrics_test = self._calculate_metrics(test_returns)
        
        logger.info(f"Full: Sharpe={metrics_full['sharpe']:.2f}, MDD={metrics_full['mdd']:.2%}")
        logger.info(f"Train: Sharpe={metrics_train['sharpe']:.2f}, MDD={metrics_train['mdd']:.2%}")
        logger.info(f"Test (OOS): Sharpe={metrics_test['sharpe']:.2f}, MDD={metrics_test['mdd']:.2%}")
        
        return BacktestResult(
            returns=managed_returns,
            metrics_full=metrics_full,
            metrics_train=metrics_train,
            metrics_test=metrics_test,
            regime_series=regime_series,
            config={
                "train_ratio": train_ratio,
                "start_date": str(start_date) if start_date else None,
                "end_date": str(end_date) if end_date else None,
                "turbo_config": {
                    "base_leverage": self.turbo_aarm.config.base_leverage,
                    "max_leverage": self.turbo_aarm.config.max_leverage,
                    "target_volatility": self.turbo_aarm.config.target_volatility,
                    "cb_trigger": self.turbo_aarm.config.cb_trigger,
                    "cb_reduction_factor": self.turbo_aarm.config.cb_reduction_factor,
                }
            }
        )
    
    def _calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics"""
        if len(returns) < 2:
            return {
                "sharpe": 0.0,
                "return": 0.0,
                "mdd": 0.0,
                "volatility": 0.0,
                "calmar": 0.0,
            }
        
        n_years = len(returns) / 252
        
        # Annualized return
        cum_return = (1 + returns).prod() - 1
        ann_return = (1 + cum_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Volatility
        ann_vol = returns.std() * np.sqrt(252)
        
        # Sharpe
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Max drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        mdd = drawdowns.min()
        
        # Calmar
        calmar = ann_return / abs(mdd) if mdd < 0 else 0
        
        return {
            "sharpe": sharpe,
            "return": ann_return,
            "mdd": mdd,
            "volatility": ann_vol,
            "calmar": calmar,
            "total_return": cum_return,
            "n_days": len(returns),
        }
    
    def run_grid_search(
        self,
        base_returns: pd.Series,
        regime_series: pd.Series,
        param_grid: Optional[Dict[str, List]] = None,
        train_ratio: float = 0.7,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Run grid search to find optimal AARM parameters
        
        Args:
            base_returns: Base ensemble returns
            regime_series: Regime series
            param_grid: Parameter grid (uses default if None)
            train_ratio: Train ratio for validation
        
        Returns:
            (best_params, results_df)
        """
        import itertools
        
        if param_grid is None:
            param_grid = {
                "base_leverage": [0.8, 1.0, 1.2],
                "max_leverage": [1.5, 1.8, 2.0],
                "target_volatility": [0.15, 0.18, 0.20],
                "cb_trigger": [-0.06, -0.08, -0.10],
                "cb_reduction_factor": [0.3, 0.4, 0.5],
            }
        
        # Generate combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))
        
        logger.info(f"Grid search: {len(combinations)} combinations")
        
        # Train/test split
        split_idx = int(len(base_returns) * train_ratio)
        train_returns = base_returns.iloc[:split_idx]
        train_regime = regime_series.iloc[:split_idx]
        
        results = []
        
        for combo in combinations:
            params = dict(zip(keys, combo))
            
            # Skip invalid combos
            if params["max_leverage"] < params["base_leverage"]:
                continue
            
            # Create AARM with these params
            from risk.aarm_core import AARMConfig
            config = AARMConfig(**params)
            aarm = TurboAARM()
            aarm.config = config
            
            # Apply
            managed = aarm.apply_turbo(train_returns, train_regime)
            
            # Calculate metrics
            metrics = self._calculate_metrics(managed)
            
            results.append({
                **params,
                "sharpe": metrics["sharpe"],
                "return": metrics["return"],
                "mdd": metrics["mdd"],
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("sharpe", ascending=False)
        
        best_params = results_df.iloc[0].to_dict()
        
        logger.info(f"Best params: {best_params}")
        
        return best_params, results_df


def verify_performance(
    result: BacktestResult,
    target_sharpe_full: float = 3.86,
    target_sharpe_oos: float = 4.37,
    target_mdd_full: float = -0.1263,
    target_mdd_oos: float = -0.101,
    tolerance: float = 0.1
) -> Dict[str, Any]:
    """
    Verify backtest performance against targets
    
    Args:
        result: BacktestResult from run_backtest
        target_*: Target performance values
        tolerance: Acceptable deviation (10% default)
    
    Returns:
        Verification results
    """
    checks = {
        "sharpe_full": {
            "actual": result.metrics_full["sharpe"],
            "target": target_sharpe_full,
            "passed": abs(result.metrics_full["sharpe"] - target_sharpe_full) / target_sharpe_full < tolerance
        },
        "sharpe_oos": {
            "actual": result.metrics_test["sharpe"],
            "target": target_sharpe_oos,
            "passed": abs(result.metrics_test["sharpe"] - target_sharpe_oos) / target_sharpe_oos < tolerance
        },
        "mdd_full": {
            "actual": result.metrics_full["mdd"],
            "target": target_mdd_full,
            "passed": result.metrics_full["mdd"] >= target_mdd_full * (1 + tolerance)
        },
        "mdd_oos": {
            "actual": result.metrics_test["mdd"],
            "target": target_mdd_oos,
            "passed": result.metrics_test["mdd"] >= target_mdd_oos * (1 + tolerance)
        }
    }
    
    all_passed = all(c["passed"] for c in checks.values())
    
    return {
        "checks": checks,
        "all_passed": all_passed,
        "summary": f"{'✓ All checks passed' if all_passed else '✗ Some checks failed'}"
    }
