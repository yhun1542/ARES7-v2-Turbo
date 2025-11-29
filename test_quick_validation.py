#!/usr/bin/env python3
"""
Quick Validation Test
======================
ARES-Ultimate 패키지의 핵심 기능을 빠르게 검증하는 스크립트
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("=" * 60)
print("ARES-Ultimate Quick Validation Test")
print("=" * 60)
print()

# Test 1: Core imports
print("Test 1: Core Imports")
print("-" * 60)
try:
    from core.interfaces import Regime, IBroker, IRiskManager, IStrategyEngine
    from core.utils import get_logger, load_config
    print("✅ Core interfaces imported successfully")
except Exception as e:
    print(f"❌ Core imports failed: {e}")
    sys.exit(1)

# Test 2: Risk module imports
print("\nTest 2: Risk Module Imports")
print("-" * 60)
try:
    from risk.regime_filter import RegimeFilter
    from risk.aarm_core import TurboAARM
    from risk.cvar_utils import calculate_cvar
    print("✅ Risk modules imported successfully")
except Exception as e:
    print(f"❌ Risk imports failed: {e}")
    sys.exit(1)

# Test 3: Ensemble imports
print("\nTest 3: Ensemble Module Imports")
print("-" * 60)
try:
    from ensemble.turbo_aarm import TurboAARMEnsemble
    from ensemble.dynamic_ensemble import DynamicEnsemble
    print("✅ Ensemble modules imported successfully")
except Exception as e:
    print(f"❌ Ensemble imports failed: {e}")
    sys.exit(1)

# Test 4: Engine imports
print("\nTest 4: Engine Module Imports")
print("-" * 60)
try:
    from engines.ares7_qm_regime.strategy import ARES7QMRegimeStrategy
    print("✅ Engine modules imported successfully")
except Exception as e:
    print(f"❌ Engine imports failed: {e}")
    sys.exit(1)

# Test 5: Broker imports
print("\nTest 5: Broker Module Imports")
print("-" * 60)
try:
    from brokers.base_broker import BaseBroker
    from brokers.ibkr_client import IBKRClient
    from brokers.kis_client import KISClient
    print("✅ Broker modules imported successfully")
except Exception as e:
    print(f"❌ Broker imports failed: {e}")
    sys.exit(1)

# Test 6: Data client imports
print("\nTest 6: Data Client Imports")
print("-" * 60)
try:
    from data.polygon_client import PolygonClient
    from data.fred_client import FREDClient
    from data.sf1_client import SF1Client
    print("✅ Data clients imported successfully")
except Exception as e:
    print(f"❌ Data client imports failed: {e}")
    sys.exit(1)

# Test 7: Config loading
print("\nTest 7: Configuration Loading")
print("-" * 60)
try:
    config = load_config("config/ares7_qm_turbo_final_251129.yaml")
    print(f"✅ Config loaded: {config['strategy']['name']}")
    print(f"   Version: {config['strategy']['version']}")
    print(f"   Target Sharpe: 3.86 (Full), 4.37 (OOS)")
except Exception as e:
    print(f"❌ Config loading failed: {e}")
    sys.exit(1)

# Test 8: Regime Filter functionality
print("\nTest 8: Regime Filter Functionality")
print("-" * 60)
try:
    # Create synthetic data
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    spx_prices = pd.Series(100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01), index=dates)
    vix_series = pd.Series(15 + np.random.randn(len(dates)) * 5, index=dates).clip(lower=10)
    
    regime_filter = RegimeFilter()
    
    # Test single regime detection
    regime = regime_filter.detect_regime(
        spx_price=spx_prices.iloc[-1],
        spx_ma200=spx_prices.rolling(200).mean().iloc[-1],
        vix=vix_series.iloc[-1],
        ret_6m=0.05,
        ret_12m=0.10
    )
    print(f"✅ Regime detection works: {regime.value}")
    
except Exception as e:
    print(f"❌ Regime filter test failed: {e}")
    sys.exit(1)

# Test 9: Turbo AARM functionality
print("\nTest 9: Turbo AARM Functionality")
print("-" * 60)
try:
    # Create synthetic returns
    returns = pd.Series(np.random.randn(252) * 0.01, index=pd.date_range('2020-01-01', periods=252))
    
    aarm = TurboAARM(
        base_leverage=1.2,
        max_leverage=1.8,
        target_volatility=0.18,
        cb_trigger=-0.06,
        cb_reduction_factor=0.4
    )
    
    # Test position sizing
    position_size = aarm.calculate_position_size(
        current_returns=returns,
        current_drawdown=-0.03
    )
    
    print(f"✅ Turbo AARM works: position_size={position_size:.2f}")
    
except Exception as e:
    print(f"❌ Turbo AARM test failed: {e}")
    sys.exit(1)

# Test 10: Ensemble functionality
print("\nTest 10: Ensemble Functionality")
print("-" * 60)
try:
    # Create synthetic strategy returns
    dates = pd.date_range(start='2020-01-01', periods=252)
    qm_returns = pd.Series(np.random.randn(252) * 0.015 + 0.0005, index=dates)
    defensive_returns = pd.Series(np.random.randn(252) * 0.008 + 0.0003, index=dates)
    
    ensemble = DynamicEnsemble()
    
    # Test blending
    blended_returns = ensemble.blend_strategies(
        qm_returns=qm_returns,
        defensive_returns=defensive_returns,
        regime=Regime.BULL
    )
    
    print(f"✅ Ensemble blending works: mean_return={blended_returns.mean():.6f}")
    
except Exception as e:
    print(f"❌ Ensemble test failed: {e}")
    sys.exit(1)

# Final summary
print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print()
print("Summary:")
print("  - All core modules import successfully")
print("  - Configuration loads correctly")
print("  - Regime Filter works")
print("  - Turbo AARM works")
print("  - Ensemble blending works")
print()
print("✅ ARES-Ultimate package is ready for deployment!")
print()
