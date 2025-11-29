#!/usr/bin/env python3
"""
EC2 Quick Validation Test
==========================
EC2 환경에서 빠른 검증 테스트

1. 모듈 import 테스트
2. 환경변수 확인
3. Synthetic 백테스트 실행
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 80)
print("EC2 Quick Validation Test")
print("=" * 80)
print()

# Test 1: Environment Variables
print("Test 1: Environment Variables")
print("-" * 80)

required_vars = [
    "POLYGON_API_KEY",
    "FRED_API_KEY",
    "SHARADAR_API_KEY",
]

all_set = True
for var in required_vars:
    value = os.getenv(var)
    if value:
        print(f"✅ {var}: {value[:20]}...")
    else:
        print(f"❌ {var}: NOT SET")
        all_set = False

if not all_set:
    print("\n❌ Some environment variables are missing")
    sys.exit(1)

print("\n✅ All required environment variables are set!")

# Test 2: Core Module Imports
print("\nTest 2: Core Module Imports")
print("-" * 80)

try:
    from core.interfaces import Regime, IBroker, IRiskManager, IStrategyEngine
    from core.utils import get_logger, load_config
    from risk.regime_filter import RegimeFilter
    from risk.aarm_core import TurboAARM
    from ensemble.turbo_aarm import TurboAARMEnsemble
    from backtest.run_backtest import BacktestRunner, BacktestConfig, run_synthetic_backtest
    print("✅ All core modules imported successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Synthetic Backtest
print("\nTest 3: Synthetic Backtest (Fast)")
print("-" * 80)

try:
    print("Running synthetic backtest...")
    
    # Run synthetic backtest
    output = run_synthetic_backtest()
    
    # Print results
    print("\n" + "=" * 80)
    print("Backtest Results")
    print("=" * 80)
    print(f"Full Period Sharpe:  {output.metrics_full.sharpe_ratio:.2f}")
    print(f"Full Period MDD:     {output.metrics_full.max_drawdown:.2%}")
    print(f"Train Sharpe:        {output.metrics_train.sharpe_ratio:.2f}")
    print(f"Test (OOS) Sharpe:   {output.metrics_test.sharpe_ratio:.2f}")
    print(f"Test (OOS) MDD:      {output.metrics_test.max_drawdown:.2%}")
    print()
    
    # Check if results are reasonable
    if output.metrics_full.sharpe_ratio > 0.5:
        print("✅ Synthetic backtest completed successfully")
        print("✅ Performance metrics are reasonable")
    else:
        print("⚠️  Synthetic backtest completed but metrics are low")
    
except Exception as e:
    print(f"❌ Synthetic backtest failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final Summary
print("\n" + "=" * 80)
print("✅ EC2 QUICK VALIDATION PASSED!")
print("=" * 80)
print()
print("Summary:")
print("  ✅ Environment variables set")
print("  ✅ All core modules import successfully")
print("  ✅ Synthetic backtest runs successfully")
print()
print("EC2 deployment is ready for real data backtest!")
print()
