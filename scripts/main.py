#!/usr/bin/env python3
"""
ARES-Ultimate Main Entry Point
===============================

사용법:
    python scripts/main.py backtest     # 백테스트 실행
    python scripts/main.py live-ibkr    # IBKR 라이브 (shadow)
    python scripts/main.py live-kis     # KIS 라이브 (shadow)
    python scripts/main.py --help       # 도움말
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.utils import get_logger, setup_logging

logger = get_logger(__name__)


def run_backtest(args):
    """Run backtest"""
    from backtest.run_backtest import run_synthetic_backtest, run_full_backtest, generate_backtest_report, BacktestConfig
    from backtest.load_real_data import load_backtest_data
    from datetime import datetime
    import asyncio
    
    logger.info("Running backtest...")
    
    if args.synthetic:
        output = run_synthetic_backtest()
    else:
        # Load real data
        logger.info("Loading real data from Polygon, FRED, SF1...")
        
        config = BacktestConfig(
            start_date=datetime(2016, 3, 1),
            end_date=datetime(2025, 11, 18),
            initial_capital=1_000_000,
            train_ratio=0.7,
        )
        
        # Load data asynchronously
        data = asyncio.run(load_backtest_data(
            start_date=config.start_date,
            end_date=config.end_date,
            universe="SP100",
            use_cache=True
        ))
        
        logger.info(f"Data loaded: {data['prices'].shape[0]} days, {data['prices'].shape[1]} symbols")
        
        # Run backtest
        output = run_full_backtest(
            prices=data['prices'],
            spx=data['spx'],
            vix=data['vix'],
            config=config
        )
    
    report = generate_backtest_report(output)
    print(report)
    
    if args.save_report:
        with open(args.save_report, "w") as f:
            f.write(report)
        logger.info(f"Report saved to {args.save_report}")


async def run_live_ibkr(args):
    """Run live trading with IBKR"""
    from orchestration.live_orchestrator import (
        LiveOrchestrator,
        OrchestratorConfig,
        OrchestratorMode,
    )
    from engines.ares7_qm_regime.strategy import ARES7QMRegimeStrategy
    from risk.aarm_core import TurboAARM
    from brokers.ibkr_client import IBKRClient
    from data.polygon_client import PolygonClient
    
    mode = OrchestratorMode.LIVE if args.live else OrchestratorMode.SHADOW
    
    logger.info(f"Starting IBKR orchestrator in {mode.value} mode")
    
    strategy = ARES7QMRegimeStrategy("config/ares7_qm_turbo_final_251129.yaml")
    broker = IBKRClient(mode="paper" if not args.live else "live")
    data_providers = {"polygon": PolygonClient()}
    risk_manager = TurboAARM()
    
    config = OrchestratorConfig(
        mode=mode,
        update_interval_seconds=60,
        rebalance_frequency="daily",
        rebalance_time="15:30",
    )
    
    orchestrator = LiveOrchestrator(
        strategy=strategy,
        broker=broker,
        data_providers=data_providers,
        risk_manager=risk_manager,
        config=config,
    )
    
    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        await orchestrator.stop()


async def run_live_kis(args):
    """Run live trading with KIS"""
    from orchestration.live_orchestrator import (
        LiveOrchestrator,
        OrchestratorConfig,
        OrchestratorMode,
    )
    from engines.ares7_qm_regime.strategy import ARES7QMRegimeStrategy
    from risk.aarm_core import TurboAARM
    from brokers.kis_client import KISClient
    from data.polygon_client import PolygonClient
    
    mode = OrchestratorMode.LIVE if args.live else OrchestratorMode.SHADOW
    
    logger.info(f"Starting KIS orchestrator in {mode.value} mode")
    
    strategy = ARES7QMRegimeStrategy("config/ares7_qm_turbo_final_251129.yaml")
    broker = KISClient(mode="paper" if not args.live else "live")
    data_providers = {"polygon": PolygonClient()}
    risk_manager = TurboAARM()
    
    config = OrchestratorConfig(
        mode=mode,
        update_interval_seconds=60,
        rebalance_frequency="daily",
        rebalance_time="14:30",  # KRX time
    )
    
    orchestrator = LiveOrchestrator(
        strategy=strategy,
        broker=broker,
        data_providers=data_providers,
        risk_manager=risk_manager,
        config=config,
    )
    
    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        await orchestrator.stop()


def main():
    parser = argparse.ArgumentParser(
        description="ARES-Ultimate Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/main.py backtest --synthetic
    python scripts/main.py live-ibkr
    python scripts/main.py live-ibkr --live  # CAUTION: Real orders!
    python scripts/main.py live-kis
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Backtest command
    bt_parser = subparsers.add_parser("backtest", help="Run backtest")
    bt_parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    bt_parser.add_argument("--save-report", type=str, help="Save report to file")
    bt_parser.add_argument("--config", type=str, help="Strategy config path")
    
    # Live IBKR command
    ibkr_parser = subparsers.add_parser("live-ibkr", help="Run live with IBKR")
    ibkr_parser.add_argument("--live", action="store_true", help="Enable live orders (CAUTION!)")
    
    # Live KIS command
    kis_parser = subparsers.add_parser("live-kis", help="Run live with KIS")
    kis_parser.add_argument("--live", action="store_true", help="Enable live orders (CAUTION!)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    setup_logging()
    
    # Execute command
    if args.command == "backtest":
        run_backtest(args)
    elif args.command == "live-ibkr":
        asyncio.run(run_live_ibkr(args))
    elif args.command == "live-kis":
        asyncio.run(run_live_kis(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
