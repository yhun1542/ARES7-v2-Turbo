#!/usr/bin/env python3
"""
Turbo Backtest Runner
=====================
Polygon Flatfiles + CPU 최적화 통합 실행

실행 방법:
    python run_turbo_backtest.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import structlog

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.polygon_flatfiles_daily_loader import PolygonFlatfilesDailyLoader
from backtest.turbo_cpu_backtest import TurboCPUBacktest
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 로거 설정
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer()
    ]
)
logger = structlog.get_logger()


def main():
    """메인 실행 함수"""
    
    print("=" * 80)
    print("ARES-Ultimate Turbo Backtest")
    print("=" * 80)
    print()
    print("🚀 Polygon Flatfiles + CPU Optimization")
    print("⚡ Expected speed: 50-60x improvement")
    print()
    
    # S&P 100 심볼
    sp100_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'UNH', 'JNJ',
        'V', 'XOM', 'WMT', 'JPM', 'PG', 'MA', 'CVX', 'HD', 'LLY', 'ABBV',
        'MRK', 'PEP', 'KO', 'COST', 'AVGO', 'MCD', 'CSCO', 'TMO', 'ACN', 'ABT',
        'ADBE', 'DHR', 'VZ', 'NKE', 'NFLX', 'CRM', 'TXN', 'NEE', 'PM', 'UNP',
        'RTX', 'ORCL', 'BMY', 'HON', 'QCOM', 'LOW', 'UPS', 'INTC', 'LIN', 'AMGN',
        'BA', 'SBUX', 'INTU', 'AMD', 'CAT', 'GE', 'DE', 'SPGI', 'AXP', 'BLK',
        'MDLZ', 'GILD', 'MMM', 'PLD', 'ADI', 'CI', 'ISRG', 'TJX', 'BKNG', 'SYK',
        'REGN', 'ZTS', 'MO', 'CVS', 'DUK', 'CB', 'SO', 'PGR', 'TGT', 'CL',
        'SCHW', 'USB', 'BDX', 'EOG', 'MMC', 'ITW', 'AON', 'HCA', 'SLB', 'APD',
        'NSC', 'FIS', 'CME', 'COP', 'ICE', 'EL', 'WM', 'EMR', 'GD', 'NOC'
    ]
    
    # 백테스트 기간
    start_date = "2016-03-01"
    end_date = "2025-11-18"
    
    # Step 1: Polygon Flatfiles에서 데이터 로드 (일별 CSV)
    logger.info("=" * 80)
    logger.info("Step 1: Loading data from Polygon Flatfiles (Daily CSV)")
    logger.info("=" * 80)
    
    loader = PolygonFlatfilesDailyLoader(
        access_key_id=os.getenv("POLYGON_ACCESS_KEY_ID"),
        secret_access_key=os.getenv("POLYGON_SECRET_ACCESS_KEY"),
        endpoint_url="https://files.massive.com",
        cache_dir="/home/ubuntu/workspace/ARES-Ultimate-251129/data_cache"
    )
    
    # 데이터 로드
    data = loader.load_stocks_data(
        symbols=sp100_symbols,
        start_date=start_date,
        end_date=end_date,
        use_cache=True
    )
    
    if data.empty:
        logger.error("No data loaded! Exiting.")
        sys.exit(1)
    
    logger.info(f"✅ Data loaded: {len(data)} rows, {data['symbol'].nunique()} symbols")
    
    # Step 2: CPU 최적화 백테스트 실행
    logger.info("")
    logger.info("=" * 80)
    logger.info("Step 2: Running Turbo CPU Backtest")
    logger.info("=" * 80)
    
    backtest = TurboCPUBacktest()
    
    results = backtest.run_optimized_backtest(
        data=data,
        train_window=1260,  # 5년 -> 2.5년 (252*5)
        test_ratio=0.3
    )
    
    # Step 3: 결과 출력
    logger.info("")
    logger.info("=" * 80)
    logger.info("Backtest Results")
    logger.info("=" * 80)
    
    metrics = results['metrics']
    
    print()
    print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:        {metrics['max_drawdown']:.2%}")
    print(f"Total Return:        {metrics['total_return']:.2%}")
    print(f"Annualized Return:   {metrics['annualized_return']:.2%}")
    print(f"Number of Days:      {metrics['n_days']}")
    print()
    
    # Step 4: 결과 저장
    output_dir = Path("/home/ubuntu/workspace/ARES-Ultimate-251129/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"turbo_backtest_results_{timestamp}.parquet"
    
    results['returns'].to_parquet(output_file, index=False)
    
    logger.info(f"✅ Results saved to: {output_file}")
    
    print()
    print("=" * 80)
    print("✅ Turbo Backtest Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
