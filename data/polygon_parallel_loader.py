"""
Polygon REST API Parallel Loader
=================================
Polygon REST API를 병렬로 호출하여 빠르게 데이터 로드

유료 계정 장점:
- 높은 rate limit (분당 수천 건)
- 동시 요청 가능
- 빠른 응답 속도
"""

import os
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
from pathlib import Path
import structlog

logger = structlog.get_logger()


class PolygonParallelLoader:
    """Polygon REST API 병렬 데이터 로더"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_concurrent: int = 50,  # 동시 요청 수
        cache_dir: str = "/tmp/polygon_cache"
    ):
        """
        Parameters
        ----------
        api_key : str
            Polygon API 키
        max_concurrent : int
            최대 동시 요청 수
        cache_dir : str
            로컬 캐시 디렉토리
        """
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        self.max_concurrent = max_concurrent
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://api.polygon.io"
        
        logger.info("Polygon Parallel Loader initialized",
                   max_concurrent=max_concurrent)
    
    async def load_stocks_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        주식 데이터 병렬 로드
        
        Parameters
        ----------
        symbols : List[str]
            종목 심볼 리스트
        start_date : str
            시작 날짜 (YYYY-MM-DD)
        end_date : str
            종료 날짜 (YYYY-MM-DD)
        use_cache : bool
            캐시 사용 여부
            
        Returns
        -------
        pd.DataFrame
            OHLCV 데이터
        """
        logger.info("Loading stocks data in parallel",
                   n_symbols=len(symbols),
                   start_date=start_date,
                   end_date=end_date,
                   max_concurrent=self.max_concurrent)
        
        # 세마포어로 동시 요청 수 제한
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # 비동기 세션 생성
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for symbol in symbols:
                task = self._fetch_symbol_data(
                    session,
                    semaphore,
                    symbol,
                    start_date,
                    end_date,
                    use_cache
                )
                tasks.append(task)
            
            # 모든 요청 병렬 실행
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 병합
        valid_results = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]
        
        if not valid_results:
            logger.warning("No data loaded")
            return pd.DataFrame()
        
        df = pd.concat(valid_results, ignore_index=True)
        
        # 정렬
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        logger.info("Data loaded successfully",
                   n_rows=len(df),
                   n_symbols=df['symbol'].nunique(),
                   date_range=f"{df['date'].min()} to {df['date'].max()}")
        
        return df
    
    async def _fetch_symbol_data(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        symbol: str,
        start_date: str,
        end_date: str,
        use_cache: bool
    ) -> pd.DataFrame:
        """단일 심볼 데이터 가져오기"""
        
        # 캐시 확인
        cache_file = self.cache_dir / f"{symbol}_{start_date}_{end_date}.parquet"
        
        if use_cache and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.debug(f"Loaded from cache: {symbol}")
                return df
            except:
                pass
        
        # API 호출
        async with semaphore:
            try:
                url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
                params = {
                    "adjusted": "true",
                    "sort": "asc",
                    "limit": 50000,
                    "apiKey": self.api_key
                }
                
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'results' in data and data['results']:
                            df = pd.DataFrame(data['results'])
                            
                            # 컬럼 정리
                            df['symbol'] = symbol
                            df['date'] = pd.to_datetime(df['t'], unit='ms')
                            df = df.rename(columns={
                                'o': 'open',
                                'h': 'high',
                                'l': 'low',
                                'c': 'close',
                                'v': 'volume'
                            })
                            
                            df = df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']]
                            
                            # 캐시 저장
                            if use_cache:
                                df.to_parquet(cache_file, compression='snappy', index=False)
                            
                            logger.debug(f"Fetched {len(df)} rows for {symbol}")
                            
                            return df
                        else:
                            logger.warning(f"No data for {symbol}")
                            return pd.DataFrame()
                    else:
                        logger.warning(f"Failed to fetch {symbol}: HTTP {response.status}")
                        return pd.DataFrame()
                        
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                return pd.DataFrame()


async def main_async():
    """비동기 메인 함수"""
    
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
    
    # 로더 초기화
    loader = PolygonParallelLoader(max_concurrent=50)
    
    # 데이터 로드
    df = await loader.load_stocks_data(
        symbols=sp100_symbols,
        start_date="2016-03-01",
        end_date="2025-11-18",
        use_cache=True
    )
    
    # 저장
    if not df.empty:
        output_file = "/tmp/sp100_data_parallel.parquet"
        df.to_parquet(output_file, compression='snappy', index=False)
        
        file_size_mb = Path(output_file).stat().st_size / 1024 / 1024
        
        logger.info("Data saved",
                   output_file=output_file,
                   file_size_mb=f"{file_size_mb:.2f} MB",
                   n_rows=len(df),
                   n_symbols=df['symbol'].nunique())
        
        print(f"\n✅ Data download complete: {output_file}")
    else:
        print("\n❌ No data loaded")


# 사용 예시
if __name__ == "__main__":
    asyncio.run(main_async())
