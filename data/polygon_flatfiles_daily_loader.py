"""
Polygon Flatfiles Daily CSV Loader
===================================
Massive.com Flatfiles에서 일별 CSV.GZ 파일 로드

파일 형식: us_stocks_sip/day_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz
"""

import os
import boto3
from botocore.config import Config
import pandas as pd
import gzip
from datetime import datetime, timedelta
from typing import List, Optional
from pathlib import Path
import structlog
from io import BytesIO, StringIO

logger = structlog.get_logger()


class PolygonFlatfilesDailyLoader:
    """Polygon Flatfiles 일별 CSV 로더"""
    
    def __init__(
        self,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        endpoint_url: str = "https://files.massive.com",
        bucket: str = "flatfiles",
        cache_dir: str = "/tmp/polygon_cache"
    ):
        self.access_key_id = access_key_id or os.getenv("POLYGON_ACCESS_KEY_ID")
        self.secret_access_key = secret_access_key or os.getenv("POLYGON_SECRET_ACCESS_KEY")
        self.endpoint_url = endpoint_url
        self.bucket = bucket
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # S3 클라이언트 초기화
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            endpoint_url=self.endpoint_url,
            config=Config(signature_version='s3v4')
        )
        
        logger.info("Polygon Flatfiles Daily Loader initialized",
                   endpoint=self.endpoint_url,
                   bucket=self.bucket)
    
    def load_stocks_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        주식 데이터 로드 (일별 CSV 파일)
        
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
        logger.info("Loading stocks data from daily CSV files",
                   n_symbols=len(symbols),
                   start_date=start_date,
                   end_date=end_date)
        
        # 날짜 범위 생성
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_data = []
        current_dt = start_dt
        
        while current_dt <= end_dt:
            # 주말 건너뛰기
            if current_dt.weekday() < 5:  # 월~금
                date_str = current_dt.strftime("%Y-%m-%d")
                
                # 일별 데이터 로드
                daily_data = self._load_daily_data(date_str, symbols, use_cache)
                
                if daily_data is not None and not daily_data.empty:
                    all_data.append(daily_data)
            
            # 다음 날로 이동
            current_dt += timedelta(days=1)
        
        if not all_data:
            logger.warning("No data loaded")
            return pd.DataFrame()
        
        # 데이터 병합
        df = pd.concat(all_data, ignore_index=True)
        
        # 심볼 필터링
        df = df[df['symbol'].isin(symbols)]
        
        # 정렬
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        logger.info("Data loaded successfully",
                   n_rows=len(df),
                   n_symbols=df['symbol'].nunique(),
                   date_range=f"{df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def _load_daily_data(
        self,
        date_str: str,
        symbols: List[str],
        use_cache: bool
    ) -> Optional[pd.DataFrame]:
        """일별 데이터 로드"""
        
        # 캐시 확인
        cache_file = self.cache_dir / f"daily_{date_str}.parquet"
        
        if use_cache and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.debug(f"Loaded from cache: {date_str}")
                return df
            except:
                pass
        
        # S3에서 다운로드
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        year = dt.year
        month = dt.month
        
        s3_key = f"us_stocks_sip/day_aggs_v1/{year}/{month:02d}/{date_str}.csv.gz"
        
        try:
            logger.debug(f"Downloading: {s3_key}")
            
            response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
            
            # GZIP 압축 해제
            with gzip.GzipFile(fileobj=BytesIO(response['Body'].read())) as gz:
                csv_content = gz.read().decode('utf-8')
            
            # CSV 파싱
            df = pd.read_csv(StringIO(csv_content))
            
            # 컬럼 정리
            if 'ticker' in df.columns:
                df = df.rename(columns={'ticker': 'symbol'})
            
            # 날짜 추가
            df['date'] = date_str
            
            # 필요한 컬럼만 선택
            required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
            df = df[[col for col in required_cols if col in df.columns]]
            
            # 캐시 저장
            if use_cache:
                df.to_parquet(cache_file, compression='snappy', index=False)
            
            logger.debug(f"Loaded {len(df)} rows for {date_str}")
            
            return df
            
        except Exception as e:
            logger.debug(f"Failed to load {date_str}: {e}")
            return None


# 사용 예시
if __name__ == "__main__":
    loader = PolygonFlatfilesDailyLoader()
    
    # 테스트: 최근 5일 데이터
    df = loader.load_stocks_data(
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        start_date="2024-11-01",
        end_date="2024-11-05",
        use_cache=True
    )
    
    print(df.head(20))
    print(f"\nTotal rows: {len(df)}")
