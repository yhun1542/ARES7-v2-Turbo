"""
Polygon Flatfiles S3 Data Loader
==================================
Polygon S3 Flatfiles에서 대량 데이터를 빠르게 로드

장점:
- API 호출 없이 S3에서 직접 다운로드
- Rate limit 완전 우회
- Parquet 형식으로 최적화됨
- 수년치 데이터를 몇 초 만에 다운로드
"""

import os
import boto3
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import structlog

logger = structlog.get_logger()


class PolygonFlatfilesLoader:
    """Polygon S3 Flatfiles 데이터 로더"""
    
    def __init__(
        self,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        endpoint_url: str = "https://files.massive.com",
        bucket: str = "flatfiles",
        cache_dir: str = "/tmp/polygon_cache"
    ):
        """
        Parameters
        ----------
        access_key_id : str
            Polygon S3 Access Key ID
        secret_access_key : str
            Polygon S3 Secret Access Key
        endpoint_url : str
            S3 endpoint URL
        bucket : str
            S3 bucket name
        cache_dir : str
            로컬 캐시 디렉토리
        """
        self.access_key_id = access_key_id or os.getenv("POLYGON_ACCESS_KEY_ID")
        self.secret_access_key = secret_access_key or os.getenv("POLYGON_SECRET_ACCESS_KEY")
        self.endpoint_url = endpoint_url
        self.bucket = bucket
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # S3 클라이언트 초기화
        from botocore.config import Config
        
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            endpoint_url=self.endpoint_url,
            config=Config(signature_version='s3v4')
        )
        
        logger.info("Polygon Flatfiles loader initialized", 
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
        주식 데이터 로드 (S3 Flatfiles)
        
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
        logger.info("Loading stocks data from Flatfiles",
                   n_symbols=len(symbols),
                   start_date=start_date,
                   end_date=end_date)
        
        # 날짜 범위 생성
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_data = []
        
        # 월별로 데이터 로드 (Flatfiles는 월별로 저장됨)
        current_dt = start_dt
        while current_dt <= end_dt:
            year = current_dt.year
            month = current_dt.month
            
            # 캐시 확인
            cache_file = self.cache_dir / f"stocks_{year}_{month:02d}.parquet"
            
            if use_cache and cache_file.exists():
                logger.info(f"Loading from cache: {cache_file}")
                monthly_data = pd.read_parquet(cache_file)
            else:
                # S3에서 다운로드
                monthly_data = self._download_monthly_data(year, month)
                
                if monthly_data is not None and use_cache:
                    monthly_data.to_parquet(cache_file, compression='snappy')
                    logger.info(f"Cached to: {cache_file}")
            
            if monthly_data is not None:
                # 심볼 필터링
                monthly_data = monthly_data[monthly_data['symbol'].isin(symbols)]
                all_data.append(monthly_data)
            
            # 다음 달로 이동
            if month == 12:
                current_dt = datetime(year + 1, 1, 1)
            else:
                current_dt = datetime(year, month + 1, 1)
        
        if not all_data:
            logger.warning("No data loaded")
            return pd.DataFrame()
        
        # 데이터 병합
        df = pd.concat(all_data, ignore_index=True)
        
        # 날짜 필터링
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        # 정렬
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        logger.info("Data loaded successfully",
                   n_rows=len(df),
                   n_symbols=df['symbol'].nunique(),
                   date_range=f"{df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def _download_monthly_data(self, year: int, month: int) -> Optional[pd.DataFrame]:
        """
        월별 데이터 다운로드
        
        Polygon Flatfiles 경로 예시:
        us_stocks_sip/day_aggs_v1/2024/01/2024-01.parquet
        """
        s3_key = f"us_stocks_sip/day_aggs_v1/{year}/{month:02d}/{year}-{month:02d}.parquet"
        
        logger.info(f"Downloading from S3: s3://{self.bucket}/{s3_key}")
        
        try:
            # S3에서 다운로드
            response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
            
            # Parquet 읽기
            df = pd.read_parquet(response['Body'])
            
            logger.info(f"Downloaded {len(df)} rows for {year}-{month:02d}")
            
            return df
            
        except Exception as e:
            logger.warning(f"Failed to download {s3_key}: {e}")
            return None
    
    def list_available_files(self, prefix: str = "us_stocks_sip/day_aggs_v1/") -> List[str]:
        """
        사용 가능한 파일 목록 조회
        
        Parameters
        ----------
        prefix : str
            S3 prefix
            
        Returns
        -------
        List[str]
            파일 경로 리스트
        """
        logger.info(f"Listing files with prefix: {prefix}")
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )
            
            files = [obj['Key'] for obj in response.get('Contents', [])]
            
            logger.info(f"Found {len(files)} files")
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    def download_all_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        output_file: str
    ):
        """
        전체 데이터를 한 번에 다운로드하여 저장
        
        Parameters
        ----------
        symbols : List[str]
            종목 심볼 리스트
        start_date : str
            시작 날짜
        end_date : str
            종료 날짜
        output_file : str
            출력 파일 경로 (parquet)
        """
        logger.info("Downloading all data",
                   n_symbols=len(symbols),
                   start_date=start_date,
                   end_date=end_date,
                   output_file=output_file)
        
        # 데이터 로드
        df = self.load_stocks_data(symbols, start_date, end_date, use_cache=True)
        
        if df.empty:
            logger.error("No data to save")
            return
        
        # Parquet 저장
        df.to_parquet(output_file, compression='snappy', index=False)
        
        file_size_mb = Path(output_file).stat().st_size / 1024 / 1024
        
        logger.info("Data saved successfully",
                   output_file=output_file,
                   file_size_mb=f"{file_size_mb:.2f} MB",
                   n_rows=len(df),
                   n_symbols=df['symbol'].nunique())


# 사용 예시
if __name__ == "__main__":
    # S&P 100 심볼 (예시)
    sp100_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'UNH', 'JNJ',
        'V', 'XOM', 'WMT', 'JPM', 'PG', 'MA', 'CVX', 'HD', 'LLY', 'ABBV',
        # ... (100개 전체)
    ]
    
    # 로더 초기화
    loader = PolygonFlatfilesLoader()
    
    # 데이터 다운로드
    loader.download_all_data(
        symbols=sp100_symbols,
        start_date="2016-03-01",
        end_date="2025-11-18",
        output_file="/tmp/sp100_data.parquet"
    )
    
    print("✅ Data download complete!")
