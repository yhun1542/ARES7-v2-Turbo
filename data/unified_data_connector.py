"""
ARES Unified Data Connector
============================
5개 데이터 소스를 통합하여 알파/베타 생성에 활용

Data Sources:
1. Polygon (Massive.com) - 가격, 기술 지표, 펀더멘탈
2. Nasdaq (Sharadar) - SF1 펀더멘탈 데이터
3. FRED - 매크로 경제 지표
4. Alpha Vantage - 고급 기술 지표, 뉴스 감성
5. SEC - 공시, 내부자 거래
"""

import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class UnifiedDataConnector:
    """통합 데이터 커넥터"""
    
    def __init__(self):
        # API Keys
        self.polygon_api_key = os.getenv("POLYGON_API_KEY", "w7KprL4_lK7uutSH0dYGARkucXHOFXCN")
        self.sharadar_api_key = os.getenv("SHARADAR_API_KEY", "H6zH4Q2CDr9uTFk9koqJ")
        self.fred_api_key = os.getenv("FRED_API_KEY", "b4a5371d46459ba15138393980de28d5")
        self.alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_KEY", "WA6OEWIF23A4LVGN")
        self.sec_api_key = os.getenv("SEC_API_KEY", "c2c08a95c67793b5a8bbba1e51611ed466900124e70c0615badefea2c6d429f9")
        
        # Base URLs
        self.polygon_base = "https://api.polygon.io"
        self.sharadar_base = "https://data.nasdaq.com/api/v3/datatables"
        self.fred_base = "https://api.stlouisfed.org/fred"
        self.alpha_vantage_base = "https://www.alphavantage.co/query"
        self.sec_base = "https://api.sec-api.io"
        
        # Cache
        self.cache = {}
        
    # ============================================================================
    # 1. POLYGON - 가격 및 시장 데이터
    # ============================================================================
    
    async def get_polygon_aggregates(
        self, 
        symbol: str,
        start_date: str,
        end_date: str,
        timespan: str = "day"
    ) -> pd.DataFrame:
        """
        Polygon Aggregates (OHLCV) 데이터
        
        Args:
            symbol: 종목 코드 (e.g., "AAPL")
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)
            timespan: 시간 단위 (minute, hour, day, week, month)
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        cache_key = f"polygon_agg_{symbol}_{start_date}_{end_date}_{timespan}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        url = f"{self.polygon_base}/v2/aggs/ticker/{symbol}/range/1/{timespan}/{start_date}/{end_date}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.polygon_api_key
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if "results" in data:
                        df = pd.DataFrame(data["results"])
                        df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
                        df = df.rename(columns={
                            "o": "open",
                            "h": "high",
                            "l": "low",
                            "c": "close",
                            "v": "volume"
                        })
                        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
                        df = df.set_index("timestamp")
                        
                        self.cache[cache_key] = df
                        return df
                else:
                    logger.error(f"Polygon API error: {response.status}")
                    return pd.DataFrame()
    
    async def get_polygon_technical_indicators(
        self,
        symbol: str,
        indicator: str,
        start_date: str,
        end_date: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Polygon 기술 지표
        
        Args:
            symbol: 종목 코드
            indicator: 지표 이름 (sma, ema, rsi, macd)
            start_date: 시작일
            end_date: 종료일
            **kwargs: 지표별 파라미터 (e.g., window=20 for SMA)
        
        Returns:
            DataFrame with indicator values
        """
        url = f"{self.polygon_base}/v1/indicators/{indicator}/{symbol}"
        params = {
            "timestamp.gte": start_date,
            "timestamp.lte": end_date,
            "apiKey": self.polygon_api_key,
            **kwargs
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if "results" in data:
                        df = pd.DataFrame(data["results"])
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df = df.set_index("timestamp")
                        return df
                else:
                    logger.error(f"Polygon Technical Indicators error: {response.status}")
                    return pd.DataFrame()
    
    async def get_polygon_financials(
        self,
        symbol: str,
        filing_type: str = "10-K"
    ) -> pd.DataFrame:
        """
        Polygon 재무제표 데이터
        
        Args:
            symbol: 종목 코드
            filing_type: 공시 유형 (10-K, 10-Q)
        
        Returns:
            DataFrame with financial statements
        """
        url = f"{self.polygon_base}/vX/reference/financials"
        params = {
            "ticker": symbol,
            "filing_date.gte": (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d"),
            "filing_type": filing_type,
            "apiKey": self.polygon_api_key
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if "results" in data:
                        return pd.DataFrame(data["results"])
                else:
                    logger.error(f"Polygon Financials error: {response.status}")
                    return pd.DataFrame()
    
    # ============================================================================
    # 2. NASDAQ (SHARADAR) - SF1 펀더멘탈 데이터
    # ============================================================================
    
    async def get_sharadar_sf1(
        self,
        ticker: str,
        dimension: str = "MRQ"  # MRQ (Quarterly), MRY (Annual), ARQ, ART
    ) -> pd.DataFrame:
        """
        Sharadar SF1 Core Fundamentals
        
        Args:
            ticker: 종목 코드
            dimension: 데이터 차원 (MRQ, MRY, ARQ, ART)
        
        Returns:
            DataFrame with fundamental metrics
        """
        cache_key = f"sharadar_sf1_{ticker}_{dimension}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        url = f"{self.sharadar_base}/SHARADAR/SF1"
        params = {
            "ticker": ticker,
            "dimension": dimension,
            "api_key": self.sharadar_api_key
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if "datatable" in data and "data" in data["datatable"]:
                        columns = [col["name"] for col in data["datatable"]["columns"]]
                        df = pd.DataFrame(data["datatable"]["data"], columns=columns)
                        df["datekey"] = pd.to_datetime(df["datekey"])
                        df = df.set_index("datekey")
                        
                        self.cache[cache_key] = df
                        return df
                else:
                    logger.error(f"Sharadar SF1 error: {response.status}")
                    return pd.DataFrame()
    
    def calculate_quality_score(self, sf1_data: pd.DataFrame) -> pd.Series:
        """
        Quality Score 계산
        
        Quality Score = 0.4 * ROE + 0.3 * ROA + 0.3 * ROIC
        
        Args:
            sf1_data: Sharadar SF1 DataFrame
        
        Returns:
            Series with quality scores
        """
        roe = sf1_data.get("roe", 0)
        roa = sf1_data.get("roa", 0)
        roic = sf1_data.get("roic", 0)
        
        quality_score = 0.4 * roe + 0.3 * roa + 0.3 * roic
        return quality_score
    
    def calculate_financial_health(self, sf1_data: pd.DataFrame) -> pd.Series:
        """
        Financial Health Score 계산
        
        Args:
            sf1_data: Sharadar SF1 DataFrame
        
        Returns:
            Series with financial health scores
        """
        current_ratio = sf1_data.get("currentratio", 1)
        debt_to_equity = sf1_data.get("de", 1)
        
        # Avoid division by zero
        debt_to_equity = debt_to_equity.replace(0, 0.01)
        
        financial_health = 0.5 * current_ratio + 0.5 * (1 / debt_to_equity)
        return financial_health
    
    # ============================================================================
    # 3. FRED - 매크로 경제 지표
    # ============================================================================
    
    async def get_fred_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        FRED 경제 지표 데이터
        
        Args:
            series_id: FRED 시리즈 ID (e.g., "VIXCLS", "DGS10", "T10Y2Y")
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)
        
        Returns:
            DataFrame with economic data
        """
        cache_key = f"fred_{series_id}_{start_date}_{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        url = f"{self.fred_base}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.fred_api_key,
            "file_type": "json"
        }
        
        if start_date:
            params["observation_start"] = start_date
        if end_date:
            params["observation_end"] = end_date
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if "observations" in data:
                        df = pd.DataFrame(data["observations"])
                        df["date"] = pd.to_datetime(df["date"])
                        df["value"] = pd.to_numeric(df["value"], errors="coerce")
                        df = df[["date", "value"]].set_index("date")
                        df = df.rename(columns={"value": series_id})
                        
                        self.cache[cache_key] = df
                        return df
                else:
                    logger.error(f"FRED API error: {response.status}")
                    return pd.DataFrame()
    
    async def get_macro_indicators(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        주요 매크로 지표 일괄 조회
        
        Returns:
            DataFrame with VIX, Yield Curve, Unemployment, etc.
        """
        series_ids = [
            "VIXCLS",      # VIX
            "T10Y2Y",      # 10Y-2Y Yield Curve
            "DGS10",       # 10-Year Treasury
            "DGS2",        # 2-Year Treasury
            "UNRATE",      # Unemployment Rate
            "DTWEXBGS",    # Dollar Index
            "BAMLH0A0HYM2" # High Yield Spread
        ]
        
        tasks = [self.get_fred_series(sid, start_date, end_date) for sid in series_ids]
        results = await asyncio.gather(*tasks)
        
        # Merge all series
        df = pd.concat(results, axis=1)
        df = df.ffill()  # Forward fill missing values
        return df
    
    def detect_regime(self, macro_data: pd.DataFrame) -> pd.Series:
        """
        레짐 감지
        
        BULL: VIX < 20, Yield Curve > 0, Unemployment < 5%
        BEAR: VIX > 30, Yield Curve < 0, Unemployment > 6%
        HIGH_VOL: VIX > 25
        NEUTRAL: Otherwise
        
        Args:
            macro_data: DataFrame with VIX, T10Y2Y, UNRATE
        
        Returns:
            Series with regime labels
        """
        vix = macro_data.get("VIXCLS", 20)
        yield_curve = macro_data.get("T10Y2Y", 0)
        unemployment = macro_data.get("UNRATE", 5)
        
        regime = pd.Series("NEUTRAL", index=macro_data.index)
        
        regime[(vix > 30) | (yield_curve < 0)] = "BEAR"
        regime[vix > 25] = "HIGH_VOL"
        regime[(vix < 20) & (yield_curve > 0) & (unemployment < 5)] = "BULL"
        
        return regime
    
    # ============================================================================
    # 4. ALPHA VANTAGE - 기술 지표 및 뉴스 감성
    # ============================================================================
    
    async def get_alpha_vantage_indicator(
        self,
        symbol: str,
        function: str,
        interval: str = "daily",
        **kwargs
    ) -> pd.DataFrame:
        """
        Alpha Vantage 기술 지표
        
        Args:
            symbol: 종목 코드
            function: 지표 함수 (RSI, MACD, BBANDS, STOCH, ADX)
            interval: 시간 간격 (daily, weekly, monthly)
            **kwargs: 지표별 파라미터
        
        Returns:
            DataFrame with indicator values
        """
        url = self.alpha_vantage_base
        params = {
            "function": function,
            "symbol": symbol,
            "interval": interval,
            "apikey": self.alpha_vantage_api_key,
            **kwargs
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract technical analysis data
                    key = [k for k in data.keys() if "Technical" in k]
                    if key:
                        df = pd.DataFrame.from_dict(data[key[0]], orient="index")
                        df.index = pd.to_datetime(df.index)
                        df = df.apply(pd.to_numeric, errors="coerce")
                        return df
                else:
                    logger.error(f"Alpha Vantage error: {response.status}")
                    return pd.DataFrame()
    
    async def get_news_sentiment(
        self,
        tickers: List[str],
        time_from: Optional[str] = None,
        time_to: Optional[str] = None
    ) -> pd.DataFrame:
        """
        뉴스 감성 분석
        
        Args:
            tickers: 종목 코드 리스트
            time_from: 시작 시간 (YYYYMMDDTHHMM)
            time_to: 종료 시간 (YYYYMMDDTHHMM)
        
        Returns:
            DataFrame with news sentiment scores
        """
        url = self.alpha_vantage_base
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ",".join(tickers),
            "apikey": self.alpha_vantage_api_key
        }
        
        if time_from:
            params["time_from"] = time_from
        if time_to:
            params["time_to"] = time_to
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if "feed" in data:
                        return pd.DataFrame(data["feed"])
                else:
                    logger.error(f"News Sentiment error: {response.status}")
                    return pd.DataFrame()
    
    # ============================================================================
    # 5. SEC - 공시 및 내부자 거래
    # ============================================================================
    
    async def get_insider_transactions(
        self,
        ticker: str,
        start_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        내부자 거래 데이터 (Form 3/4/5)
        
        Args:
            ticker: 종목 코드
            start_date: 시작일 (YYYY-MM-DD)
        
        Returns:
            DataFrame with insider transactions
        """
        url = f"{self.sec_base}/insider-trading"
        params = {
            "ticker": ticker,
            "token": self.sec_api_key
        }
        
        if start_date:
            params["from"] = start_date
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if "data" in data:
                        return pd.DataFrame(data["data"])
                else:
                    logger.error(f"SEC Insider Trading error: {response.status}")
                    return pd.DataFrame()
    
    def calculate_insider_signal(self, insider_data: pd.DataFrame) -> float:
        """
        내부자 거래 시그널 계산
        
        Args:
            insider_data: DataFrame with insider transactions
        
        Returns:
            Insider signal score (-1 to 1)
        """
        if insider_data.empty:
            return 0.0
        
        buys = insider_data[insider_data["transactionType"] == "Buy"]["shares"].sum()
        sells = insider_data[insider_data["transactionType"] == "Sell"]["shares"].sum()
        
        total = buys + sells
        if total == 0:
            return 0.0
        
        signal = (buys - sells) / total
        return signal
    
    # ============================================================================
    # ALPHA/BETA 생성 파이프라인
    # ============================================================================
    
    async def generate_alpha_signals(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        통합 알파 시그널 생성
        
        Returns:
            DataFrame with columns:
            - quality_alpha
            - momentum_alpha
            - technical_alpha
            - value_alpha
            - sentiment_alpha
            - combined_alpha
        """
        # 1. Quality Alpha (Sharadar)
        sf1_data = await self.get_sharadar_sf1(symbol)
        quality_score = self.calculate_quality_score(sf1_data)
        financial_health = self.calculate_financial_health(sf1_data)
        quality_alpha = 0.6 * quality_score + 0.4 * financial_health
        
        # 2. Momentum Alpha (Polygon)
        price_data = await self.get_polygon_aggregates(symbol, start_date, end_date)
        price_momentum = price_data["close"].pct_change(63)  # 3-month momentum
        volume_momentum = price_data["volume"].pct_change(63)
        momentum_alpha = 0.7 * price_momentum + 0.3 * volume_momentum
        
        # 3. Technical Alpha (Polygon + Alpha Vantage)
        rsi_data = await self.get_polygon_technical_indicators(symbol, "rsi", start_date, end_date, window=14)
        macd_data = await self.get_polygon_technical_indicators(symbol, "macd", start_date, end_date)
        
        if not rsi_data.empty:
            rsi_signal = (rsi_data["value"] - 50) / 50
        else:
            rsi_signal = pd.Series(0, index=price_data.index)
        
        if not macd_data.empty:
            macd_signal = macd_data["value"] / price_data["close"]
        else:
            macd_signal = pd.Series(0, index=price_data.index)
        
        technical_alpha = 0.5 * rsi_signal + 0.5 * macd_signal
        
        # 4. Value Alpha (Sharadar)
        pe_ratio = sf1_data.get("pe", 20)
        pb_ratio = sf1_data.get("pb", 3)
        ps_ratio = sf1_data.get("ps", 2)
        
        pe_score = 1 / (pe_ratio + 1)
        pb_score = 1 / (pb_ratio + 1)
        ps_score = 1 / (ps_ratio + 1)
        
        value_alpha = 0.4 * pe_score + 0.3 * pb_score + 0.3 * ps_score
        
        # 5. Sentiment Alpha (Alpha Vantage + SEC)
        insider_data = await self.get_insider_transactions(symbol)
        insider_signal = self.calculate_insider_signal(insider_data)
        sentiment_alpha = pd.Series(insider_signal, index=price_data.index)
        
        # Combine all alphas
        alpha_df = pd.DataFrame({
            "quality_alpha": quality_alpha,
            "momentum_alpha": momentum_alpha,
            "technical_alpha": technical_alpha,
            "value_alpha": value_alpha,
            "sentiment_alpha": sentiment_alpha
        }, index=price_data.index)
        
        # QM Overlay (60% Quality + 40% Momentum)
        alpha_df["combined_alpha"] = 0.6 * alpha_df["quality_alpha"] + 0.4 * alpha_df["momentum_alpha"]
        
        return alpha_df
    
    async def calculate_beta_exposure(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        market_symbol: str = "SPY"
    ) -> pd.DataFrame:
        """
        베타 노출 계산
        
        Returns:
            DataFrame with columns:
            - market_beta
            - macro_beta
        """
        # Get stock and market returns
        stock_data = await self.get_polygon_aggregates(symbol, start_date, end_date)
        market_data = await self.get_polygon_aggregates(market_symbol, start_date, end_date)
        
        stock_returns = stock_data["close"].pct_change()
        market_returns = market_data["close"].pct_change()
        
        # Market Beta
        covariance = stock_returns.cov(market_returns)
        market_variance = market_returns.var()
        market_beta = covariance / market_variance if market_variance > 0 else 1.0
        
        # Macro Beta (FRED)
        macro_data = await self.get_macro_indicators(start_date, end_date)
        
        if not macro_data.empty:
            rate_changes = macro_data["DGS10"].pct_change()
            dollar_changes = macro_data["DTWEXBGS"].pct_change()
            vix_changes = macro_data["VIXCLS"].pct_change()
            
            rate_beta = stock_returns.corr(rate_changes)
            dollar_beta = stock_returns.corr(dollar_changes)
            vix_beta = stock_returns.corr(vix_changes)
            
            macro_beta = 0.4 * rate_beta + 0.3 * dollar_beta + 0.3 * vix_beta
        else:
            macro_beta = 0.0
        
        beta_df = pd.DataFrame({
            "market_beta": market_beta,
            "macro_beta": macro_beta
        }, index=stock_data.index)
        
        return beta_df
    
    async def generate_regime_adjusted_signals(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        레짐 조정 시그널 생성
        
        Returns:
            DataFrame with regime-adjusted alpha signals
        """
        # Generate alpha signals
        alpha_df = await self.generate_alpha_signals(symbol, start_date, end_date)
        
        # Get macro data and detect regime
        macro_data = await self.get_macro_indicators(start_date, end_date)
        regime = self.detect_regime(macro_data)
        
        # Regime-adjusted alpha
        regime_adjusted_alpha = alpha_df["combined_alpha"].copy()
        
        regime_adjusted_alpha[regime == "BULL"] *= 1.2
        regime_adjusted_alpha[regime == "BEAR"] *= 0.5
        regime_adjusted_alpha[regime == "HIGH_VOL"] *= 0.7
        
        alpha_df["regime"] = regime
        alpha_df["regime_adjusted_alpha"] = regime_adjusted_alpha
        
        return alpha_df


# ============================================================================
# 사용 예제
# ============================================================================

async def main():
    """사용 예제"""
    connector = UnifiedDataConnector()
    
    symbol = "AAPL"
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    
    # 1. 알파 시그널 생성
    alpha_signals = await connector.generate_alpha_signals(symbol, start_date, end_date)
    print("Alpha Signals:")
    print(alpha_signals.tail())
    
    # 2. 베타 노출 계산
    beta_exposure = await connector.calculate_beta_exposure(symbol, start_date, end_date)
    print("\nBeta Exposure:")
    print(beta_exposure.tail())
    
    # 3. 레짐 조정 시그널
    regime_signals = await connector.generate_regime_adjusted_signals(symbol, start_date, end_date)
    print("\nRegime-Adjusted Signals:")
    print(regime_signals.tail())


if __name__ == "__main__":
    asyncio.run(main())
