"""
Sideways Regime Alpha Module
횡보장 전용 알파: 레인지 트레이딩, 페어/스프레드, 미세 역추세
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SidewaysRegimeAlpha:
    """
    횡보장 전용 알파 생성기
    
    전략:
    1. 레인지 트레이딩: z-score 기반 평균 회귀
    2. RSI 역추세: 과매수/과매도 반전
    3. 페어/스프레드: 섹터 내 상대 강도
    """
    
    def __init__(
        self,
        zscore_window: int = 20,
        zscore_threshold: float = 1.5,
        rsi_window: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70
    ):
        self.zscore_window = zscore_window
        self.zscore_threshold = zscore_threshold
        self.rsi_window = rsi_window
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        
        logger.info("SidewaysRegimeAlpha initialized")
    
    def calculate_zscore(self, prices: pd.Series) -> float:
        """
        Z-score 계산 (평균 회귀 신호)
        
        Returns:
            z-score (양수: 과매수, 음수: 과매도)
        """
        if len(prices) < self.zscore_window:
            return 0.0
        
        window = prices.iloc[-self.zscore_window:]
        mean = window.mean()
        std = window.std()
        
        if std == 0:
            return 0.0
        
        current_price = prices.iloc[-1]
        zscore = (current_price - mean) / std
        
        return zscore
    
    def calculate_rsi(self, prices: pd.Series) -> float:
        """
        RSI 계산
        
        Returns:
            RSI (0-100)
        """
        if len(prices) < self.rsi_window + 1:
            return 50.0  # 중립
        
        # 가격 변화
        delta = prices.diff()
        
        # 상승/하락 분리
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        # 평균 계산
        avg_gain = gain.rolling(window=self.rsi_window).mean().iloc[-1]
        avg_loss = loss.rolling(window=self.rsi_window).mean().iloc[-1]
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_range_trading_signal(self, prices: pd.Series) -> float:
        """
        레인지 트레이딩 신호 (z-score 기반)
        
        Returns:
            신호 강도 (-1 to 1)
        """
        zscore = self.calculate_zscore(prices)
        
        # 과매수 → 매도 신호 (음수)
        if zscore > self.zscore_threshold:
            signal = -min(1.0, (zscore - self.zscore_threshold) / self.zscore_threshold)
        
        # 과매도 → 매수 신호 (양수)
        elif zscore < -self.zscore_threshold:
            signal = min(1.0, (-zscore - self.zscore_threshold) / self.zscore_threshold)
        
        else:
            signal = 0.0
        
        return signal
    
    def generate_rsi_reversal_signal(self, prices: pd.Series) -> float:
        """
        RSI 역추세 신호
        
        Returns:
            신호 강도 (-1 to 1)
        """
        rsi = self.calculate_rsi(prices)
        
        # 과매도 → 매수 신호
        if rsi < self.rsi_oversold:
            signal = min(1.0, (self.rsi_oversold - rsi) / self.rsi_oversold)
        
        # 과매수 → 매도 신호
        elif rsi > self.rsi_overbought:
            signal = -min(1.0, (rsi - self.rsi_overbought) / (100 - self.rsi_overbought))
        
        else:
            signal = 0.0
        
        return signal
    
    def generate_pair_spread_signal(
        self,
        prices: pd.Series,
        sector_prices: pd.DataFrame
    ) -> float:
        """
        페어/스프레드 신호 (섹터 내 상대 강도)
        
        Args:
            prices: 개별 종목 가격
            sector_prices: 같은 섹터 종목들의 가격 (DataFrame)
            
        Returns:
            신호 강도 (-1 to 1)
        """
        if len(sector_prices.columns) < 2:
            return 0.0
        
        # 섹터 평균 대비 상대 강도
        sector_mean = sector_prices.mean(axis=1)
        
        if len(prices) < 20 or len(sector_mean) < 20:
            return 0.0
        
        # 상대 성과 (최근 20일)
        stock_return = (prices.iloc[-1] / prices.iloc[-20]) - 1
        sector_return = (sector_mean.iloc[-1] / sector_mean.iloc[-20]) - 1
        
        relative_performance = stock_return - sector_return
        
        # 평균 회귀 신호 (상대적으로 강하면 매도, 약하면 매수)
        signal = -np.tanh(relative_performance * 10)  # -1 to 1
        
        return signal
    
    def generate_combined_signal(
        self,
        prices: pd.Series,
        sector_prices: pd.DataFrame = None
    ) -> Dict:
        """
        통합 신호 생성
        
        Returns:
            신호 딕셔너리
        """
        # 개별 신호
        range_signal = self.generate_range_trading_signal(prices)
        rsi_signal = self.generate_rsi_reversal_signal(prices)
        
        if sector_prices is not None:
            pair_signal = self.generate_pair_spread_signal(prices, sector_prices)
        else:
            pair_signal = 0.0
        
        # 가중 평균 (동일 가중)
        combined = (range_signal + rsi_signal + pair_signal) / 3
        
        return {
            'range_trading': range_signal,
            'rsi_reversal': rsi_signal,
            'pair_spread': pair_signal,
            'combined': combined
        }


# 테스트 코드
if __name__ == "__main__":
    print("=" * 80)
    print("Sideways Regime Alpha Test")
    print("=" * 80)
    print()
    
    # 테스트 데이터 (횡보장 시뮬레이션)
    np.random.seed(42)
    n_days = 100
    
    # 횡보 가격 (평균 회귀)
    prices = pd.Series(
        100 + 10 * np.sin(np.linspace(0, 4 * np.pi, n_days)) + np.random.randn(n_days) * 2
    )
    
    # 섹터 가격 (3종목)
    sector_prices = pd.DataFrame({
        'STOCK_A': prices + np.random.randn(n_days) * 1,
        'STOCK_B': prices + np.random.randn(n_days) * 1,
        'STOCK_C': prices + np.random.randn(n_days) * 1
    })
    
    # 알파 생성
    alpha_gen = SidewaysRegimeAlpha()
    signals = alpha_gen.generate_combined_signal(prices, sector_prices)
    
    # 결과 출력
    print("Sideways Regime Signals:")
    print("-" * 80)
    print(f"  Range Trading:  {signals['range_trading']:7.4f}")
    print(f"  RSI Reversal:   {signals['rsi_reversal']:7.4f}")
    print(f"  Pair Spread:    {signals['pair_spread']:7.4f}")
    print(f"  Combined:       {signals['combined']:7.4f}")
    print()
    
    # 시계열 신호 테스트
    print("Time Series Signals (Last 10 days):")
    print("-" * 80)
    
    for i in range(-10, 0):
        window_prices = prices.iloc[:n_days+i]
        window_sector = sector_prices.iloc[:n_days+i]
        
        sig = alpha_gen.generate_combined_signal(window_prices, window_sector)
        
        print(f"  Day {n_days+i:3d}: Combined={sig['combined']:7.4f}, "
              f"Range={sig['range_trading']:7.4f}, "
              f"RSI={sig['rsi_reversal']:7.4f}")
    
    print()
    print("=" * 80)
    print("✅ Sideways Regime Alpha Test Complete")
    print("=" * 80)
