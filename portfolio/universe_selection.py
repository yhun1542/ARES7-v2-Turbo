"""
Universe Selection with Top-K and Sector Cap
유니버스 선택: 상위 K 채택 + 섹터 캡
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniverseSelector:
    """
    유니버스 선택 규칙
    
    1. 알파 신호 상위 K(60-80) 종목만 채택
    2. 섹터 캡 ≤25%
    3. HRP 75% + softmax(α) 25% 블렌딩
    """
    
    def __init__(
        self,
        top_k: int = 70,
        sector_cap: float = 0.25,
        hrp_weight: float = 0.75
    ):
        """
        Args:
            top_k: 선택할 상위 종목 수 (기본 70)
            sector_cap: 섹터별 최대 비중 (기본 0.25 = 25%)
            hrp_weight: HRP 가중치 (기본 0.75 = 75%)
        """
        self.top_k = top_k
        self.sector_cap = sector_cap
        self.hrp_weight = hrp_weight
        self.alpha_weight = 1 - hrp_weight
        
        logger.info(f"UniverseSelector initialized: top_k={top_k}, sector_cap={sector_cap}")
    
    def select_top_k(
        self,
        alpha_signals: pd.Series,
        min_signal: float = 0.0
    ) -> List[str]:
        """
        알파 신호 상위 K 종목 선택
        
        Args:
            alpha_signals: 종목별 알파 신호 (Series)
            min_signal: 최소 신호 임계값 (기본 0.0)
            
        Returns:
            선택된 종목 리스트
        """
        # 양수 신호만 필터링
        positive_signals = alpha_signals[alpha_signals > min_signal]
        
        # 상위 K 선택
        top_k_signals = positive_signals.nlargest(self.top_k)
        
        logger.info(f"Selected {len(top_k_signals)} stocks from {len(alpha_signals)} universe")
        
        return top_k_signals.index.tolist()
    
    def apply_sector_cap(
        self,
        weights: pd.Series,
        sector_map: Dict[str, str]
    ) -> pd.Series:
        """
        섹터 캡 적용
        
        Args:
            weights: 종목별 가중치 (Series)
            sector_map: 종목 → 섹터 매핑 (Dict)
            
        Returns:
            섹터 캡 적용된 가중치
        """
        # 섹터별 가중치 합계
        sector_weights = pd.Series(
            {sector: 0.0 for sector in set(sector_map.values())}
        )
        
        for ticker, weight in weights.items():
            if ticker in sector_map:
                sector = sector_map[ticker]
                sector_weights[sector] += weight
        
        # 섹터 캡 초과 확인
        over_cap_sectors = sector_weights[sector_weights > self.sector_cap]
        
        if len(over_cap_sectors) > 0:
            logger.warning(f"Sectors over cap: {over_cap_sectors.to_dict()}")
            
            # 섹터 캡 적용 (비례 축소)
            adjusted_weights = weights.copy()
            
            for sector in over_cap_sectors.index:
                sector_tickers = [
                    t for t, s in sector_map.items()
                    if s == sector and t in weights.index
                ]
                
                sector_total = weights[sector_tickers].sum()
                scale_factor = self.sector_cap / sector_total
                
                adjusted_weights[sector_tickers] *= scale_factor
            
            # 재정규화
            adjusted_weights /= adjusted_weights.sum()
            
            logger.info("Sector cap applied and weights renormalized")
            return adjusted_weights
        
        return weights
    
    def blend_weights(
        self,
        hrp_weights: pd.Series,
        alpha_signals: pd.Series
    ) -> pd.Series:
        """
        HRP 75% + softmax(α) 25% 블렌딩
        
        Args:
            hrp_weights: HRP 가중치
            alpha_signals: 알파 신호
            
        Returns:
            블렌딩된 가중치
        """
        # softmax(α) 계산
        exp_alpha = np.exp(alpha_signals - alpha_signals.max())  # numerical stability
        softmax_weights = exp_alpha / exp_alpha.sum()
        
        # 블렌딩
        blended = (
            self.hrp_weight * hrp_weights +
            self.alpha_weight * softmax_weights
        )
        
        # 정규화
        blended /= blended.sum()
        
        logger.info(f"Blended weights: HRP {self.hrp_weight:.0%} + Alpha {self.alpha_weight:.0%}")
        
        return blended
    
    def construct_portfolio(
        self,
        alpha_signals: pd.Series,
        hrp_weights: pd.Series,
        sector_map: Dict[str, str]
    ) -> pd.Series:
        """
        포트폴리오 구성 (전체 파이프라인)
        
        Args:
            alpha_signals: 알파 신호
            hrp_weights: HRP 가중치
            sector_map: 섹터 매핑
            
        Returns:
            최종 포트폴리오 가중치
        """
        # 1. 상위 K 선택
        selected_tickers = self.select_top_k(alpha_signals)
        
        # 2. 선택된 종목만 필터링
        alpha_filtered = alpha_signals[selected_tickers]
        hrp_filtered = hrp_weights[selected_tickers]
        
        # 3. HRP + Alpha 블렌딩
        blended_weights = self.blend_weights(hrp_filtered, alpha_filtered)
        
        # 4. 섹터 캡 적용
        final_weights = self.apply_sector_cap(blended_weights, sector_map)
        
        logger.info(f"Portfolio constructed: {len(final_weights)} stocks")
        
        return final_weights


# 테스트 코드
if __name__ == "__main__":
    print("=" * 80)
    print("Universe Selection Test")
    print("=" * 80)
    print()
    
    # 테스트 데이터
    np.random.seed(42)
    
    # 100종목 유니버스
    tickers = [f"STOCK_{i:03d}" for i in range(100)]
    
    # 알파 신호 (정규분포)
    alpha_signals = pd.Series(
        np.random.randn(100) * 0.1 + 0.05,
        index=tickers
    )
    
    # HRP 가중치 (균등)
    hrp_weights = pd.Series(1/100, index=tickers)
    
    # 섹터 매핑 (10개 섹터)
    sectors = ['Tech', 'Finance', 'Healthcare', 'Consumer', 'Energy',
               'Industrial', 'Materials', 'Utilities', 'RealEstate', 'Telecom']
    sector_map = {ticker: sectors[i % 10] for i, ticker in enumerate(tickers)}
    
    # 유니버스 선택
    selector = UniverseSelector(top_k=70, sector_cap=0.25, hrp_weight=0.75)
    portfolio = selector.construct_portfolio(alpha_signals, hrp_weights, sector_map)
    
    # 결과 출력
    print(f"Selected stocks: {len(portfolio)}")
    print(f"Total weight: {portfolio.sum():.4f}")
    print()
    
    # 섹터별 비중
    sector_weights = pd.Series(
        {sector: 0.0 for sector in sectors}
    )
    for ticker, weight in portfolio.items():
        sector = sector_map[ticker]
        sector_weights[sector] += weight
    
    print("Sector Weights:")
    print("-" * 80)
    for sector, weight in sector_weights.sort_values(ascending=False).items():
        print(f"  {sector:15s}: {weight:6.2%}")
    print()
    
    # 섹터 캡 확인
    max_sector_weight = sector_weights.max()
    print(f"Max sector weight: {max_sector_weight:.2%} (cap: {selector.sector_cap:.0%})")
    
    if max_sector_weight <= selector.sector_cap:
        print("✅ Sector cap satisfied")
    else:
        print("❌ Sector cap violated")
    
    print()
    print("=" * 80)
