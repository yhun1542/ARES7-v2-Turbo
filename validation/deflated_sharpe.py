"""
Deflated Sharpe Ratio (DSR) and Superior Predictive Ability (SPA) Test
탐색 편향 보정을 위한 통계적 검증
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeflatedSharpeRatio:
    """
    Deflated Sharpe Ratio (DSR) 계산
    
    다중 탐색으로 인한 과추정을 보정하여 실제 Sharpe Ratio의 통계적 유의성을 평가합니다.
    
    Reference:
    Bailey, D. H., & López de Prado, M. (2014). 
    "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
    Journal of Portfolio Management, 40(5), 94-107.
    """
    
    def __init__(self, n_trials: int = 100, skewness: float = 0.0, kurtosis: float = 3.0):
        """
        Args:
            n_trials: 탐색한 전략 개수 (백테스트 횟수)
            skewness: 수익률 분포의 왜도
            kurtosis: 수익률 분포의 첨도
        """
        self.n_trials = n_trials
        self.skewness = skewness
        self.kurtosis = kurtosis
    
    def compute_dsr(
        self,
        sharpe_ratio: float,
        n_observations: int,
        confidence_level: float = 0.95
    ) -> Tuple[float, float, bool]:
        """
        Deflated Sharpe Ratio 계산
        
        Args:
            sharpe_ratio: 관측된 Sharpe Ratio
            n_observations: 관측 개수 (일수)
            confidence_level: 신뢰 수준
            
        Returns:
            (DSR, p-value, is_significant)
        """
        # 1. 비정규성 조정
        adjusted_sr = self._adjust_for_non_normality(sharpe_ratio, n_observations)
        
        # 2. 다중 탐색 조정
        expected_max_sr = self._expected_maximum_sr(n_observations)
        std_max_sr = self._std_maximum_sr(n_observations)
        
        # 3. Deflated Sharpe Ratio
        if std_max_sr > 0:
            dsr = (adjusted_sr - expected_max_sr) / std_max_sr
        else:
            dsr = 0.0
        
        # 4. p-value 계산
        p_value = 1 - stats.norm.cdf(dsr)
        
        # 5. 통계적 유의성
        critical_value = stats.norm.ppf(confidence_level)
        is_significant = dsr > critical_value
        
        logger.info(f"Deflated Sharpe Ratio:")
        logger.info(f"  Observed SR: {sharpe_ratio:.4f}")
        logger.info(f"  Adjusted SR: {adjusted_sr:.4f}")
        logger.info(f"  Expected Max SR: {expected_max_sr:.4f}")
        logger.info(f"  Std Max SR: {std_max_sr:.4f}")
        logger.info(f"  DSR: {dsr:.4f}")
        logger.info(f"  p-value: {p_value:.4f}")
        logger.info(f"  Significant ({confidence_level:.0%}): {is_significant}")
        
        return dsr, p_value, is_significant
    
    def _adjust_for_non_normality(self, sr: float, n: int) -> float:
        """비정규성 조정"""
        # 왜도와 첨도를 고려한 조정
        adjustment = (
            (1 - self.skewness * sr + (self.kurtosis - 1) / 4 * sr ** 2) ** 0.5
        )
        return sr / adjustment
    
    def _expected_maximum_sr(self, n: int) -> float:
        """다중 탐색 시 기대되는 최대 Sharpe Ratio"""
        # Euler-Mascheroni constant
        gamma = 0.5772156649
        
        # Expected maximum of N standard normal variables
        return (1 - gamma) * stats.norm.ppf(1 - 1 / self.n_trials) + gamma * stats.norm.ppf(1 - 1 / (self.n_trials * np.e))
    
    def _std_maximum_sr(self, n: int) -> float:
        """최대 Sharpe Ratio의 표준편차"""
        return (1 / (2 * np.log(self.n_trials))) ** 0.5


class SPATest:
    """
    Superior Predictive Ability (SPA) Test
    
    벤치마크 대비 전략의 우월성을 통계적으로 검증합니다.
    
    Reference:
    Hansen, P. R. (2005). 
    "A Test for Superior Predictive Ability." 
    Journal of Business & Economic Statistics, 23(4), 365-380.
    """
    
    def __init__(self, n_bootstrap: int = 10000):
        """
        Args:
            n_bootstrap: 부트스트랩 반복 횟수
        """
        self.n_bootstrap = n_bootstrap
    
    def test(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        confidence_level: float = 0.95
    ) -> Tuple[float, float, bool]:
        """
        SPA 테스트 수행
        
        Args:
            strategy_returns: 전략 수익률
            benchmark_returns: 벤치마크 수익률
            confidence_level: 신뢰 수준
            
        Returns:
            (test_statistic, p_value, is_superior)
        """
        # 1. 초과 수익률
        excess_returns = strategy_returns - benchmark_returns
        
        # 2. 테스트 통계량
        test_stat = np.mean(excess_returns) / (np.std(excess_returns) / np.sqrt(len(excess_returns)))
        
        # 3. 부트스트랩 p-value
        p_value = self._bootstrap_p_value(excess_returns)
        
        # 4. 우월성 판단
        is_superior = p_value < (1 - confidence_level)
        
        logger.info(f"SPA Test:")
        logger.info(f"  Test Statistic: {test_stat:.4f}")
        logger.info(f"  p-value: {p_value:.4f}")
        logger.info(f"  Superior ({confidence_level:.0%}): {is_superior}")
        
        return test_stat, p_value, is_superior
    
    def _bootstrap_p_value(self, excess_returns: pd.Series) -> float:
        """부트스트랩으로 p-value 계산"""
        n = len(excess_returns)
        mean_excess = np.mean(excess_returns)
        
        # 부트스트랩 샘플링
        bootstrap_means = []
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(excess_returns, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        # p-value: 부트스트랩 평균이 관측 평균보다 큰 비율
        p_value = np.mean(np.array(bootstrap_means) >= mean_excess)
        
        return p_value


def validate_strategy_significance(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    n_trials: int = 100,
    confidence_level: float = 0.95
) -> Dict:
    """
    전략의 통계적 유의성 종합 검증
    
    Args:
        returns: 전략 수익률
        benchmark_returns: 벤치마크 수익률
        n_trials: 탐색한 전략 개수
        confidence_level: 신뢰 수준
        
    Returns:
        검증 결과 딕셔너리
    """
    # 1. Sharpe Ratio 계산
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    # 2. 왜도 및 첨도
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns) + 3  # excess kurtosis -> kurtosis
    
    # 3. Deflated Sharpe Ratio
    dsr_calculator = DeflatedSharpeRatio(
        n_trials=n_trials,
        skewness=skewness,
        kurtosis=kurtosis
    )
    dsr, dsr_pvalue, dsr_significant = dsr_calculator.compute_dsr(
        sharpe_ratio=sharpe_ratio,
        n_observations=len(returns),
        confidence_level=confidence_level
    )
    
    # 4. SPA Test
    spa_tester = SPATest()
    spa_stat, spa_pvalue, spa_superior = spa_tester.test(
        strategy_returns=returns,
        benchmark_returns=benchmark_returns,
        confidence_level=confidence_level
    )
    
    # 5. 종합 결과
    result = {
        'sharpe_ratio': sharpe_ratio,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'n_trials': n_trials,
        'n_observations': len(returns),
        'dsr': dsr,
        'dsr_pvalue': dsr_pvalue,
        'dsr_significant': dsr_significant,
        'spa_statistic': spa_stat,
        'spa_pvalue': spa_pvalue,
        'spa_superior': spa_superior,
        'overall_pass': dsr_significant and spa_superior
    }
    
    return result


# 테스트 코드
if __name__ == "__main__":
    print("=" * 60)
    print("Deflated Sharpe Ratio & SPA Test")
    print("=" * 60)
    print()
    
    # 테스트 데이터 생성
    np.random.seed(42)
    n_days = 500
    
    # 전략 수익률 (Sharpe 2.5)
    strategy_returns = pd.Series(np.random.randn(n_days) * 0.01 + 0.0005)
    
    # 벤치마크 수익률 (Sharpe 1.0)
    benchmark_returns = pd.Series(np.random.randn(n_days) * 0.01 + 0.0002)
    
    # 검증
    result = validate_strategy_significance(
        returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        n_trials=100,
        confidence_level=0.95
    )
    
    print()
    print("=" * 60)
    print("Validation Result:")
    print("=" * 60)
    for key, value in result.items():
        print(f"{key}: {value}")
    
    print()
    if result['overall_pass']:
        print("✅ Strategy is statistically significant and superior to benchmark!")
    else:
        print("❌ Strategy failed statistical validation.")
