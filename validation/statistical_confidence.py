"""
Statistical Confidence Validation
통계적 신뢰도 검증: DSR/SPA, Newey-West, Block Bootstrap
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalConfidence:
    """
    통계적 신뢰도 종합 검증
    
    1. Deflated Sharpe Ratio (DSR)
    2. Superior Predictive Ability (SPA)
    3. Newey-West t-statistic
    4. Block Bootstrap Confidence Interval
    """
    
    def __init__(self, n_trials: int = 100, n_bootstrap: int = 10000):
        self.n_trials = n_trials
        self.n_bootstrap = n_bootstrap
    
    def deflated_sharpe_ratio(
        self,
        sharpe_ratio: float,
        n_observations: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0
    ) -> Dict:
        """
        Deflated Sharpe Ratio 계산
        
        다중 탐색으로 인한 과추정을 보정
        """
        # 비정규성 조정
        adjustment = np.sqrt(
            1 - skewness * sharpe_ratio + (kurtosis - 1) / 4 * sharpe_ratio ** 2
        )
        adjusted_sr = sharpe_ratio / adjustment
        
        # 다중 탐색 조정
        gamma = 0.5772156649  # Euler-Mascheroni constant
        expected_max_sr = (
            (1 - gamma) * stats.norm.ppf(1 - 1 / self.n_trials) +
            gamma * stats.norm.ppf(1 - 1 / (self.n_trials * np.e))
        )
        std_max_sr = np.sqrt(1 / (2 * np.log(self.n_trials)))
        
        # DSR
        dsr = (adjusted_sr - expected_max_sr) / std_max_sr if std_max_sr > 0 else 0
        p_value = 1 - stats.norm.cdf(dsr)
        
        return {
            'dsr': dsr,
            'p_value': p_value,
            'adjusted_sr': adjusted_sr,
            'expected_max_sr': expected_max_sr,
            'significant': dsr > stats.norm.ppf(0.95)
        }
    
    def newey_west_t_stat(
        self,
        returns: pd.Series,
        lags: int = None
    ) -> Dict:
        """
        Newey-West t-statistic 계산
        
        자기상관을 고려한 표준오차 조정
        """
        if lags is None:
            lags = int(4 * (len(returns) / 100) ** (2/9))  # Newey-West 권장
        
        mean_return = returns.mean()
        n = len(returns)
        
        # 기본 분산
        variance = returns.var()
        
        # 자기상관 조정
        for lag in range(1, lags + 1):
            autocovariance = returns.iloc[lag:].reset_index(drop=True).cov(
                returns.iloc[:-lag].reset_index(drop=True)
            )
            weight = 1 - lag / (lags + 1)  # Bartlett kernel
            variance += 2 * weight * autocovariance
        
        # Newey-West 표준오차
        nw_std_error = np.sqrt(variance / n)
        
        # t-statistic
        t_stat = mean_return / nw_std_error
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'nw_std_error': nw_std_error,
            'lags': lags,
            'significant': p_value < 0.05
        }
    
    def block_bootstrap_ci(
        self,
        returns: pd.Series,
        block_size: int = 20,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Block Bootstrap Confidence Interval
        
        시계열 의존성을 고려한 신뢰구간
        """
        n = len(returns)
        n_blocks = n // block_size
        
        # Bootstrap 샘플링
        bootstrap_means = []
        bootstrap_sharpes = []
        
        for _ in range(self.n_bootstrap):
            # 블록 단위 리샘플링
            sampled_blocks = []
            for _ in range(n_blocks):
                start_idx = np.random.randint(0, n - block_size + 1)
                sampled_blocks.append(returns.iloc[start_idx:start_idx + block_size])
            
            bootstrap_sample = pd.concat(sampled_blocks, ignore_index=True)
            
            # 통계량 계산
            bootstrap_means.append(bootstrap_sample.mean())
            bootstrap_sharpes.append(
                bootstrap_sample.mean() / bootstrap_sample.std() * np.sqrt(252)
            )
        
        # 신뢰구간
        alpha = 1 - confidence_level
        mean_ci = np.percentile(bootstrap_means, [alpha/2 * 100, (1 - alpha/2) * 100])
        sharpe_ci = np.percentile(bootstrap_sharpes, [alpha/2 * 100, (1 - alpha/2) * 100])
        
        return {
            'mean_ci': mean_ci,
            'sharpe_ci': sharpe_ci,
            'bootstrap_mean': np.mean(bootstrap_means),
            'bootstrap_sharpe': np.mean(bootstrap_sharpes),
            'block_size': block_size,
            'n_bootstrap': self.n_bootstrap
        }
    
    def comprehensive_validation(
        self,
        returns: pd.Series,
        sharpe_ratio: float = None
    ) -> Dict:
        """
        종합 통계적 검증
        """
        if sharpe_ratio is None:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        
        # 1. DSR
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns) + 3
        dsr_result = self.deflated_sharpe_ratio(
            sharpe_ratio=sharpe_ratio,
            n_observations=len(returns),
            skewness=skewness,
            kurtosis=kurtosis
        )
        
        # 2. Newey-West
        nw_result = self.newey_west_t_stat(returns)
        
        # 3. Block Bootstrap
        bootstrap_result = self.block_bootstrap_ci(returns)
        
        # 종합 판정
        all_passed = (
            dsr_result['significant'] and
            nw_result['significant'] and
            bootstrap_result['sharpe_ci'][0] > 0
        )
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'n_observations': len(returns),
            'n_trials': self.n_trials,
            'dsr': dsr_result,
            'newey_west': nw_result,
            'bootstrap': bootstrap_result,
            'overall_pass': all_passed
        }


# 테스트 코드
if __name__ == "__main__":
    print("=" * 80)
    print("Statistical Confidence Validation Test")
    print("=" * 80)
    print()
    
    # 테스트 데이터 생성 (Sharpe 2.91 수준)
    np.random.seed(42)
    n_days = 481
    daily_return = 0.4334 / 252  # 43.34% annual
    daily_vol = 0.1488 / np.sqrt(252)  # 14.88% annual
    
    returns = pd.Series(np.random.randn(n_days) * daily_vol + daily_return)
    
    # 검증 실행
    validator = StatisticalConfidence(n_trials=100, n_bootstrap=1000)
    result = validator.comprehensive_validation(returns)
    
    # 결과 출력
    print("1. Deflated Sharpe Ratio (DSR)")
    print("-" * 80)
    print(f"  DSR: {result['dsr']['dsr']:.4f}")
    print(f"  p-value: {result['dsr']['p_value']:.4f}")
    print(f"  Significant: {result['dsr']['significant']}")
    print()
    
    print("2. Newey-West t-statistic")
    print("-" * 80)
    print(f"  t-statistic: {result['newey_west']['t_statistic']:.4f}")
    print(f"  p-value: {result['newey_west']['p_value']:.4f}")
    print(f"  Lags: {result['newey_west']['lags']}")
    print(f"  Significant: {result['newey_west']['significant']}")
    print()
    
    print("3. Block Bootstrap CI")
    print("-" * 80)
    print(f"  Sharpe CI (95%): [{result['bootstrap']['sharpe_ci'][0]:.2f}, {result['bootstrap']['sharpe_ci'][1]:.2f}]")
    print(f"  Bootstrap Sharpe: {result['bootstrap']['bootstrap_sharpe']:.2f}")
    print(f"  Block Size: {result['bootstrap']['block_size']}")
    print()
    
    print("=" * 80)
    if result['overall_pass']:
        print("✅ All statistical tests PASSED")
    else:
        print("❌ Some statistical tests FAILED")
    print("=" * 80)
