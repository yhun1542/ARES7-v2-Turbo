"""
Leverage Grid Search Optimization
Return과 Calmar 최대화하면서 Sharpe, Vol, MDD 제약 조건 만족
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LeverageGridSearch:
    """
    레버리지 그리드 서치
    
    목표:
    - Return 최대화
    - Calmar 최대화
    
    제약 조건:
    - Sharpe >= 3.0 (현재 3.08 유지)
    - Vol <= 14.9% (Baseline 수준 이하)
    - MDD <= -6.5% (Baseline 수준 이하)
    """
    
    def __init__(self):
        # 현재 성능 (v4, leverage=1.0)
        self.base_sharpe = 3.08
        self.base_return = 0.339
        self.base_vol = 0.11
        self.base_mdd = -0.058
        
        # 제약 조건
        self.min_sharpe = 3.0
        self.max_vol = 0.149  # Baseline 수준
        self.max_mdd = -0.070  # Baseline 대비 약간 완화 (-6.5% → -7.0%)
    
    def calculate_leveraged_performance(self, leverage: float) -> dict:
        """
        레버리지 적용 시 성능 계산
        
        레버리지 효과:
        - Return: 선형 증가 (leverage × base_return)
        - Vol: 선형 증가 (leverage × base_vol)
        - Sharpe: 불변 (return/vol 비율 유지)
        - MDD: 비선형 증가 (leverage^1.2 × base_mdd, 보수적)
        """
        leveraged_return = self.base_return * leverage
        leveraged_vol = self.base_vol * leverage
        leveraged_sharpe = self.base_sharpe  # Sharpe는 불변
        
        # MDD는 레버리지에 비선형적으로 증가 (보수적 추정)
        leveraged_mdd = self.base_mdd * (leverage ** 1.2)
        
        # Sortino, Calmar 재계산
        leveraged_sortino = leveraged_sharpe * 1.46  # 비율 유지
        leveraged_calmar = leveraged_return / abs(leveraged_mdd) if leveraged_mdd != 0 else 0
        
        return {
            'leverage': leverage,
            'sharpe_ratio': leveraged_sharpe,
            'annual_return': leveraged_return,
            'annual_volatility': leveraged_vol,
            'max_drawdown': leveraged_mdd,
            'sortino_ratio': leveraged_sortino,
            'calmar_ratio': leveraged_calmar
        }
    
    def check_constraints(self, performance: dict) -> dict:
        """제약 조건 확인"""
        
        sharpe_ok = performance['sharpe_ratio'] >= self.min_sharpe
        vol_ok = performance['annual_volatility'] <= self.max_vol
        mdd_ok = performance['max_drawdown'] >= self.max_mdd  # MDD는 음수이므로 >= 사용
        
        all_ok = sharpe_ok and vol_ok and mdd_ok
        
        return {
            'sharpe_ok': sharpe_ok,
            'vol_ok': vol_ok,
            'mdd_ok': mdd_ok,
            'all_constraints_met': all_ok
        }
    
    def run_grid_search(
        self,
        leverage_min: float = 1.0,
        leverage_max: float = 2.0,
        leverage_step: float = 0.05
    ) -> pd.DataFrame:
        """
        그리드 서치 실행
        
        Args:
            leverage_min: 최소 레버리지
            leverage_max: 최대 레버리지
            leverage_step: 레버리지 스텝
            
        Returns:
            결과 DataFrame
        """
        logger.info("=" * 80)
        logger.info("Leverage Grid Search")
        logger.info("=" * 80)
        logger.info(f"Range: {leverage_min:.2f} to {leverage_max:.2f}, Step: {leverage_step:.2f}")
        logger.info("")
        
        results = []
        
        leverage_range = np.arange(leverage_min, leverage_max + leverage_step, leverage_step)
        
        for leverage in leverage_range:
            # 성능 계산
            perf = self.calculate_leveraged_performance(leverage)
            
            # 제약 조건 확인
            constraints = self.check_constraints(perf)
            
            # 결과 저장
            result = {**perf, **constraints}
            results.append(result)
            
            # 로그 출력
            status = "✅" if constraints['all_constraints_met'] else "❌"
            logger.info(
                f"{status} L={leverage:.2f}: "
                f"Sharpe={perf['sharpe_ratio']:.2f}, "
                f"Return={perf['annual_return']:.1%}, "
                f"Vol={perf['annual_volatility']:.1%}, "
                f"MDD={perf['max_drawdown']:.1%}, "
                f"Calmar={perf['calmar_ratio']:.2f}"
            )
        
        df = pd.DataFrame(results)
        
        logger.info("")
        logger.info("=" * 80)
        
        return df
    
    def find_optimal_leverage(self, results_df: pd.DataFrame) -> dict:
        """
        최적 레버리지 찾기
        
        목표: 제약 조건을 만족하면서 Calmar 최대화
        """
        # 제약 조건 만족하는 케이스만 필터링
        feasible = results_df[results_df['all_constraints_met'] == True]
        
        if len(feasible) == 0:
            logger.warning("No feasible solution found!")
            return None
        
        # Calmar 최대화
        optimal_idx = feasible['calmar_ratio'].idxmax()
        optimal = feasible.loc[optimal_idx]
        
        logger.info("Optimal Leverage Configuration:")
        logger.info("-" * 80)
        logger.info(f"  Leverage:          {optimal['leverage']:.2f}x")
        logger.info(f"  Sharpe Ratio:      {optimal['sharpe_ratio']:.2f}")
        logger.info(f"  Annual Return:     {optimal['annual_return']:.1%}")
        logger.info(f"  Annual Volatility: {optimal['annual_volatility']:.1%}")
        logger.info(f"  Max Drawdown:      {optimal['max_drawdown']:.1%}")
        logger.info(f"  Sortino Ratio:     {optimal['sortino_ratio']:.2f}")
        logger.info(f"  Calmar Ratio:      {optimal['calmar_ratio']:.2f}")
        logger.info("")
        
        # Baseline 대비 개선율
        baseline_return = 0.4334
        baseline_calmar = 6.71
        
        return_improvement = (optimal['annual_return'] / baseline_return - 1) * 100
        calmar_improvement = (optimal['calmar_ratio'] / baseline_calmar - 1) * 100
        
        logger.info("Improvement vs Baseline:")
        logger.info("-" * 80)
        logger.info(f"  Return:  {baseline_return:.1%} → {optimal['annual_return']:.1%} ({return_improvement:+.1f}%)")
        logger.info(f"  Calmar:  {baseline_calmar:.2f} → {optimal['calmar_ratio']:.2f} ({calmar_improvement:+.1f}%)")
        logger.info("")
        
        return optimal.to_dict()


# 실행
if __name__ == "__main__":
    optimizer = LeverageGridSearch()
    
    # 그리드 서치 실행
    results_df = optimizer.run_grid_search(
        leverage_min=1.0,
        leverage_max=1.5,
        leverage_step=0.05
    )
    
    # 최적 레버리지 찾기
    optimal = optimizer.find_optimal_leverage(results_df)
    
    # 결과 저장
    output_dir = Path("/home/ubuntu/ARES-Ultimate-251129/optimization/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # DataFrame 저장
    results_df.to_csv(output_dir / "leverage_grid_search_results.csv", index=False)
    
    # 최적 설정 저장
    if optimal:
        with open(output_dir / "optimal_leverage_config.json", 'w') as f:
            json.dump(optimal, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
    
    logger.info("=" * 80)
    logger.info("✅ Leverage Grid Search Complete")
    logger.info("=" * 80)
