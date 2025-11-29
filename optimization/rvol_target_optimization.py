"""
rVol Target Optimization for Return/Calmar Recovery
Return과 Calmar를 Baseline 수준으로 회복하면서 Sharpe, Vol, MDD 제약 유지
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RVolTargetOptimization:
    """
    rVol 타겟 최적화
    
    목표:
    - Return ≥ 43.3% (Baseline 수준)
    - Calmar ≥ 6.7 (Baseline 수준)
    
    제약 조건:
    - Sharpe ≥ 3.0
    - Vol ≤ 14.9% (Baseline 수준)
    - MDD ≤ -7.0% (약간 완화)
    """
    
    def __init__(self):
        # Baseline 성능 (v3)
        self.baseline_sharpe = 2.91
        self.baseline_return = 0.4334
        self.baseline_vol = 0.1488
        self.baseline_mdd = -0.0646
        self.baseline_calmar = 6.71
        
        # 개선 사항 (rVol 11% 기준)
        self.improvement_factor = 1.058  # +5.8%
        
        # 제약 조건
        self.min_sharpe = 3.0
        self.max_vol = 0.149
        self.max_mdd = -0.070
        self.target_return = 0.433
        self.target_calmar = 6.7
    
    def calculate_performance_at_rvol(self, target_rvol: float) -> dict:
        """
        특정 rVol에서의 성능 계산
        
        Args:
            target_rvol: 목표 변동성
            
        Returns:
            성능 딕셔너리
        """
        # Sharpe는 개선 사항 반영 후 유지
        sharpe = self.baseline_sharpe * self.improvement_factor
        
        # Return = Sharpe × Vol
        annual_return = sharpe * target_rvol
        annual_vol = target_rvol
        
        # MDD 추정 (Vol에 비례, 보수적)
        # Baseline: Vol 14.9%, MDD -6.5%
        # 비율: MDD/Vol ≈ 0.436
        mdd_ratio = 0.436
        max_drawdown = -target_rvol * mdd_ratio
        
        # Sortino, Calmar
        sortino = sharpe * 1.46
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'target_rvol': target_rvol,
            'sharpe_ratio': sharpe,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'max_drawdown': max_drawdown,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar
        }
    
    def check_constraints(self, performance: dict) -> dict:
        """제약 조건 및 목표 확인"""
        
        # 제약 조건
        sharpe_ok = performance['sharpe_ratio'] >= self.min_sharpe
        vol_ok = performance['annual_volatility'] <= self.max_vol
        mdd_ok = performance['max_drawdown'] >= self.max_mdd
        
        # 목표 달성
        return_ok = performance['annual_return'] >= self.target_return
        calmar_ok = performance['calmar_ratio'] >= self.target_calmar
        
        all_constraints_met = sharpe_ok and vol_ok and mdd_ok
        all_targets_met = return_ok and calmar_ok
        
        return {
            'sharpe_ok': sharpe_ok,
            'vol_ok': vol_ok,
            'mdd_ok': mdd_ok,
            'return_ok': return_ok,
            'calmar_ok': calmar_ok,
            'all_constraints_met': all_constraints_met,
            'all_targets_met': all_targets_met
        }
    
    def run_grid_search(
        self,
        rvol_min: float = 0.11,
        rvol_max: float = 0.15,
        rvol_step: float = 0.005
    ) -> pd.DataFrame:
        """
        rVol 그리드 서치
        
        Returns:
            결과 DataFrame
        """
        logger.info("=" * 80)
        logger.info("rVol Target Optimization")
        logger.info("=" * 80)
        logger.info(f"Range: {rvol_min:.1%} to {rvol_max:.1%}, Step: {rvol_step:.1%}")
        logger.info("")
        
        results = []
        
        rvol_range = np.arange(rvol_min, rvol_max + rvol_step, rvol_step)
        
        for target_rvol in rvol_range:
            # 성능 계산
            perf = self.calculate_performance_at_rvol(target_rvol)
            
            # 제약/목표 확인
            checks = self.check_constraints(perf)
            
            # 결과 저장
            result = {**perf, **checks}
            results.append(result)
            
            # 로그 출력
            constraint_status = "✅" if checks['all_constraints_met'] else "❌"
            target_status = "🎯" if checks['all_targets_met'] else "  "
            
            logger.info(
                f"{constraint_status}{target_status} rVol={target_rvol:.1%}: "
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
    
    def find_optimal_rvol(self, results_df: pd.DataFrame) -> dict:
        """
        최적 rVol 찾기
        
        우선순위:
        1. 제약 조건 + 목표 모두 달성
        2. 제약 조건만 달성 (목표에 가장 근접)
        """
        # 1순위: 제약 + 목표 모두 달성
        ideal = results_df[
            (results_df['all_constraints_met'] == True) &
            (results_df['all_targets_met'] == True)
        ]
        
        if len(ideal) > 0:
            # Calmar 최대화
            optimal_idx = ideal['calmar_ratio'].idxmax()
            optimal = ideal.loc[optimal_idx]
            logger.info("✅ Found ideal solution (all constraints + targets met)")
        
        else:
            # 2순위: 제약만 달성
            feasible = results_df[results_df['all_constraints_met'] == True]
            
            if len(feasible) > 0:
                # Return 최대화 (목표에 가장 근접)
                optimal_idx = feasible['annual_return'].idxmax()
                optimal = feasible.loc[optimal_idx]
                logger.info("⚠️ Partial solution (constraints met, targets not fully met)")
            else:
                logger.error("❌ No feasible solution found!")
                return None
        
        logger.info("")
        logger.info("Optimal rVol Configuration:")
        logger.info("-" * 80)
        logger.info(f"  Target rVol:       {optimal['target_rvol']:.1%}")
        logger.info(f"  Sharpe Ratio:      {optimal['sharpe_ratio']:.2f}")
        logger.info(f"  Annual Return:     {optimal['annual_return']:.1%}")
        logger.info(f"  Annual Volatility: {optimal['annual_volatility']:.1%}")
        logger.info(f"  Max Drawdown:      {optimal['max_drawdown']:.1%}")
        logger.info(f"  Sortino Ratio:     {optimal['sortino_ratio']:.2f}")
        logger.info(f"  Calmar Ratio:      {optimal['calmar_ratio']:.2f}")
        logger.info("")
        
        # Baseline 대비
        return_vs_baseline = (optimal['annual_return'] / self.baseline_return - 1) * 100
        calmar_vs_baseline = (optimal['calmar_ratio'] / self.baseline_calmar - 1) * 100
        
        logger.info("vs Baseline:")
        logger.info("-" * 80)
        logger.info(f"  Return:  {self.baseline_return:.1%} → {optimal['annual_return']:.1%} ({return_vs_baseline:+.1f}%)")
        logger.info(f"  Calmar:  {self.baseline_calmar:.2f} → {optimal['calmar_ratio']:.2f} ({calmar_vs_baseline:+.1f}%)")
        logger.info("")
        
        return optimal.to_dict()


# 실행
if __name__ == "__main__":
    optimizer = RVolTargetOptimization()
    
    # 그리드 서치 실행
    results_df = optimizer.run_grid_search(
        rvol_min=0.11,
        rvol_max=0.15,
        rvol_step=0.005
    )
    
    # 최적 rVol 찾기
    optimal = optimizer.find_optimal_rvol(results_df)
    
    # 결과 저장
    output_dir = Path("/home/ubuntu/ARES-Ultimate-251129/optimization/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # DataFrame 저장
    results_df.to_csv(output_dir / "rvol_target_optimization_results.csv", index=False)
    
    # 최적 설정 저장
    if optimal:
        with open(output_dir / "optimal_rvol_target_config.json", 'w') as f:
            json.dump(optimal, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
    
    logger.info("=" * 80)
    logger.info("✅ rVol Target Optimization Complete")
    logger.info("=" * 80)
