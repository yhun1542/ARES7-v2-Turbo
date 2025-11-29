"""
rVol Efficiency Curve Optimization
Sharpe-vs-Vol 효율곡선에서 최적 rVol 탐색
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RVolEfficiencyCurve:
    """
    rVol 효율곡선 분석
    
    목표: 11%, 12%, 13%, 14% rVol에서 Sharpe-vs-Vol 효율 최대화 지점 탐색
    """
    
    def __init__(self):
        self.results = []
    
    def calculate_sharpe_at_rvol(
        self,
        base_returns: pd.Series,
        target_rvol: float,
        current_rvol: float = 0.1488
    ) -> dict:
        """
        특정 rVol에서의 Sharpe 추정
        
        Args:
            base_returns: 기준 수익률 시리즈
            target_rvol: 목표 변동성
            current_rvol: 현재 변동성
            
        Returns:
            성능 지표 딕셔너리
        """
        # 변동성 스케일링
        scale_factor = target_rvol / current_rvol
        
        # 수익률 조정 (변동성에 비례)
        adjusted_returns = base_returns * scale_factor
        
        # 성능 계산
        annual_return = adjusted_returns.mean() * 252
        annual_vol = adjusted_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # 효율성 (Sharpe / Vol)
        efficiency = sharpe / annual_vol if annual_vol > 0 else 0
        
        return {
            'target_rvol': target_rvol,
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'sharpe': sharpe,
            'efficiency': efficiency,
            'scale_factor': scale_factor
        }
    
    def run_rvol_sweep(
        self,
        base_returns: pd.Series,
        rvol_range: list = [0.11, 0.12, 0.13, 0.14],
        current_rvol: float = 0.1488
    ) -> pd.DataFrame:
        """
        rVol 범위에서 효율곡선 계산
        
        Returns:
            결과 DataFrame
        """
        logger.info(f"Running rVol sweep: {rvol_range}")
        
        results = []
        
        for target_rvol in rvol_range:
            result = self.calculate_sharpe_at_rvol(
                base_returns,
                target_rvol,
                current_rvol
            )
            results.append(result)
            
            logger.info(
                f"  rVol={target_rvol:.1%}: Sharpe={result['sharpe']:.2f}, "
                f"Efficiency={result['efficiency']:.2f}"
            )
        
        self.results = results
        return pd.DataFrame(results)
    
    def plot_efficiency_curve(self, output_path: Path = None):
        """효율곡선 시각화"""
        
        if not self.results:
            logger.warning("No results to plot")
            return
        
        df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Sharpe vs rVol
        axes[0, 0].plot(df['target_rvol'], df['sharpe'], 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Target rVol')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].set_title('Sharpe Ratio vs rVol')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Efficiency vs rVol
        axes[0, 1].plot(df['target_rvol'], df['efficiency'], 'o-', linewidth=2, markersize=8, color='green')
        axes[0, 1].set_xlabel('Target rVol')
        axes[0, 1].set_ylabel('Efficiency (Sharpe/Vol)')
        axes[0, 1].set_title('Efficiency vs rVol')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Return vs Vol
        axes[1, 0].plot(df['annual_vol'], df['annual_return'], 'o-', linewidth=2, markersize=8, color='orange')
        axes[1, 0].set_xlabel('Annual Volatility')
        axes[1, 0].set_ylabel('Annual Return')
        axes[1, 0].set_title('Return vs Volatility')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Summary table
        axes[1, 1].axis('off')
        table_data = []
        for _, row in df.iterrows():
            table_data.append([
                f"{row['target_rvol']:.1%}",
                f"{row['sharpe']:.2f}",
                f"{row['efficiency']:.2f}",
                f"{row['annual_return']:.1%}"
            ])
        
        table = axes[1, 1].table(
            cellText=table_data,
            colLabels=['rVol', 'Sharpe', 'Efficiency', 'Return'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def find_optimal_rvol(self) -> dict:
        """
        최적 rVol 찾기
        
        Returns:
            최적 설정 딕셔너리
        """
        if not self.results:
            logger.warning("No results available")
            return {}
        
        df = pd.DataFrame(self.results)
        
        # 효율성 최대화
        optimal_idx = df['efficiency'].idxmax()
        optimal = df.iloc[optimal_idx]
        
        logger.info("=" * 80)
        logger.info("Optimal rVol Configuration")
        logger.info("=" * 80)
        logger.info(f"  Target rVol:     {optimal['target_rvol']:.1%}")
        logger.info(f"  Sharpe Ratio:    {optimal['sharpe']:.2f}")
        logger.info(f"  Efficiency:      {optimal['efficiency']:.2f}")
        logger.info(f"  Annual Return:   {optimal['annual_return']:.1%}")
        logger.info(f"  Annual Vol:      {optimal['annual_vol']:.1%}")
        logger.info("=" * 80)
        
        return optimal.to_dict()


# 테스트 코드
if __name__ == "__main__":
    print("=" * 80)
    print("rVol Efficiency Curve Optimization")
    print("=" * 80)
    print()
    
    # 테스트 데이터 (Sharpe 2.91 수준)
    np.random.seed(42)
    n_days = 481
    daily_return = 0.4334 / 252  # 43.34% annual
    daily_vol = 0.1488 / np.sqrt(252)  # 14.88% annual
    
    base_returns = pd.Series(np.random.randn(n_days) * daily_vol + daily_return)
    
    # 효율곡선 분석
    optimizer = RVolEfficiencyCurve()
    results_df = optimizer.run_rvol_sweep(
        base_returns,
        rvol_range=[0.11, 0.12, 0.13, 0.14],
        current_rvol=0.1488
    )
    
    print()
    print("Results:")
    print("-" * 80)
    print(results_df.to_string(index=False))
    print()
    
    # 최적 rVol
    optimal = optimizer.find_optimal_rvol()
    
    # 플롯 저장
    output_dir = Path("/home/ubuntu/ARES-Ultimate-251129/optimization/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    optimizer.plot_efficiency_curve(output_dir / "rvol_efficiency_curve.png")
    
    print()
    print("=" * 80)
    print("✅ rVol Efficiency Curve Optimization Complete")
    print("=" * 80)
