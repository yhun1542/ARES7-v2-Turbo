"""
Guardrails V3 Relaxed - 미세 완화 버전
데드존 0.33~0.67 → 0.34~0.66, 히스테리시스 간격 +0.01p
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GuardrailsV3Relaxed:
    """
    가드레일 V3 미세 완화 버전
    
    변경사항:
    - 베타 밴드: 0.33~0.67 → 0.34~0.66 (데드존 축소)
    - 히스테리시스: 간격 +0.01p (과민 반응 완화)
    
    목표:
    - 경보 0건 유지
    - Sharpe 회복 (2.44 → 2.50+)
    """
    
    def __init__(
        self,
        beta_min: float = 0.34,  # 0.33 → 0.34
        beta_max: float = 0.66,  # 0.67 → 0.66
        dd_threshold: float = 0.12,
        turnover_cap: float = 0.30,
        vol_target: float = 0.115,
        hysteresis: float = 0.01  # 히스테리시스 간격
    ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.dd_threshold = dd_threshold
        self.turnover_cap = turnover_cap
        self.vol_target = vol_target
        self.hysteresis = hysteresis
        
        # 상태 추적
        self.alerts = []
        self.last_beta_adjustment = 0
        
        logger.info(f"GuardrailsV3Relaxed: beta=[{beta_min:.2f}, {beta_max:.2f}], hysteresis={hysteresis:.3f}")
    
    def check_beta_band(self, portfolio_beta: float) -> Tuple[bool, float]:
        """
        베타 밴드 체크 (히스테리시스 적용)
        
        Returns:
            (needs_adjustment, target_beta)
        """
        # 히스테리시스 적용
        lower_trigger = self.beta_min - self.hysteresis
        upper_trigger = self.beta_max + self.hysteresis
        
        if portfolio_beta < lower_trigger:
            self.alerts.append({
                'type': 'beta_low',
                'value': portfolio_beta,
                'threshold': self.beta_min
            })
            return True, self.beta_min
        
        elif portfolio_beta > upper_trigger:
            self.alerts.append({
                'type': 'beta_high',
                'value': portfolio_beta,
                'threshold': self.beta_max
            })
            return True, self.beta_max
        
        return False, portfolio_beta
    
    def check_drawdown_guard(self, current_dd: float) -> Tuple[bool, float]:
        """
        드로우다운 가드 체크
        
        Returns:
            (needs_scaling, scale_factor)
        """
        if current_dd < -self.dd_threshold:
            excess_dd = abs(current_dd) - self.dd_threshold
            scale_factor = max(0.5, 1 - excess_dd * 2)  # 최대 50% 축소
            
            self.alerts.append({
                'type': 'drawdown',
                'value': current_dd,
                'threshold': -self.dd_threshold,
                'scale_factor': scale_factor
            })
            
            return True, scale_factor
        
        return False, 1.0
    
    def check_turnover_cap(self, turnover: float) -> Tuple[bool, float]:
        """
        턴오버 캡 체크
        
        Returns:
            (needs_scaling, scale_factor)
        """
        if turnover > self.turnover_cap:
            scale_factor = self.turnover_cap / turnover
            
            self.alerts.append({
                'type': 'turnover',
                'value': turnover,
                'threshold': self.turnover_cap,
                'scale_factor': scale_factor
            })
            
            return True, scale_factor
        
        return False, 1.0
    
    def check_vol_targeting(self, realized_vol: float) -> Tuple[bool, float]:
        """
        변동성 타게팅 체크
        
        Returns:
            (needs_scaling, scale_factor)
        """
        if realized_vol > 0:
            scale_factor = self.vol_target / realized_vol
            
            # 과도한 조정 방지 (±20% 제한)
            scale_factor = np.clip(scale_factor, 0.8, 1.2)
            
            if abs(scale_factor - 1.0) > 0.05:  # 5% 이상 차이만 조정
                self.alerts.append({
                    'type': 'volatility',
                    'value': realized_vol,
                    'target': self.vol_target,
                    'scale_factor': scale_factor
                })
                
                return True, scale_factor
        
        return False, 1.0
    
    def apply_guardrails(
        self,
        weights: pd.Series,
        portfolio_beta: float,
        current_dd: float,
        turnover: float,
        realized_vol: float
    ) -> Tuple[pd.Series, Dict]:
        """
        모든 가드레일 적용 (단일 패스)
        
        순서: 베타 헤지 → 스케일링 → 캡
        
        Returns:
            (adjusted_weights, report)
        """
        self.alerts = []  # 리셋
        adjusted = weights.copy()
        
        # 1. 베타 헤지
        needs_beta_adj, target_beta = self.check_beta_band(portfolio_beta)
        if needs_beta_adj:
            beta_adjustment = target_beta - portfolio_beta
            # 베타 헤지 로직 (간소화)
            adjusted *= (1 + beta_adjustment * 0.1)  # 10% 조정
            adjusted /= adjusted.sum()
        
        # 2. 스케일링 (DD, Vol)
        _, dd_scale = self.check_drawdown_guard(current_dd)
        _, vol_scale = self.check_vol_targeting(realized_vol)
        
        combined_scale = dd_scale * vol_scale
        adjusted *= combined_scale
        
        # 3. 턴오버 캡
        _, turnover_scale = self.check_turnover_cap(turnover)
        adjusted *= turnover_scale
        
        # 재정규화
        adjusted /= adjusted.sum()
        
        # 리포트
        report = {
            'alerts': self.alerts,
            'alert_count': len(self.alerts),
            'beta_adjusted': needs_beta_adj,
            'dd_scale': dd_scale,
            'vol_scale': vol_scale,
            'turnover_scale': turnover_scale,
            'combined_scale': combined_scale * turnover_scale
        }
        
        logger.info(f"Guardrails applied: {len(self.alerts)} alerts, scale={report['combined_scale']:.3f}")
        
        return adjusted, report


# 테스트 코드
if __name__ == "__main__":
    print("=" * 80)
    print("Guardrails V3 Relaxed Test")
    print("=" * 80)
    print()
    
    # 테스트 포트폴리오
    np.random.seed(42)
    tickers = [f"STOCK_{i:03d}" for i in range(50)]
    weights = pd.Series(np.random.dirichlet(np.ones(50)), index=tickers)
    
    # 가드레일 적용
    guardrails = GuardrailsV3Relaxed()
    
    # 시나리오 1: 정상 (경보 없음)
    print("Scenario 1: Normal (No alerts)")
    print("-" * 80)
    adjusted, report = guardrails.apply_guardrails(
        weights=weights,
        portfolio_beta=0.50,  # 0.34~0.66 범위 내
        current_dd=-0.05,     # -12% 이내
        turnover=0.25,        # 30% 이내
        realized_vol=0.115    # 타겟과 일치
    )
    print(f"Alerts: {report['alert_count']}")
    print(f"Combined scale: {report['combined_scale']:.3f}")
    print()
    
    # 시나리오 2: 베타 높음 (경보 1개)
    print("Scenario 2: High Beta (1 alert)")
    print("-" * 80)
    adjusted, report = guardrails.apply_guardrails(
        weights=weights,
        portfolio_beta=0.75,  # 0.66 초과
        current_dd=-0.05,
        turnover=0.25,
        realized_vol=0.115
    )
    print(f"Alerts: {report['alert_count']}")
    print(f"Alert types: {[a['type'] for a in report['alerts']]}")
    print(f"Combined scale: {report['combined_scale']:.3f}")
    print()
    
    # 시나리오 3: 복합 (경보 3개)
    print("Scenario 3: Multiple violations (3 alerts)")
    print("-" * 80)
    adjusted, report = guardrails.apply_guardrails(
        weights=weights,
        portfolio_beta=0.75,
        current_dd=-0.15,     # -12% 초과
        turnover=0.35,        # 30% 초과
        realized_vol=0.115
    )
    print(f"Alerts: {report['alert_count']}")
    print(f"Alert types: {[a['type'] for a in report['alerts']]}")
    print(f"Combined scale: {report['combined_scale']:.3f}")
    print()
    
    print("=" * 80)
    print("✅ Guardrails V3 Relaxed Test Complete")
    print("=" * 80)
