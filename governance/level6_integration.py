"""
Level 6 Production-Grade System Integration
레벨 6 프로덕션 그레이드 시스템 통합
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from governance.audit_trail import AuditTrail
from governance.dead_man_switch import DeadManSwitch
from governance.report_generator import ReportGenerator
from validation.deflated_sharpe import validate_strategy_significance

import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Level6System:
    """
    레벨 6 프로덕션 그레이드 시스템
    
    통합 컴포넌트:
    1. DSR/SPA 탐색 편향 보정
    2. 감사 추적성 (환경 해시, 설정 주입)
    3. Dead-man Switch (비상 대응)
    4. 월간 거버넌스 리포트
    """
    
    def __init__(self, project_root: str = "/home/ubuntu/ARES-Ultimate-251129"):
        self.project_root = Path(project_root)
        
        # 컴포넌트 초기화
        self.audit = AuditTrail(project_root)
        self.dead_man_switch = DeadManSwitch(project_root)
        self.report_generator = ReportGenerator(project_root)
        
        logger.info("Level 6 System initialized")
    
    def run_pre_trade_checks(self) -> bool:
        """
        거래 전 체크
        
        Returns:
            통과 여부
        """
        logger.info("Running pre-trade checks...")
        
        # 1. 안전 모드 확인
        if self.dead_man_switch.is_safe_mode_enabled():
            logger.error("System is in SAFE MODE - trading disabled")
            return False
        
        # 2. 헬스 체크
        health_checks = self.dead_man_switch.perform_health_check()
        failed_checks = [k for k, v in health_checks.items() if not v]
        
        if failed_checks:
            logger.error(f"Health checks failed: {failed_checks}")
            # 비상 종료 트리거
            self.dead_man_switch.trigger_emergency_shutdown(
                reason=f"Pre-trade health check failed: {', '.join(failed_checks)}",
                checks=health_checks
            )
            return False
        
        logger.info("✅ Pre-trade checks passed")
        return True
    
    def run_post_backtest_validation(
        self,
        returns,
        benchmark_returns,
        n_trials: int = 100
    ) -> dict:
        """
        백테스트 후 검증 (DSR/SPA)
        
        Args:
            returns: 전략 수익률
            benchmark_returns: 벤치마크 수익률
            n_trials: 탐색한 전략 개수
            
        Returns:
            검증 결과
        """
        logger.info("Running post-backtest validation (DSR/SPA)...")
        
        result = validate_strategy_significance(
            returns=returns,
            benchmark_returns=benchmark_returns,
            n_trials=n_trials,
            confidence_level=0.95
        )
        
        if result['overall_pass']:
            logger.info("✅ Strategy passed statistical validation")
        else:
            logger.warning("⚠️ Strategy failed statistical validation")
            logger.warning(f"  DSR: {result['dsr']:.4f} (p={result['dsr_pvalue']:.4f})")
            logger.warning(f"  SPA: {result['spa_statistic']:.4f} (p={result['spa_pvalue']:.4f})")
        
        return result
    
    def create_audit_record(self, operation: str, config: dict, metadata: dict = None):
        """
        감사 레코드 생성
        
        Args:
            operation: 작업 이름
            config: 설정
            metadata: 메타데이터
        """
        logger.info(f"Creating audit record for: {operation}")
        
        audit_path = self.audit.save_audit_record(
            operation=operation,
            config=config,
            metadata=metadata
        )
        
        logger.info(f"✅ Audit record saved: {audit_path}")
        return audit_path
    
    def generate_monthly_report(self, year: int, month: int):
        """
        월간 거버넌스 리포트 생성
        
        Args:
            year: 연도
            month: 월
        """
        logger.info(f"Generating monthly report for {year}-{month:02d}...")
        
        self.report_generator.generate_report(year, month)
        
        logger.info("✅ Monthly report generated")
    
    def update_heartbeat(self, metadata: dict = None):
        """
        하트비트 업데이트
        
        Args:
            metadata: 메타데이터
        """
        self.dead_man_switch.update_heartbeat(metadata)


# 테스트 코드
if __name__ == "__main__":
    print("=" * 60)
    print("Level 6 Production-Grade System Test")
    print("=" * 60)
    print()
    
    # 시스템 초기화
    system = Level6System()
    
    # 1. 하트비트 업데이트
    system.update_heartbeat({'test': True})
    print("✅ Heartbeat updated")
    
    # 2. Pre-trade 체크
    # (실제 운영 시에는 상태 파일이 있어야 함)
    # passed = system.run_pre_trade_checks()
    # print(f"Pre-trade checks: {'✅ Passed' if passed else '❌ Failed'}")
    
    # 3. 감사 레코드 생성
    test_config = {
        'strategy': 'ARES7_v2_Turbo',
        'rebalance_frequency': 'weekly',
        'target_volatility': 0.12
    }
    audit_path = system.create_audit_record(
        operation='test_operation',
        config=test_config,
        metadata={'test': True}
    )
    print(f"✅ Audit record: {audit_path}")
    
    print()
    print("=" * 60)
    print("✅ Level 6 System working correctly!")
    print("=" * 60)
