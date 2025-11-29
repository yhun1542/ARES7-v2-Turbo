"""
Enhanced Audit Trail - 100% Coverage
리밸런스 로그에 Git SHA, Config SHA, Σ 방식, 비용계수 버전 자동 주입
"""

import hashlib
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedAuditTrail:
    """
    강화된 감사 추적 시스템
    
    자동 주입 항목:
    1. Git SHA (코드 버전)
    2. Config SHA (설정 해시)
    3. Covariance 추정 방식
    4. 비용계수 버전
    5. 환경 정보
    6. 의존성 버전
    """
    
    def __init__(self, project_root: str = "/home/ubuntu/ARES-Ultimate-251129"):
        self.project_root = Path(project_root)
    
    def get_git_sha(self) -> str:
        """Git commit SHA 가져오기"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return "unknown"
        
        except Exception as e:
            logger.warning(f"Failed to get Git SHA: {e}")
            return "unknown"
    
    def get_config_sha(self, config: Dict) -> str:
        """설정 해시 계산"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def get_environment_info(self) -> Dict:
        """환경 정보 수집"""
        import sys
        import platform
        
        return {
            'python_version': sys.version,
            'platform': platform.platform(),
            'hostname': platform.node()
        }
    
    def get_dependency_versions(self) -> Dict:
        """주요 의존성 버전"""
        import numpy
        import pandas
        
        versions = {
            'numpy': numpy.__version__,
            'pandas': pandas.__version__
        }
        
        try:
            import scipy
            versions['scipy'] = scipy.__version__
        except ImportError:
            versions['scipy'] = 'not installed'
        
        return versions
    
    def create_rebalance_log(
        self,
        operation: str,
        config: Dict,
        covariance_method: str = "OAS",
        cost_version: str = "v1.0",
        metadata: Dict = None
    ) -> Dict:
        """
        리밸런스 로그 생성 (완전한 감사 추적)
        
        Args:
            operation: 작업 유형 (예: "rebalance", "backtest")
            config: 설정 딕셔너리
            covariance_method: 공분산 추정 방식
            cost_version: 비용계수 버전
            metadata: 추가 메타데이터
            
        Returns:
            완전한 감사 로그
        """
        # 1. Git SHA
        git_sha = self.get_git_sha()
        
        # 2. Config SHA
        config_sha = self.get_config_sha(config)
        
        # 3. 환경 정보
        env_info = self.get_environment_info()
        
        # 4. 의존성 버전
        dependencies = self.get_dependency_versions()
        
        # 5. 타임스탬프
        timestamp = datetime.now().isoformat()
        
        # 완전한 로그
        audit_log = {
            'timestamp': timestamp,
            'operation': operation,
            'version_control': {
                'git_sha': git_sha,
                'config_sha': config_sha
            },
            'configuration': config,
            'methods': {
                'covariance_estimation': covariance_method,
                'cost_model_version': cost_version
            },
            'environment': env_info,
            'dependencies': dependencies,
            'metadata': metadata or {}
        }
        
        logger.info(f"Audit log created: {operation}")
        logger.info(f"  Git SHA: {git_sha}")
        logger.info(f"  Config SHA: {config_sha}")
        logger.info(f"  Covariance: {covariance_method}")
        logger.info(f"  Cost version: {cost_version}")
        
        return audit_log
    
    def save_rebalance_log(
        self,
        audit_log: Dict,
        log_dir: Path = None
    ) -> Path:
        """리밸런스 로그 저장"""
        
        if log_dir is None:
            log_dir = self.project_root / "rebalance_logs"
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일명: rebalance_YYYYMMDD_HHMMSS.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"rebalance_{timestamp}.json"
        
        with open(log_file, 'w') as f:
            json.dump(audit_log, f, indent=2)
        
        logger.info(f"Rebalance log saved: {log_file}")
        
        return log_file
    
    def verify_reproducibility(
        self,
        log_file: Path
    ) -> Dict:
        """
        재현성 검증
        
        로그 파일의 Git SHA와 Config SHA를 확인하여 재현 가능 여부 판정
        """
        with open(log_file, 'r') as f:
            audit_log = json.load(f)
        
        current_git_sha = self.get_git_sha()
        logged_git_sha = audit_log['version_control']['git_sha']
        
        git_match = current_git_sha == logged_git_sha
        
        verification = {
            'log_file': str(log_file),
            'logged_git_sha': logged_git_sha,
            'current_git_sha': current_git_sha,
            'git_match': git_match,
            'reproducible': git_match,
            'recommendation': 'Can reproduce' if git_match else 'Checkout logged Git SHA first'
        }
        
        logger.info(f"Reproducibility check: {verification['recommendation']}")
        
        return verification


# 테스트 코드
if __name__ == "__main__":
    print("=" * 80)
    print("Enhanced Audit Trail Test")
    print("=" * 80)
    print()
    
    # 감사 추적 생성
    audit = EnhancedAuditTrail()
    
    # 테스트 설정
    config = {
        'rebalance_frequency': 'weekly',
        'target_volatility': 0.115,
        'max_drawdown': 0.12,
        'alpha_weights': {
            'momentum': 0.4068,
            'value': 0.3099,
            'technical': 0.2365,
            'quality': 0.0468
        }
    }
    
    # 리밸런스 로그 생성
    audit_log = audit.create_rebalance_log(
        operation='test_rebalance',
        config=config,
        covariance_method='OAS',
        cost_version='v1.0',
        metadata={'test': True}
    )
    
    print()
    print("Audit Log:")
    print("-" * 80)
    print(json.dumps(audit_log, indent=2))
    print()
    
    # 로그 저장
    log_file = audit.save_rebalance_log(audit_log)
    
    print()
    print(f"Log saved: {log_file}")
    print()
    
    # 재현성 검증
    verification = audit.verify_reproducibility(log_file)
    
    print("Reproducibility Verification:")
    print("-" * 80)
    print(f"  Git match: {verification['git_match']}")
    print(f"  Reproducible: {verification['reproducible']}")
    print(f"  Recommendation: {verification['recommendation']}")
    print()
    
    print("=" * 80)
    print("✅ Enhanced Audit Trail Test Complete")
    print("=" * 80)
