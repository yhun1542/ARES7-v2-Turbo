"""
Audit Trail System
감사 추적성 - 환경 해시, 설정, 의존성 자동 주입
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuditTrail:
    """
    감사 추적 시스템
    
    모든 실행에 대해 환경 정보, 설정, 의존성을 자동으로 기록하여
    완전한 재현 가능성과 추적성을 보장합니다.
    """
    
    def __init__(self, project_root: str = "/home/ubuntu/ARES-Ultimate-Final"):
        self.project_root = Path(project_root)
        self.audit_dir = self.project_root / "audit_logs"
        self.audit_dir.mkdir(exist_ok=True)
    
    def capture_environment(self) -> Dict[str, Any]:
        """
        현재 실행 환경 캡처
        
        Returns:
            환경 정보 딕셔너리
        """
        env_info = {
            'timestamp': datetime.utcnow().isoformat(),
            'python_version': sys.version,
            'python_executable': sys.executable,
            'platform': sys.platform,
            'cwd': os.getcwd(),
            'user': os.getenv('USER', 'unknown'),
            'hostname': os.getenv('HOSTNAME', 'unknown'),
        }
        
        # Git 정보
        try:
            git_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.project_root,
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            git_branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=self.project_root,
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            git_dirty = subprocess.call(
                ['git', 'diff-index', '--quiet', 'HEAD', '--'],
                cwd=self.project_root,
                stderr=subprocess.DEVNULL
            ) != 0
            
            env_info['git'] = {
                'commit': git_hash,
                'branch': git_branch,
                'dirty': git_dirty
            }
        except Exception as e:
            logger.warning(f"Failed to capture git info: {e}")
            env_info['git'] = None
        
        return env_info
    
    def capture_dependencies(self) -> Dict[str, str]:
        """
        Python 패키지 의존성 캡처
        
        Returns:
            패키지명: 버전 딕셔너리
        """
        try:
            result = subprocess.check_output(
                [sys.executable, '-m', 'pip', 'freeze'],
                stderr=subprocess.DEVNULL
            ).decode()
            
            dependencies = {}
            for line in result.strip().split('\n'):
                if '==' in line:
                    pkg, ver = line.split('==', 1)
                    dependencies[pkg] = ver
            
            return dependencies
        except Exception as e:
            logger.warning(f"Failed to capture dependencies: {e}")
            return {}
    
    def hash_config(self, config: Dict[str, Any]) -> str:
        """
        설정 파일 해시 생성
        
        Args:
            config: 설정 딕셔너리
            
        Returns:
            SHA256 해시
        """
        # 정렬된 JSON으로 직렬화
        config_str = json.dumps(config, sort_keys=True, indent=2)
        
        # SHA256 해시
        hash_obj = hashlib.sha256(config_str.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def hash_requirements(self) -> str:
        """
        requirements.txt 해시 생성
        
        Returns:
            SHA256 해시
        """
        req_file = self.project_root / "requirements.txt"
        
        if not req_file.exists():
            logger.warning("requirements.txt not found")
            return "N/A"
        
        with open(req_file, 'rb') as f:
            hash_obj = hashlib.sha256(f.read())
            return hash_obj.hexdigest()
    
    def hash_docker_image(self) -> str:
        """
        Docker 이미지 해시 생성 (존재하는 경우)
        
        Returns:
            Docker 이미지 SHA256
        """
        try:
            result = subprocess.check_output(
                ['docker', 'images', '--no-trunc', '--quiet', 'ares7:latest'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            return result if result else "N/A"
        except Exception:
            return "N/A"
    
    def create_audit_record(
        self,
        operation: str,
        config: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        감사 레코드 생성
        
        Args:
            operation: 작업 이름 (예: "backtest", "live_trading")
            config: 설정 딕셔너리
            metadata: 추가 메타데이터
            
        Returns:
            감사 레코드
        """
        record = {
            'operation': operation,
            'environment': self.capture_environment(),
            'dependencies': self.capture_dependencies(),
            'config_hash': self.hash_config(config),
            'requirements_hash': self.hash_requirements(),
            'docker_image_hash': self.hash_docker_image(),
            'config': config,
            'metadata': metadata or {}
        }
        
        return record
    
    def save_audit_record(
        self,
        operation: str,
        config: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> Path:
        """
        감사 레코드 저장
        
        Args:
            operation: 작업 이름
            config: 설정 딕셔너리
            metadata: 추가 메타데이터
            
        Returns:
            저장된 파일 경로
        """
        record = self.create_audit_record(operation, config, metadata)
        
        # 파일명: {operation}_{timestamp}.json
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"{operation}_{timestamp}.json"
        filepath = self.audit_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(record, f, indent=2)
        
        logger.info(f"Audit record saved: {filepath}")
        
        return filepath
    
    def inject_audit_into_report(
        self,
        report_path: Path,
        audit_record: Dict[str, Any]
    ):
        """
        리포트에 감사 정보 주입
        
        Args:
            report_path: 리포트 파일 경로
            audit_record: 감사 레코드
        """
        # 리포트 읽기
        with open(report_path, 'r') as f:
            content = f.read()
        
        # 감사 섹션 생성
        audit_section = self._format_audit_section(audit_record)
        
        # 리포트 끝에 추가
        with open(report_path, 'a') as f:
            f.write('\n\n')
            f.write(audit_section)
        
        logger.info(f"Audit info injected into report: {report_path}")
    
    def _format_audit_section(self, record: Dict[str, Any]) -> str:
        """감사 섹션 포맷팅"""
        env = record['environment']
        git = env.get('git', {})
        
        section = f"""---

## 감사 추적 정보 (Audit Trail)

### 실행 환경
- **타임스탬프:** {env['timestamp']}
- **Python 버전:** {env['python_version'].split()[0]}
- **플랫폼:** {env['platform']}
- **사용자:** {env['user']}@{env['hostname']}
- **작업 디렉토리:** {env['cwd']}

### 버전 관리
- **Git Commit:** {git.get('commit', 'N/A')[:8] if git else 'N/A'}
- **Git Branch:** {git.get('branch', 'N/A') if git else 'N/A'}
- **Working Tree Clean:** {'No' if git and git.get('dirty') else 'Yes'}

### 의존성 해시
- **Config SHA256:** `{record['config_hash'][:16]}...`
- **Requirements SHA256:** `{record['requirements_hash'][:16]}...`
- **Docker Image SHA256:** `{record['docker_image_hash'][:16] if record['docker_image_hash'] != 'N/A' else 'N/A'}...`

### 주요 패키지 버전
"""
        
        # 주요 패키지만 표시
        key_packages = ['numpy', 'pandas', 'numba', 'scikit-learn']
        deps = record['dependencies']
        
        for pkg in key_packages:
            if pkg in deps:
                section += f"- **{pkg}:** {deps[pkg]}\n"
        
        section += f"\n**전체 의존성:** {len(deps)} packages\n"
        
        return section
    
    def verify_reproducibility(
        self,
        audit_record_path: Path
    ) -> bool:
        """
        재현 가능성 검증
        
        Args:
            audit_record_path: 감사 레코드 파일 경로
            
        Returns:
            재현 가능 여부
        """
        with open(audit_record_path, 'r') as f:
            old_record = json.load(f)
        
        # 현재 환경 캡처
        current_env = self.capture_environment()
        current_deps = self.capture_dependencies()
        
        # Git 커밋 비교
        old_git = old_record['environment'].get('git', {})
        current_git = current_env.get('git', {})
        
        git_match = (
            old_git and current_git and
            old_git.get('commit') == current_git.get('commit')
        )
        
        # 의존성 비교
        old_deps = old_record['dependencies']
        deps_match = old_deps == current_deps
        
        # 결과
        reproducible = git_match and deps_match
        
        if not reproducible:
            logger.warning("Reproducibility check failed:")
            if not git_match:
                logger.warning(f"  Git commit mismatch: {old_git.get('commit', 'N/A')[:8]} vs {current_git.get('commit', 'N/A')[:8]}")
            if not deps_match:
                logger.warning(f"  Dependencies mismatch")
        else:
            logger.info("✅ Reproducibility verified")
        
        return reproducible


# 테스트 코드
if __name__ == "__main__":
    print("=" * 60)
    print("Audit Trail System Test")
    print("=" * 60)
    print()
    
    # 감사 추적 시스템 초기화
    audit = AuditTrail()
    
    # 테스트 설정
    test_config = {
        'strategy': 'ARES7_v2_Turbo',
        'rebalance_frequency': 'weekly',
        'target_volatility': 0.12,
        'max_drawdown': 0.08,
        'alpha_weights': {
            'momentum': 0.4068,
            'value': 0.3099,
            'technical': 0.2365,
            'quality': 0.0468
        }
    }
    
    # 감사 레코드 생성 및 저장
    audit_path = audit.save_audit_record(
        operation='test_backtest',
        config=test_config,
        metadata={'test': True}
    )
    
    print(f"Audit record saved: {audit_path}")
    print()
    
    # 감사 레코드 읽기
    with open(audit_path, 'r') as f:
        record = json.load(f)
    
    print("Environment:")
    print(json.dumps(record['environment'], indent=2))
    print()
    
    print(f"Config Hash: {record['config_hash']}")
    print(f"Requirements Hash: {record['requirements_hash']}")
    print()
    
    print(f"Total Dependencies: {len(record['dependencies'])}")
    print()
    
    print("✅ Audit trail system working correctly!")
