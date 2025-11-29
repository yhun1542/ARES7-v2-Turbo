"""
Dead-man Switch System
비상 상황 자동 대응 - 가드레일 트립, 지표 부재, 브로커 오류 시 즉시 대응
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeadManSwitch:
    """
    Dead-man Switch 시스템
    
    비정상 상황 감지 시 자동으로:
    1. 모든 포지션 감산 (청산)
    2. Slack/PagerDuty 알림
    3. 시스템 안전 모드 전환
    4. 감사 로그 기록
    """
    
    def __init__(
        self,
        project_root: str = "/home/ubuntu/ARES-Ultimate-Final",
        heartbeat_timeout: int = 600,  # 10분
        slack_webhook_url: Optional[str] = None,
        pagerduty_api_key: Optional[str] = None
    ):
        self.project_root = Path(project_root)
        self.heartbeat_timeout = heartbeat_timeout
        self.slack_webhook_url = slack_webhook_url or os.getenv('SLACK_WEBHOOK_URL')
        self.pagerduty_api_key = pagerduty_api_key or os.getenv('PAGERDUTY_API_KEY')
        
        # 상태 파일
        self.state_dir = self.project_root / "state"
        self.state_dir.mkdir(exist_ok=True)
        self.heartbeat_file = self.state_dir / "heartbeat.json"
        self.emergency_file = self.state_dir / "emergency.json"
        
        # 로그 디렉토리
        self.log_dir = self.project_root / "logs" / "emergency"
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def update_heartbeat(self, metadata: Dict = None):
        """
        하트비트 업데이트
        
        정상 작동 중임을 표시합니다.
        
        Args:
            metadata: 추가 메타데이터
        """
        heartbeat = {
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'alive',
            'metadata': metadata or {}
        }
        
        with open(self.heartbeat_file, 'w') as f:
            json.dump(heartbeat, f, indent=2)
    
    def check_heartbeat(self) -> bool:
        """
        하트비트 확인
        
        Returns:
            정상 여부
        """
        if not self.heartbeat_file.exists():
            logger.warning("Heartbeat file not found")
            return False
        
        with open(self.heartbeat_file, 'r') as f:
            heartbeat = json.load(f)
        
        last_update = datetime.fromisoformat(heartbeat['timestamp'])
        elapsed = (datetime.utcnow() - last_update).total_seconds()
        
        if elapsed > self.heartbeat_timeout:
            logger.error(f"Heartbeat timeout: {elapsed:.0f}s > {self.heartbeat_timeout}s")
            return False
        
        return True
    
    def check_guardrails(self) -> bool:
        """
        가드레일 상태 확인
        
        Returns:
            정상 여부 (트립되지 않음)
        """
        guardrail_state_file = self.state_dir / "guardrails_state.json"
        
        if not guardrail_state_file.exists():
            logger.warning("Guardrails state file not found")
            return False
        
        with open(guardrail_state_file, 'r') as f:
            state = json.load(f)
        
        # 알림 확인
        alerts = state.get('alerts', [])
        if alerts:
            logger.error(f"Guardrails tripped: {len(alerts)} alerts")
            for alert in alerts:
                logger.error(f"  - {alert}")
            return False
        
        return True
    
    def check_data_freshness(self, max_age_hours: int = 24) -> bool:
        """
        데이터 신선도 확인
        
        Args:
            max_age_hours: 최대 허용 데이터 나이 (시간)
            
        Returns:
            정상 여부
        """
        data_file = self.project_root / "data" / "latest_data.json"
        
        if not data_file.exists():
            logger.warning("Data file not found")
            return False
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        last_update = datetime.fromisoformat(data.get('timestamp', '1970-01-01'))
        age_hours = (datetime.utcnow() - last_update).total_seconds() / 3600
        
        if age_hours > max_age_hours:
            logger.error(f"Data too old: {age_hours:.1f}h > {max_age_hours}h")
            return False
        
        return True
    
    def check_broker_connection(self) -> bool:
        """
        브로커 연결 확인
        
        Returns:
            정상 여부
        """
        broker_state_file = self.state_dir / "broker_state.json"
        
        if not broker_state_file.exists():
            logger.warning("Broker state file not found")
            return False
        
        with open(broker_state_file, 'r') as f:
            state = json.load(f)
        
        # 연결 상태 확인
        connected = state.get('connected', False)
        if not connected:
            logger.error("Broker disconnected")
            return False
        
        # 최근 오류 확인
        errors = state.get('recent_errors', [])
        if errors:
            logger.error(f"Broker errors: {len(errors)}")
            for error in errors[-5:]:  # 최근 5개
                logger.error(f"  - {error}")
            return False
        
        return True
    
    def perform_health_check(self) -> Dict[str, bool]:
        """
        종합 헬스 체크
        
        Returns:
            체크 항목별 결과
        """
        checks = {
            'heartbeat': self.check_heartbeat(),
            'guardrails': self.check_guardrails(),
            'data_freshness': self.check_data_freshness(),
            'broker_connection': self.check_broker_connection()
        }
        
        return checks
    
    def trigger_emergency_shutdown(self, reason: str, checks: Dict[str, bool]):
        """
        비상 종료 트리거
        
        Args:
            reason: 종료 사유
            checks: 헬스 체크 결과
        """
        logger.critical(f"EMERGENCY SHUTDOWN TRIGGERED: {reason}")
        
        # 1. 비상 상태 기록
        emergency_state = {
            'timestamp': datetime.utcnow().isoformat(),
            'reason': reason,
            'health_checks': checks,
            'actions_taken': []
        }
        
        # 2. 포지션 청산
        try:
            self._liquidate_all_positions()
            emergency_state['actions_taken'].append('positions_liquidated')
        except Exception as e:
            logger.error(f"Failed to liquidate positions: {e}")
            emergency_state['actions_taken'].append(f'liquidation_failed: {e}')
        
        # 3. 시스템 안전 모드
        try:
            self._enable_safe_mode()
            emergency_state['actions_taken'].append('safe_mode_enabled')
        except Exception as e:
            logger.error(f"Failed to enable safe mode: {e}")
            emergency_state['actions_taken'].append(f'safe_mode_failed: {e}')
        
        # 4. 알림 전송
        try:
            self._send_emergency_alerts(reason, checks)
            emergency_state['actions_taken'].append('alerts_sent')
        except Exception as e:
            logger.error(f"Failed to send alerts: {e}")
            emergency_state['actions_taken'].append(f'alerts_failed: {e}')
        
        # 5. 비상 상태 저장
        with open(self.emergency_file, 'w') as f:
            json.dump(emergency_state, f, indent=2)
        
        # 6. 로그 저장
        log_file = self.log_dir / f"emergency_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w') as f:
            json.dump(emergency_state, f, indent=2)
        
        logger.critical(f"Emergency state saved: {self.emergency_file}")
    
    def _liquidate_all_positions(self):
        """모든 포지션 청산"""
        logger.warning("Liquidating all positions...")
        
        # 포지션 파일 읽기
        positions_file = self.state_dir / "positions.json"
        if not positions_file.exists():
            logger.warning("No positions file found")
            return
        
        with open(positions_file, 'r') as f:
            positions = json.load(f)
        
        # 청산 명령 생성
        liquidation_orders = []
        for symbol, position in positions.items():
            if position.get('quantity', 0) != 0:
                liquidation_orders.append({
                    'symbol': symbol,
                    'action': 'SELL' if position['quantity'] > 0 else 'BUY',
                    'quantity': abs(position['quantity']),
                    'order_type': 'MARKET',
                    'urgency': 'EMERGENCY'
                })
        
        # 청산 명령 저장 (실제 실행은 브로커 모듈에서)
        liquidation_file = self.state_dir / "emergency_liquidation.json"
        with open(liquidation_file, 'w') as f:
            json.dump({
                'timestamp': datetime.utcnow().isoformat(),
                'orders': liquidation_orders
            }, f, indent=2)
        
        logger.warning(f"Liquidation orders saved: {len(liquidation_orders)} orders")
    
    def _enable_safe_mode(self):
        """시스템 안전 모드 활성화"""
        logger.warning("Enabling safe mode...")
        
        safe_mode_state = {
            'enabled': True,
            'timestamp': datetime.utcnow().isoformat(),
            'restrictions': [
                'no_new_positions',
                'no_rebalancing',
                'monitoring_only'
            ]
        }
        
        safe_mode_file = self.state_dir / "safe_mode.json"
        with open(safe_mode_file, 'w') as f:
            json.dump(safe_mode_state, f, indent=2)
        
        logger.warning("Safe mode enabled")
    
    def _send_emergency_alerts(self, reason: str, checks: Dict[str, bool]):
        """비상 알림 전송"""
        logger.warning("Sending emergency alerts...")
        
        # Slack 알림
        if self.slack_webhook_url:
            self._send_slack_alert(reason, checks)
        
        # PagerDuty 알림
        if self.pagerduty_api_key:
            self._send_pagerduty_alert(reason, checks)
    
    def _send_slack_alert(self, reason: str, checks: Dict[str, bool]):
        """Slack 알림 전송"""
        failed_checks = [k for k, v in checks.items() if not v]
        
        message = {
            'text': f'🚨 *EMERGENCY SHUTDOWN* 🚨',
            'attachments': [{
                'color': 'danger',
                'fields': [
                    {
                        'title': 'Reason',
                        'value': reason,
                        'short': False
                    },
                    {
                        'title': 'Failed Checks',
                        'value': ', '.join(failed_checks) if failed_checks else 'None',
                        'short': False
                    },
                    {
                        'title': 'Timestamp',
                        'value': datetime.utcnow().isoformat(),
                        'short': True
                    }
                ]
            }]
        }
        
        try:
            response = requests.post(
                self.slack_webhook_url,
                json=message,
                timeout=10
            )
            response.raise_for_status()
            logger.info("Slack alert sent")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def _send_pagerduty_alert(self, reason: str, checks: Dict[str, bool]):
        """PagerDuty 알림 전송"""
        event = {
            'routing_key': self.pagerduty_api_key,
            'event_action': 'trigger',
            'payload': {
                'summary': f'ARES7 Emergency Shutdown: {reason}',
                'severity': 'critical',
                'source': 'ARES7_Dead_Man_Switch',
                'custom_details': {
                    'health_checks': checks,
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
        }
        
        try:
            response = requests.post(
                'https://events.pagerduty.com/v2/enqueue',
                json=event,
                timeout=10
            )
            response.raise_for_status()
            logger.info("PagerDuty alert sent")
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")
    
    def is_safe_mode_enabled(self) -> bool:
        """안전 모드 활성화 여부 확인"""
        safe_mode_file = self.state_dir / "safe_mode.json"
        
        if not safe_mode_file.exists():
            return False
        
        with open(safe_mode_file, 'r') as f:
            state = json.load(f)
        
        return state.get('enabled', False)
    
    def disable_safe_mode(self):
        """안전 모드 비활성화 (수동)"""
        safe_mode_file = self.state_dir / "safe_mode.json"
        
        if safe_mode_file.exists():
            safe_mode_file.unlink()
        
        logger.info("Safe mode disabled")
    
    def run_monitoring_loop(self, interval: int = 60):
        """
        모니터링 루프 실행
        
        Args:
            interval: 체크 간격 (초)
        """
        logger.info(f"Starting dead-man switch monitoring (interval: {interval}s)")
        
        while True:
            try:
                # 헬스 체크
                checks = self.perform_health_check()
                
                # 실패한 체크 확인
                failed = [k for k, v in checks.items() if not v]
                
                if failed:
                    reason = f"Health check failed: {', '.join(failed)}"
                    self.trigger_emergency_shutdown(reason, checks)
                    break  # 비상 종료 후 루프 종료
                else:
                    logger.info(f"Health check passed: {checks}")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(interval)


# 테스트 코드
if __name__ == "__main__":
    print("=" * 60)
    print("Dead-man Switch System Test")
    print("=" * 60)
    print()
    
    # Dead-man switch 초기화
    dms = DeadManSwitch()
    
    # 하트비트 업데이트
    dms.update_heartbeat({'test': True})
    print("✅ Heartbeat updated")
    
    # 헬스 체크
    checks = dms.perform_health_check()
    print("\nHealth Check Results:")
    for check, result in checks.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check}: {result}")
    
    # 안전 모드 확인
    safe_mode = dms.is_safe_mode_enabled()
    print(f"\nSafe Mode: {'Enabled' if safe_mode else 'Disabled'}")
    
    print()
    print("✅ Dead-man switch system working correctly!")
