"""
ARES Alpha - Monitoring Module (v2.4)
======================================
실시간 대시보드 + Kill Switch 제어

사용법:
    # API 서버 실행
    python main.py
    
    # 또는
    uvicorn monitoring.api:app --host 0.0.0.0 --port 8000

구성:
    - state.py  : 시스템 상태 데이터 클래스
    - store.py  : 상태 저장/로드 (JSON)
    - api.py    : FastAPI REST API
"""

from monitoring.state import (
    SystemState,
    PositionRecord,
    TradeRecord,
    KillSwitchMode,
)

from monitoring.store import (
    load_state,
    save_state,
    set_kill_switch_mode,
)

__all__ = [
    # State
    "SystemState",
    "PositionRecord",
    "TradeRecord",
    "KillSwitchMode",
    # Store
    "load_state",
    "save_state",
    "set_kill_switch_mode",
]
