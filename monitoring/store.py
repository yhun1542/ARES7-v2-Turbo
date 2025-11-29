# monitoring/store.py
"""
ARES Alpha - State Store
=========================
시스템 상태 저장/로드 (JSON 파일 기반)
"""

from pathlib import Path
import json
from monitoring.state import SystemState, KillSwitchMode

STATE_FILE = Path("system_state.json")


def load_state() -> SystemState:
    """파일에서 상태 로드, 없으면 기본값 생성"""
    if not STATE_FILE.exists():
        return SystemState.now_default()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return SystemState.from_dict(data)
    except Exception as e:
        print(f"[Store] Error loading state: {e}")
        return SystemState.now_default()


def save_state(state: SystemState):
    """상태를 파일로 저장"""
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[Store] Error saving state: {e}")


def set_kill_switch_mode(mode: KillSwitchMode) -> SystemState:
    """Kill Switch 상태만 변경하고 저장"""
    state = load_state()
    state.kill_switch = mode
    # 실제 운영 시에는 여기서 오케스트레이터에 신호를 보내거나 DB를 업데이트함
    save_state(state)
    return state
