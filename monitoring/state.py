# monitoring/state.py
"""
ARES Alpha - System State (v2.4)
=================================
프론트엔드 v2.4에 맞춘 데이터 모델

주요 필드:
- shares, cur_price: 포지션 상세
- month_pnl, month_return: 월간 수익
- times: 차트 X축 시간 라벨
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Literal, Dict, Any
from datetime import datetime

KillSwitchMode = Literal["RUNNING", "STOP_NEW_ORDERS", "FULL_EXIT"]


@dataclass
class PositionRecord:
    """포지션 레코드"""
    symbol: str
    weight: float       # 비중 (0.0 ~ 1.0)
    shares: float       # 보유 수량
    price: float        # 현재가 (USD)
    value: float        # 평가금액 (USD)
    pnl_pct: float      # 수익률


@dataclass
class TradeRecord:
    """체결 레코드"""
    time: str           # HH:MM:SS
    symbol: str
    side: str           # BUY / SELL
    price: float        # 체결가
    cur_price: float    # 현재가 (ROI 계산용)
    roi: float          # 수익률


@dataclass
class SystemState:
    """
    ARES Alpha 시스템 전체 상태 (v2.4)
    
    프론트엔드 대시보드에 표시되는 모든 정보를 담는 컨테이너
    """
    # --- 1. 기본 정보 ---
    timestamp: str = ""           # ISO Format String
    usd_krw: float = 1435.0       # 환율

    # --- 2. 상단 Hero Section (4 Columns) ---
    equity: float = 0.0           # 총 자산 (USD)
    cum_pnl: float = 0.0          # Total P&L (USD)
    todays_pnl: float = 0.0       # Today P&L (USD)
    todays_return: float = 0.0    # Today %
    month_pnl: float = 0.0        # This Month P&L (USD)
    month_return: float = 0.0     # This Month %

    # --- 3. Chart Data ---
    equity_curve: List[float] = field(default_factory=list)  # Y축 데이터
    times: List[str] = field(default_factory=list)           # X축 시간 라벨

    # --- 4. Risk Metrics ---
    current_dd: float = 0.0       # MDD
    current_leverage: float = 1.0 # 레버리지
    net_exposure: float = 0.0     # 순노출
    gross_exposure: float = 0.0   # 총노출
    regime: str = "NEUTRAL"       # BULL / BEAR / NEUTRAL / HIGH_VOL
    vix_guard_on: bool = False    # VIX Guard

    # --- 5. Tables ---
    top_positions: List[Dict[str, Any]] = field(default_factory=list)
    recent_trades: List[Dict[str, Any]] = field(default_factory=list)

    # --- 6. Control ---
    kill_switch: KillSwitchMode = "RUNNING"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemState":
        """딕셔너리에서 SystemState 생성"""
        # 중첩 객체 복원 로직
        pos_data = data.get('top_positions', [])
        trades_data = data.get('recent_trades', [])
        
        # 원본 딕셔너리에서 리스트 제거 후 복원
        clean_data = {k: v for k, v in data.items() if k not in ['top_positions', 'recent_trades']}
        
        state = cls(**clean_data)
        state.top_positions = pos_data
        state.recent_trades = trades_data
        return state

    @classmethod
    def now_default(cls) -> "SystemState":
        """서버 시작 시 보여줄 초기 Mock 데이터 (프론트엔드 v2.4 디자인 검증용)"""
        now = datetime.now()
        return cls(
            timestamp=now.isoformat(),
            usd_krw=1435.0,
            equity=108500.50,
            cum_pnl=8500.50,
            todays_pnl=1250.20,
            todays_return=0.0116,
            month_pnl=4250.80,
            month_return=0.041,
            equity_curve=[105000, 105500, 104800, 106000, 107200, 106800, 108500],
            times=['09:30', '10:30', '11:30', '12:30', '13:30', '14:30', '15:30'],
            current_dd=-0.045,
            current_leverage=1.35,
            net_exposure=0.65,
            gross_exposure=0.9,
            regime="BULL",
            vix_guard_on=False,
            top_positions=[
                {"symbol": "TQQQ", "weight": 0.35, "shares": 607, "price": 62.50, "value": 37975, "pnl_pct": 0.15},
                {"symbol": "SOXL", "weight": 0.25, "shares": 793, "price": 34.20, "value": 27125, "pnl_pct": -0.05},
                {"symbol": "NVDA", "weight": 0.20, "shares": 23, "price": 920.00, "value": 21700, "pnl_pct": 0.42},
                {"symbol": "MSFT", "weight": 0.10, "shares": 26, "price": 415.50, "value": 10850, "pnl_pct": 0.02},
                {"symbol": "AAPL", "weight": 0.10, "shares": 62, "price": 175.20, "value": 10850, "pnl_pct": -0.01},
            ],
            recent_trades=[
                {"time": now.strftime("%H:%M:%S"), "symbol": "TQQQ", "side": "BUY", "price": 62.10, "cur_price": 62.50, "roi": 0.0064},
                {"time": now.strftime("%H:%M:%S"), "symbol": "SOXL", "side": "SELL", "price": 34.50, "cur_price": 34.20, "roi": -0.0087},
                {"time": "11:10:05", "symbol": "NVDA", "side": "BUY", "price": 910.00, "cur_price": 920.00, "roi": 0.0109},
                {"time": "10:05:22", "symbol": "MSFT", "side": "BUY", "price": 412.00, "cur_price": 415.50, "roi": 0.0085},
            ],
            kill_switch="RUNNING"
        )
