"""
ARES-Ultimate Core Interfaces
==============================
핵심 추상화 계층: 모든 컴포넌트가 구현해야 할 인터페이스 정의

- IDataProvider: 데이터 제공자 인터페이스
- IStrategyEngine: 전략 엔진 인터페이스
- IBroker: 브로커 인터페이스
- IRiskManager: 리스크 관리 인터페이스
- Signal, Order, Position, Fill: 도메인 객체
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, Sequence

import pandas as pd


# =============================================================================
# Enums
# =============================================================================

class Side(Enum):
    """주문 방향"""
    BUY = "BUY"
    SELL = "SELL"
    
    def __str__(self) -> str:
        return self.value


class OrderType(Enum):
    """주문 유형"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    MOC = "MOC"  # Market on Close
    MOO = "MOO"  # Market on Open
    
    def __str__(self) -> str:
        return self.value


class TimeInForce(Enum):
    """주문 유효 기간"""
    DAY = "DAY"
    GTC = "GTC"      # Good Till Cancel
    IOC = "IOC"      # Immediate or Cancel
    FOK = "FOK"      # Fill or Kill
    GTD = "GTD"      # Good Till Date
    OPG = "OPG"      # At the Opening
    
    def __str__(self) -> str:
        return self.value


class OrderStatus(Enum):
    """주문 상태"""
    PENDING = auto()
    SUBMITTED = auto()
    PARTIAL_FILL = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()


class Regime(Enum):
    """시장 레짐"""
    BULL = "BULL"
    BEAR = "BEAR"
    HIGH_VOL = "HIGH_VOL"
    NEUTRAL = "NEUTRAL"
    
    def __str__(self) -> str:
        return self.value


class SignalType(Enum):
    """시그널 유형"""
    ENTRY = auto()      # 신규 진입
    EXIT = auto()       # 청산
    REBALANCE = auto()  # 리밸런싱
    SCALE_IN = auto()   # 증액
    SCALE_OUT = auto()  # 감액


# =============================================================================
# Domain Objects (Data Classes)
# =============================================================================

@dataclass(frozen=True)
class Signal:
    """
    전략 엔진이 생성하는 시그널
    
    Attributes:
        symbol: 종목 코드 (e.g., "AAPL", "005930")
        side: 매수/매도
        signal_type: 시그널 유형
        target_weight: 목표 포트폴리오 비중 (0.0 ~ 1.0)
        confidence: 신뢰도 (0.0 ~ 1.0)
        timestamp: 시그널 생성 시간
        metadata: 추가 메타데이터 (전략 이름, 스코어 등)
    """
    symbol: str
    side: Side
    signal_type: SignalType
    target_weight: float
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not 0.0 <= self.target_weight <= 1.0:
            raise ValueError(f"target_weight must be [0, 1], got {self.target_weight}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be [0, 1], got {self.confidence}")


@dataclass
class Order:
    """
    브로커에 제출할 주문
    
    Attributes:
        order_id: 내부 주문 ID
        symbol: 종목 코드
        side: 매수/매도
        quantity: 수량
        order_type: 주문 유형
        price: 지정가 (LIMIT, STOP_LIMIT용)
        stop_price: 스탑 가격 (STOP, STOP_LIMIT용)
        time_in_force: 주문 유효 기간
        exchange: 거래소 (e.g., "SMART", "KRX")
        currency: 통화 (e.g., "USD", "KRW")
        status: 주문 상태
        broker_order_id: 브로커 측 주문 ID
        created_at: 생성 시간
        updated_at: 최종 업데이트 시간
    """
    order_id: str
    symbol: str
    side: Side
    quantity: float
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    exchange: str = "SMART"
    currency: str = "USD"
    status: OrderStatus = OrderStatus.PENDING
    broker_order_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Fill:
    """
    체결 정보
    
    Attributes:
        fill_id: 체결 ID
        order_id: 관련 주문 ID
        symbol: 종목 코드
        side: 매수/매도
        quantity: 체결 수량
        price: 체결 가격
        commission: 수수료
        timestamp: 체결 시간
    """
    fill_id: str
    order_id: str
    symbol: str
    side: Side
    quantity: float
    price: float
    commission: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Position:
    """
    보유 포지션
    
    Attributes:
        symbol: 종목 코드
        quantity: 보유 수량 (양수=롱, 음수=숏)
        avg_cost: 평균 매입 단가
        current_price: 현재가
        market_value: 시장 가치
        unrealized_pnl: 미실현 손익
        realized_pnl: 실현 손익
        weight: 포트폴리오 비중
    """
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    weight: float = 0.0
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
    
    def update_market_data(self, current_price: float, total_portfolio_value: float) -> None:
        """시장 데이터 업데이트"""
        self.current_price = current_price
        self.market_value = self.quantity * current_price
        self.unrealized_pnl = (current_price - self.avg_cost) * self.quantity
        if total_portfolio_value > 0:
            self.weight = abs(self.market_value) / total_portfolio_value


@dataclass
class PortfolioState:
    """
    포트폴리오 상태
    
    Attributes:
        cash: 현금
        positions: 포지션 딕셔너리
        total_value: 총 자산 가치
        leverage: 레버리지
        timestamp: 상태 시점
    """
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    total_value: float = 0.0
    leverage: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def update_values(self) -> None:
        """포트폴리오 가치 재계산"""
        positions_value = sum(abs(p.market_value) for p in self.positions.values())
        self.total_value = self.cash + positions_value
        
        if self.total_value > 0:
            # 레버리지 = 총 노출 / 순자산
            gross_exposure = sum(abs(p.market_value) for p in self.positions.values())
            self.leverage = gross_exposure / self.total_value if self.total_value > 0 else 0
            
            # 각 포지션 비중 업데이트
            for pos in self.positions.values():
                pos.weight = abs(pos.market_value) / self.total_value


@dataclass
class RiskMetrics:
    """
    리스크 지표
    
    Attributes:
        current_drawdown: 현재 드로다운
        max_drawdown: 최대 드로다운
        volatility_20d: 20일 변동성
        var_95: 95% VaR
        cvar_95: 95% CVaR
        sharpe_ratio: 샤프 비율
        sortino_ratio: 소르티노 비율
        regime: 현재 레짐
    """
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    volatility_20d: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    regime: Regime = Regime.NEUTRAL
    position_scale: float = 1.0
    cb_active: bool = False


# =============================================================================
# Abstract Interfaces
# =============================================================================

class IDataProvider(ABC):
    """
    데이터 제공자 인터페이스
    
    모든 데이터 클라이언트가 구현해야 할 메서드 정의
    """
    
    @abstractmethod
    async def connect(self) -> bool:
        """연결 초기화"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """연결 종료"""
        pass
    
    @abstractmethod
    async def get_prices(
        self,
        symbols: Sequence[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        가격 데이터 조회
        
        Args:
            symbols: 종목 코드 리스트
            start_date: 시작일
            end_date: 종료일
            interval: 봉 간격 ("1d", "1h", "5m" 등)
        
        Returns:
            MultiIndex DataFrame with (date, symbol) index
            Columns: open, high, low, close, volume
        """
        pass
    
    @abstractmethod
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """최신 가격 조회"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """데이터 소스 이름"""
        pass


class IFundamentalProvider(IDataProvider):
    """재무 데이터 제공자 인터페이스"""
    
    @abstractmethod
    async def get_fundamentals(
        self,
        symbols: Sequence[str],
        metrics: Sequence[str],
        as_of_date: datetime,
        pit_delay_days: int = 90
    ) -> pd.DataFrame:
        """
        재무 데이터 조회 (Point-in-Time)
        
        Args:
            symbols: 종목 코드 리스트
            metrics: 재무 지표 리스트 (e.g., ["roe", "roic"])
            as_of_date: 기준일
            pit_delay_days: PIT 지연 일수
        
        Returns:
            DataFrame with symbol index, metric columns
        """
        pass


class IMacroProvider(IDataProvider):
    """거시경제 데이터 제공자 인터페이스"""
    
    @abstractmethod
    async def get_macro_series(
        self,
        series_ids: Sequence[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        거시경제 시계열 조회
        
        Args:
            series_ids: 시리즈 ID 리스트 (e.g., ["VIXCLS", "T10Y2Y"])
            start_date: 시작일
            end_date: 종료일
        
        Returns:
            DataFrame with date index, series columns
        """
        pass


class INewsProvider(IDataProvider):
    """뉴스/이벤트 데이터 제공자 인터페이스"""
    
    @abstractmethod
    async def get_news(
        self,
        query: str,
        symbols: Optional[Sequence[str]] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        뉴스 검색
        
        Returns:
            List of news articles with sentiment, timestamp, etc.
        """
        pass
    
    @abstractmethod
    async def get_sentiment_score(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> float:
        """
        종목별 센티먼트 점수 (-1 to 1)
        """
        pass
    
    @abstractmethod
    async def get_event_risk_flag(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> bool:
        """
        이벤트 리스크 플래그 (고위험 이벤트 감지)
        """
        pass


class IStrategyEngine(ABC):
    """
    전략 엔진 인터페이스
    
    모든 전략 엔진이 구현해야 할 메서드 정의
    """
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """전략 초기화"""
        pass
    
    @abstractmethod
    def generate_signals(
        self,
        as_of: datetime,
        portfolio_state: PortfolioState,
        risk_metrics: RiskMetrics
    ) -> List[Signal]:
        """
        시그널 생성
        
        Args:
            as_of: 기준 시점
            portfolio_state: 현재 포트폴리오 상태
            risk_metrics: 현재 리스크 지표
        
        Returns:
            시그널 리스트
        """
        pass
    
    @abstractmethod
    def update_data(self, prices: pd.DataFrame, fundamentals: Optional[pd.DataFrame] = None) -> None:
        """데이터 업데이트"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """전략 이름"""
        pass


class IBroker(ABC):
    """
    브로커 인터페이스
    
    모든 브로커 클라이언트가 구현해야 할 메서드 정의
    """
    
    @abstractmethod
    async def connect(self) -> bool:
        """브로커 연결"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """브로커 연결 종료"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """계좌 정보 조회"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> Dict[str, Position]:
        """보유 포지션 조회"""
        pass
    
    @abstractmethod
    async def get_cash_balance(self) -> float:
        """현금 잔고 조회"""
        pass
    
    @abstractmethod
    async def submit_order(self, order: Order) -> Order:
        """주문 제출"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """주문 취소"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """주문 상태 조회"""
        pass
    
    @abstractmethod
    async def get_quote(self, symbol: str) -> Dict[str, float]:
        """
        호가 조회
        
        Returns:
            {"bid": float, "ask": float, "last": float, "volume": int}
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """브로커 이름"""
        pass
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """연결 상태"""
        pass


class IRiskManager(ABC):
    """
    리스크 관리자 인터페이스
    """
    
    @abstractmethod
    def update(
        self,
        returns: pd.Series,
        portfolio_state: PortfolioState
    ) -> RiskMetrics:
        """
        리스크 지표 업데이트
        
        Args:
            returns: 수익률 시리즈
            portfolio_state: 포트폴리오 상태
        
        Returns:
            업데이트된 리스크 지표
        """
        pass
    
    @abstractmethod
    def calculate_position_scale(
        self,
        risk_metrics: RiskMetrics,
        regime: Regime
    ) -> float:
        """
        포지션 스케일 계산 (AARM 등)
        
        Returns:
            스케일 팩터 (0.0 ~ max_leverage)
        """
        pass
    
    @abstractmethod
    def check_circuit_breaker(
        self,
        risk_metrics: RiskMetrics
    ) -> bool:
        """
        서킷 브레이커 체크
        
        Returns:
            True if CB should be triggered
        """
        pass
    
    @abstractmethod
    def apply_risk_limits(
        self,
        signals: List[Signal],
        portfolio_state: PortfolioState,
        risk_metrics: RiskMetrics
    ) -> List[Signal]:
        """
        리스크 한도 적용 (포지션 사이즈, 섹터 노출 등)
        
        Returns:
            조정된 시그널 리스트
        """
        pass


# =============================================================================
# Protocol Definitions (Structural Subtyping)
# =============================================================================

class Connectable(Protocol):
    """연결 가능한 객체"""
    async def connect(self) -> bool: ...
    async def disconnect(self) -> None: ...
    @property
    def is_connected(self) -> bool: ...


class Configurable(Protocol):
    """설정 가능한 객체"""
    def initialize(self, config: Dict[str, Any]) -> None: ...


# =============================================================================
# Type Aliases
# =============================================================================

SignalList = List[Signal]
PositionDict = Dict[str, Position]
PriceDataFrame = pd.DataFrame  # MultiIndex (date, symbol), columns: OHLCV
