"""
Live Orchestrator
==================
실시간 자율 거래 루프

데이터 → 전략 → 리스크 → 브로커 전체 파이프라인 오케스트레이션.

Mode:
- SHADOW: 실제 주문 없이 로깅만
- LIVE: 실제 주문 실행

Kill Switch:
- RUNNING: 정상 운영
- STOP_NEW_ORDERS: 신규 주문 중지
- FULL_EXIT: 전체 청산 후 중지
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from core.interfaces import (
    IDataProvider,
    IStrategyEngine,
    IBroker,
    IRiskManager,
    Signal,
    Order,
    Position,
    PortfolioState,
    RiskMetrics,
    Regime,
)
from core.utils import get_logger, load_config
from engines.aresx_v110.execution_engine import ExecutionEngine

# Monitoring imports
try:
    from monitoring.store import load_state, save_state, get_kill_switch
    from monitoring.state import SystemState, PositionInfo, TradeInfo, BrokerStatus
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

logger = get_logger(__name__)


class OrchestratorMode(Enum):
    """Orchestrator 실행 모드"""
    SHADOW = "SHADOW"   # 시뮬레이션만 (주문 없음)
    LIVE = "LIVE"       # 실제 주문 실행


@dataclass
class OrchestratorConfig:
    """Orchestrator 설정"""
    mode: OrchestratorMode = OrchestratorMode.SHADOW
    update_interval_seconds: int = 60
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    rebalance_time: str = "15:30"       # HH:MM (market time)
    min_trade_value: float = 1000
    max_daily_trades: int = 100
    enable_circuit_breaker: bool = True


@dataclass
class OrchestratorState:
    """Orchestrator 상태"""
    is_running: bool = False
    last_data_update: Optional[datetime] = None
    last_signal_generation: Optional[datetime] = None
    last_rebalance: Optional[datetime] = None
    daily_trades: int = 0
    errors: List[str] = field(default_factory=list)


class LiveOrchestrator:
    """
    Live Trading Orchestrator
    
    실시간 자율 거래 시스템의 핵심 오케스트레이터.
    
    Pipeline:
    1. Data Update: 최신 가격/펀더멘탈 데이터 수집
    2. Regime Detection: 시장 레짐 판별
    3. Signal Generation: 전략 시그널 생성
    4. Risk Management: 리스크 필터 적용
    5. Order Execution: 브로커 주문 실행
    """
    
    def __init__(
        self,
        strategy: IStrategyEngine,
        broker: IBroker,
        data_providers: Dict[str, IDataProvider],
        risk_manager: IRiskManager,
        config: Optional[OrchestratorConfig] = None,
        execution_engine: Optional[ExecutionEngine] = None,
    ):
        """
        Initialize orchestrator
        
        Args:
            strategy: Trading strategy engine
            broker: Broker client
            data_providers: Dict of data providers
            risk_manager: Risk manager
            config: Orchestrator configuration
            execution_engine: Order execution engine
        """
        self.strategy = strategy
        self.broker = broker
        self.data_providers = data_providers
        self.risk_manager = risk_manager
        self.config = config or OrchestratorConfig()
        self.execution_engine = execution_engine or ExecutionEngine(broker=broker)
        
        # State
        self.state = OrchestratorState()
        self._portfolio_state: Optional[PortfolioState] = None
        self._risk_metrics: Optional[RiskMetrics] = None
        self._current_signals: List[Signal] = []
        self._returns_history: pd.Series = pd.Series(dtype=float)
        
        # Control
        self._stop_event = asyncio.Event()
    
    async def start(self) -> None:
        """Start orchestrator loop"""
        logger.info(f"Starting orchestrator in {self.config.mode.value} mode")
        
        # Connect broker
        if not await self.broker.connect():
            raise ConnectionError("Failed to connect to broker")
        
        # Connect data providers
        for name, provider in self.data_providers.items():
            await provider.connect()
            logger.info(f"Data provider {name} connected")
        
        self.state.is_running = True
        self._stop_event.clear()
        
        # Initial sync
        await self._sync_portfolio()
        
        # Main loop
        while not self._stop_event.is_set():
            try:
                await self._run_cycle()
            except Exception as e:
                logger.error(f"Orchestrator cycle error: {e}")
                self.state.errors.append(f"{datetime.now()}: {e}")
                
                # Keep only recent errors
                self.state.errors = self.state.errors[-100:]
            
            # Wait for next cycle
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.config.update_interval_seconds
                )
            except asyncio.TimeoutError:
                pass
        
        logger.info("Orchestrator stopped")
    
    async def stop(self) -> None:
        """Stop orchestrator"""
        logger.info("Stopping orchestrator...")
        self._stop_event.set()
        self.state.is_running = False
        
        # Disconnect
        await self.broker.disconnect()
        for provider in self.data_providers.values():
            await provider.disconnect()
    
    async def _run_cycle(self) -> None:
        """Run one orchestration cycle"""
        now = datetime.now()
        
        # 0. Check Kill Switch
        allow_new_orders = True
        if MONITORING_AVAILABLE:
            kill_switch = get_kill_switch()
            
            if kill_switch == "FULL_EXIT":
                logger.warning("Kill Switch: FULL_EXIT - Exiting all positions")
                await self._exit_all_positions()
                await self._update_monitoring_state()
                return
            
            elif kill_switch == "STOP_NEW_ORDERS":
                logger.info("Kill Switch: STOP_NEW_ORDERS - No new orders will be placed")
                allow_new_orders = False
        
        # 1. Update data
        await self._update_data()
        self.state.last_data_update = now
        
        # 2. Sync portfolio
        await self._sync_portfolio()
        
        # 3. Update risk metrics
        await self._update_risk_metrics()
        
        # 4. Check if rebalance is due (only if new orders allowed)
        if allow_new_orders and self._should_rebalance(now):
            await self._rebalance()
            self.state.last_rebalance = now
        
        # 5. Check circuit breaker
        if self.config.enable_circuit_breaker:
            if self._check_circuit_breaker():
                logger.warning("Circuit breaker triggered!")
                # Update monitoring state even if circuit breaker is active
                if MONITORING_AVAILABLE:
                    await self._update_monitoring_state()
                return
        
        # 6. Update monitoring state
        if MONITORING_AVAILABLE:
            await self._update_monitoring_state()
        
        logger.debug(f"Cycle completed at {now}")
    
    async def _update_data(self) -> None:
        """Update market data"""
        # Get price provider
        price_provider = self.data_providers.get("polygon") or self.data_providers.get("prices")
        
        if price_provider:
            # Get universe from strategy
            universe = getattr(self.strategy, "_universe", [])
            
            if universe:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=400)  # Need history for indicators
                
                prices = await price_provider.get_prices(
                    symbols=universe,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not prices.empty:
                    self.strategy.update_data(prices)
    
    async def _sync_portfolio(self) -> None:
        """Sync portfolio state from broker"""
        positions = await self.broker.get_positions()
        cash = await self.broker.get_cash_balance()
        
        self._portfolio_state = PortfolioState(
            cash=cash,
            positions=positions,
            timestamp=datetime.now()
        )
        self._portfolio_state.update_values()
        
        logger.debug(f"Portfolio synced: ${self._portfolio_state.total_value:,.2f}")
    
    async def _update_risk_metrics(self) -> None:
        """Update risk metrics"""
        if self._returns_history.empty:
            self._risk_metrics = RiskMetrics()
            return
        
        self._risk_metrics = self.risk_manager.update(
            returns=self._returns_history,
            portfolio_state=self._portfolio_state
        )
    
    def _should_rebalance(self, now: datetime) -> bool:
        """Check if rebalance is due"""
        if self.state.last_rebalance is None:
            return True
        
        # Parse rebalance time
        hour, minute = map(int, self.config.rebalance_time.split(":"))
        
        if now.hour == hour and now.minute >= minute:
            # Check frequency
            if self.config.rebalance_frequency == "daily":
                return self.state.last_rebalance.date() < now.date()
            elif self.config.rebalance_frequency == "weekly":
                days_since = (now - self.state.last_rebalance).days
                return days_since >= 7 or now.weekday() == 4  # Friday
            elif self.config.rebalance_frequency == "monthly":
                return self.state.last_rebalance.month != now.month
        
        return False
    
    async def _rebalance(self) -> None:
        """Execute rebalance"""
        logger.info("Starting rebalance...")
        
        # 1. Generate signals
        signals = self.strategy.generate_signals(
            as_of=datetime.now(),
            portfolio_state=self._portfolio_state,
            risk_metrics=self._risk_metrics
        )
        
        self.state.last_signal_generation = datetime.now()
        self._current_signals = signals
        
        logger.info(f"Generated {len(signals)} signals")
        
        if not signals:
            return
        
        # 2. Apply risk limits
        adjusted_signals = self.risk_manager.apply_risk_limits(
            signals=signals,
            portfolio_state=self._portfolio_state,
            risk_metrics=self._risk_metrics
        )
        
        logger.info(f"Adjusted to {len(adjusted_signals)} signals after risk limits")
        
        # 3. Get current prices
        current_prices = {}
        for signal in adjusted_signals:
            quote = await self.broker.get_quote(signal.symbol)
            current_prices[signal.symbol] = quote.get("last", 0)
        
        # 4. Convert to orders
        orders = self.execution_engine.signals_to_orders(
            signals=adjusted_signals,
            current_positions=self._portfolio_state.positions,
            portfolio_value=self._portfolio_state.total_value,
            current_prices=current_prices
        )
        
        logger.info(f"Generated {len(orders)} orders")
        
        # 5. Execute orders
        if self.config.mode == OrchestratorMode.LIVE:
            results = await self.execution_engine.execute_orders(orders)
            
            filled = sum(1 for r in results if r.status.value == "FILLED")
            logger.info(f"Executed {filled}/{len(orders)} orders")
            
            self.state.daily_trades += len(orders)
        else:
            # Shadow mode: log only
            for order in orders:
                logger.info(
                    f"[SHADOW] Would execute: {order.side.value} "
                    f"{order.quantity} {order.symbol} @ {order.price or 'MARKET'}"
                )
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker should be triggered"""
        if self._risk_metrics is None:
            return False
        
        return self._risk_metrics.cb_active
    
    async def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            "mode": self.config.mode.value,
            "is_running": self.state.is_running,
            "last_data_update": self.state.last_data_update.isoformat() if self.state.last_data_update else None,
            "last_rebalance": self.state.last_rebalance.isoformat() if self.state.last_rebalance else None,
            "daily_trades": self.state.daily_trades,
            "portfolio_value": self._portfolio_state.total_value if self._portfolio_state else 0,
            "current_drawdown": self._risk_metrics.current_drawdown if self._risk_metrics else 0,
            "cb_active": self._risk_metrics.cb_active if self._risk_metrics else False,
            "errors_count": len(self.state.errors),
        }
    
    async def _exit_all_positions(self) -> None:
        """Exit all positions (Kill Switch: FULL_EXIT)"""
        logger.warning("Executing FULL EXIT of all positions...")
        
        if self._portfolio_state is None:
            await self._sync_portfolio()
        
        positions = self._portfolio_state.positions
        
        if not positions:
            logger.info("No positions to exit")
            return
        
        # Create market sell orders for all positions
        orders = []
        for symbol, pos in positions.items():
            if pos.quantity != 0:
                from core.interfaces import Side, OrderType
                import uuid
                
                order = Order(
                    order_id=f"EXIT_{uuid.uuid4().hex[:8]}",
                    symbol=symbol,
                    side=Side.SELL if pos.quantity > 0 else Side.BUY,
                    quantity=abs(pos.quantity),
                    order_type=OrderType.MARKET,
                )
                orders.append(order)
                logger.info(f"Exit order: {order.side.value} {order.quantity} {symbol}")
        
        # Execute exit orders
        if self.config.mode == OrchestratorMode.LIVE:
            for order in orders:
                try:
                    await self.broker.submit_order(order)
                except Exception as e:
                    logger.error(f"Failed to exit {order.symbol}: {e}")
        else:
            logger.info(f"[SHADOW] Would exit {len(orders)} positions")
    
    async def _update_monitoring_state(self) -> None:
        """Update monitoring state for dashboard"""
        if not MONITORING_AVAILABLE:
            return
        
        try:
            state = load_state()
            
            # Timestamp
            state.update_timestamp()
            
            # Portfolio
            if self._portfolio_state:
                state.equity = self._portfolio_state.total_value
                state.cash_balance = self._portfolio_state.cash
                state.current_leverage = self._portfolio_state.leverage
                
                # Calculate returns
                if state.initial_capital > 0:
                    state.cum_return = (state.equity / state.initial_capital) - 1
                    state.cum_pnl = state.equity - state.initial_capital
                
                # Positions
                state.positions = []
                for symbol, pos in self._portfolio_state.positions.items():
                    state.positions.append({
                        "symbol": symbol,
                        "quantity": pos.quantity,
                        "weight": pos.weight,
                        "avg_cost": pos.avg_cost,
                        "current_price": pos.current_price,
                        "market_value": pos.market_value,
                        "unrealized_pnl": pos.unrealized_pnl,
                        "unrealized_pnl_pct": (pos.current_price / pos.avg_cost - 1) if pos.avg_cost > 0 else 0,
                    })
                state.position_count = len(state.positions)
            
            # Risk metrics
            if self._risk_metrics:
                state.current_drawdown = self._risk_metrics.current_drawdown
                state.max_drawdown = self._risk_metrics.max_drawdown
                state.volatility = self._risk_metrics.volatility_20d
                state.sharpe_ratio = self._risk_metrics.sharpe_ratio
                state.sortino_ratio = self._risk_metrics.sortino_ratio
                state.var_95 = self._risk_metrics.var_95
                state.cvar_95 = self._risk_metrics.cvar_95
                state.position_scale = self._risk_metrics.position_scale
                state.cb_active = self._risk_metrics.cb_active
                state.regime = self._risk_metrics.regime.value if hasattr(self._risk_metrics.regime, 'value') else str(self._risk_metrics.regime)
            
            # Broker status
            state.brokers = [{
                "name": self.broker.name if hasattr(self.broker, 'name') else "BROKER",
                "connected": self.broker.is_connected if hasattr(self.broker, 'is_connected') else True,
            }]
            
            save_state(state)
            
        except Exception as e:
            logger.error(f"Failed to update monitoring state: {e}")


async def main():
    """Main entry point for live trading"""
    import sys
    
    # Parse arguments
    mode = OrchestratorMode.SHADOW
    if "--live" in sys.argv:
        mode = OrchestratorMode.LIVE
        logger.warning("LIVE MODE ENABLED - Real orders will be placed!")
    
    # Load config
    config_path = "config/ares7_qm_turbo_final_251129.yaml"
    config = load_config(config_path)
    
    # Initialize components
    from engines.ares7_qm_regime.strategy import ARES7QMRegimeStrategy
    from risk.aarm_core import TurboAARM
    from brokers.ibkr_client import IBKRClient
    from data.polygon_client import PolygonClient
    
    strategy = ARES7QMRegimeStrategy(config_path)
    broker = IBKRClient(mode="paper")
    data_providers = {"polygon": PolygonClient()}
    risk_manager = TurboAARM()
    
    orchestrator_config = OrchestratorConfig(
        mode=mode,
        update_interval_seconds=60,
        rebalance_frequency="daily",
        rebalance_time="15:30",
    )
    
    orchestrator = LiveOrchestrator(
        strategy=strategy,
        broker=broker,
        data_providers=data_providers,
        risk_manager=risk_manager,
        config=orchestrator_config,
    )
    
    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())
