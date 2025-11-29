"""
ARES-X V110 Execution Engine
=============================
IBKR/KIS 브로커 연동 실행 로직

V110 코드에서 추출한 실행 관련 기능:
- 주문 생성/검증
- 슬리피지 추정
- 실행 알고리즘 (TWAP, VWAP, etc.)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from core.interfaces import (
    Order,
    OrderType,
    OrderStatus,
    Side,
    TimeInForce,
    Signal,
    Position,
    IBroker,
)
from core.utils import get_logger

logger = get_logger(__name__)


class ExecutionAlgo(Enum):
    """실행 알고리즘"""
    MARKET = "MARKET"       # 즉시 시장가
    TWAP = "TWAP"          # Time Weighted Average Price
    VWAP = "VWAP"          # Volume Weighted Average Price
    ICEBERG = "ICEBERG"     # 빙산 주문
    ADAPTIVE = "ADAPTIVE"   # 적응형


@dataclass
class ExecutionConfig:
    """실행 설정"""
    algo: ExecutionAlgo = ExecutionAlgo.MARKET
    urgency: float = 0.5           # 0=낮음, 1=높음
    max_participation: float = 0.1  # 최대 거래량 비중
    slippage_limit_bps: float = 50  # 슬리피지 한도 (bps)
    min_trade_value: float = 1000   # 최소 거래 금액
    split_count: int = 1            # 분할 횟수


@dataclass
class ExecutionResult:
    """실행 결과"""
    order_id: str
    symbol: str
    side: Side
    requested_qty: float
    filled_qty: float
    avg_price: float
    slippage_bps: float
    commission: float
    status: OrderStatus
    fills: List[Dict[str, Any]] = field(default_factory=list)
    execution_time_ms: float = 0.0


class ExecutionEngine:
    """
    Execution Engine
    
    시그널 → 주문 변환 및 실행 관리
    """
    
    def __init__(
        self,
        broker: Optional[IBroker] = None,
        config: Optional[ExecutionConfig] = None
    ):
        """
        Initialize execution engine
        
        Args:
            broker: Broker client (IBKR or KIS)
            config: Execution configuration
        """
        self.broker = broker
        self.config = config or ExecutionConfig()
        
        # Order tracking
        self._pending_orders: Dict[str, Order] = {}
        self._execution_results: List[ExecutionResult] = []
        self._order_counter = 0
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        self._order_counter += 1
        return f"ARES-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._order_counter:04d}"
    
    def signals_to_orders(
        self,
        signals: List[Signal],
        current_positions: Dict[str, Position],
        portfolio_value: float,
        current_prices: Dict[str, float]
    ) -> List[Order]:
        """
        Convert signals to orders
        
        Args:
            signals: List of trading signals
            current_positions: Current positions
            portfolio_value: Total portfolio value
            current_prices: Current prices per symbol
        
        Returns:
            List of orders to execute
        """
        orders = []
        
        for signal in signals:
            order = self._signal_to_order(
                signal=signal,
                current_position=current_positions.get(signal.symbol),
                portfolio_value=portfolio_value,
                current_price=current_prices.get(signal.symbol, 0.0)
            )
            
            if order is not None:
                orders.append(order)
        
        return orders
    
    def _signal_to_order(
        self,
        signal: Signal,
        current_position: Optional[Position],
        portfolio_value: float,
        current_price: float
    ) -> Optional[Order]:
        """Convert single signal to order"""
        if current_price <= 0:
            logger.warning(f"Invalid price for {signal.symbol}: {current_price}")
            return None
        
        # Calculate target value
        target_value = portfolio_value * signal.target_weight
        
        # Current value
        current_value = 0.0
        current_qty = 0.0
        if current_position:
            current_value = current_position.market_value
            current_qty = current_position.quantity
        
        # Delta
        delta_value = target_value - current_value
        
        # Skip small trades
        if abs(delta_value) < self.config.min_trade_value:
            return None
        
        # Calculate quantity
        delta_qty = delta_value / current_price
        
        # Determine side
        if delta_qty > 0:
            side = Side.BUY
            qty = abs(delta_qty)
        else:
            side = Side.SELL
            qty = abs(delta_qty)
        
        # Round to integer for stocks
        qty = round(qty)
        
        if qty <= 0:
            return None
        
        # Create order
        order = Order(
            order_id=self._generate_order_id(),
            symbol=signal.symbol,
            side=side,
            quantity=float(qty),
            order_type=OrderType.MARKET if self.config.algo == ExecutionAlgo.MARKET else OrderType.LIMIT,
            price=current_price if self.config.algo != ExecutionAlgo.MARKET else None,
            time_in_force=TimeInForce.DAY,
            status=OrderStatus.PENDING,
            metadata={
                "signal_weight": signal.target_weight,
                "strategy": signal.metadata.get("strategy"),
            }
        )
        
        return order
    
    def validate_order(self, order: Order) -> tuple[bool, str]:
        """
        Validate order before submission
        
        Returns:
            (is_valid, error_message)
        """
        # Check quantity
        if order.quantity <= 0:
            return False, f"Invalid quantity: {order.quantity}"
        
        # Check price for limit orders
        if order.order_type == OrderType.LIMIT and (order.price is None or order.price <= 0):
            return False, f"Invalid price for limit order: {order.price}"
        
        # Check symbol
        if not order.symbol or len(order.symbol) > 10:
            return False, f"Invalid symbol: {order.symbol}"
        
        return True, ""
    
    async def execute_order(self, order: Order) -> ExecutionResult:
        """
        Execute a single order
        
        Args:
            order: Order to execute
        
        Returns:
            ExecutionResult
        """
        start_time = datetime.now()
        
        # Validate
        is_valid, error = self.validate_order(order)
        if not is_valid:
            logger.error(f"Order validation failed: {error}")
            return ExecutionResult(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                requested_qty=order.quantity,
                filled_qty=0,
                avg_price=0,
                slippage_bps=0,
                commission=0,
                status=OrderStatus.REJECTED,
            )
        
        # Track pending
        self._pending_orders[order.order_id] = order
        
        try:
            if self.broker is None:
                # Simulation mode
                result = await self._simulate_execution(order)
            else:
                # Live execution
                result = await self._live_execution(order)
            
            # Calculate execution time
            result.execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Store result
            self._execution_results.append(result)
            
            # Remove from pending
            del self._pending_orders[order.order_id]
            
            return result
            
        except Exception as e:
            logger.error(f"Execution error for {order.order_id}: {e}")
            return ExecutionResult(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                requested_qty=order.quantity,
                filled_qty=0,
                avg_price=0,
                slippage_bps=0,
                commission=0,
                status=OrderStatus.REJECTED,
            )
    
    async def _simulate_execution(self, order: Order) -> ExecutionResult:
        """Simulate order execution"""
        # Simulate slippage
        base_slippage_bps = 5.0
        slippage_multiplier = 1.0 + np.random.normal(0, 0.5)
        slippage_bps = base_slippage_bps * slippage_multiplier
        
        # Calculate fill price
        base_price = order.price if order.price else 100.0  # Default for simulation
        slippage_factor = 1 + (slippage_bps / 10000) * (1 if order.side == Side.BUY else -1)
        fill_price = base_price * slippage_factor
        
        # Simulate commission ($0.005 per share)
        commission = order.quantity * 0.005
        
        # Simulate delay
        await asyncio.sleep(0.1)
        
        return ExecutionResult(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            requested_qty=order.quantity,
            filled_qty=order.quantity,  # Full fill in simulation
            avg_price=fill_price,
            slippage_bps=slippage_bps,
            commission=commission,
            status=OrderStatus.FILLED,
            fills=[{
                "qty": order.quantity,
                "price": fill_price,
                "timestamp": datetime.now().isoformat(),
            }]
        )
    
    async def _live_execution(self, order: Order) -> ExecutionResult:
        """Execute order through live broker"""
        if not self.broker.is_connected:
            await self.broker.connect()
        
        # Submit order
        submitted_order = await self.broker.submit_order(order)
        
        # Wait for fill (with timeout)
        timeout = 30  # seconds
        start = datetime.now()
        
        while (datetime.now() - start).seconds < timeout:
            status = await self.broker.get_order_status(submitted_order.order_id)
            
            if status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
                break
            
            await asyncio.sleep(0.5)
        
        # Get final status
        # (In real implementation, would get fill details from broker)
        
        return ExecutionResult(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            requested_qty=order.quantity,
            filled_qty=order.quantity,  # Simplified
            avg_price=order.price or 0,
            slippage_bps=0,  # Would calculate from actual fills
            commission=0,    # Would get from broker
            status=status,
        )
    
    async def execute_orders(self, orders: List[Order]) -> List[ExecutionResult]:
        """Execute multiple orders"""
        results = []
        
        for order in orders:
            result = await self.execute_order(order)
            results.append(result)
            
            # Small delay between orders
            await asyncio.sleep(0.05)
        
        logger.info(f"Executed {len(results)} orders")
        return results
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        if order_id not in self._pending_orders:
            return False
        
        if self.broker:
            success = await self.broker.cancel_order(order_id)
            if success:
                del self._pending_orders[order_id]
            return success
        else:
            del self._pending_orders[order_id]
            return True
    
    async def cancel_all_orders(self) -> int:
        """Cancel all pending orders"""
        cancelled = 0
        
        for order_id in list(self._pending_orders.keys()):
            if await self.cancel_order(order_id):
                cancelled += 1
        
        return cancelled
    
    def estimate_slippage(
        self,
        symbol: str,
        quantity: float,
        side: Side,
        avg_daily_volume: float = 1_000_000
    ) -> float:
        """
        Estimate slippage for a trade
        
        Args:
            symbol: Stock symbol
            quantity: Order quantity
            side: Buy or sell
            avg_daily_volume: Average daily volume
        
        Returns:
            Estimated slippage in bps
        """
        # Simple model: slippage increases with participation rate
        participation = quantity / avg_daily_volume
        
        # Base slippage
        base_slippage = 2.0  # 2 bps
        
        # Additional slippage from size
        size_impact = participation * 100  # 1% participation = 1 bp additional
        
        # Urgency factor
        urgency_factor = 1.0 + self.config.urgency * 0.5
        
        estimated = (base_slippage + size_impact) * urgency_factor
        
        return min(estimated, self.config.slippage_limit_bps)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self._execution_results:
            return {}
        
        filled = [r for r in self._execution_results if r.status == OrderStatus.FILLED]
        
        if not filled:
            return {"total_orders": len(self._execution_results), "filled": 0}
        
        avg_slippage = np.mean([r.slippage_bps for r in filled])
        total_commission = sum(r.commission for r in filled)
        avg_execution_time = np.mean([r.execution_time_ms for r in filled])
        
        return {
            "total_orders": len(self._execution_results),
            "filled": len(filled),
            "fill_rate": len(filled) / len(self._execution_results),
            "avg_slippage_bps": avg_slippage,
            "total_commission": total_commission,
            "avg_execution_time_ms": avg_execution_time,
        }
