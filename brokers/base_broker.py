"""
Base Broker
============
IBroker 인터페이스 공통 구현
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.interfaces import (
    IBroker,
    Order,
    OrderStatus,
    Position,
)
from core.utils import get_logger, retry_async

logger = get_logger(__name__)


class BaseBroker(IBroker):
    """
    Base Broker Implementation
    
    IBKR, KIS 등 구체 브로커의 공통 기능 구현.
    """
    
    def __init__(self, name: str, mode: str = "paper"):
        """
        Initialize base broker
        
        Args:
            name: Broker name
            mode: "paper" or "live"
        """
        self._name = name
        self._mode = mode
        self._connected = False
        
        # State
        self._positions: Dict[str, Position] = {}
        self._cash: float = 0.0
        self._orders: Dict[str, Order] = {}
        
        # Callbacks
        self._on_fill_callbacks: List[callable] = []
        self._on_error_callbacks: List[callable] = []
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    @property
    def mode(self) -> str:
        return self._mode
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from broker"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        pass
    
    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        return self._positions.copy()
    
    async def get_cash_balance(self) -> float:
        """Get cash balance"""
        return self._cash
    
    @abstractmethod
    async def submit_order(self, order: Order) -> Order:
        """Submit order to broker"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status"""
        pass
    
    @abstractmethod
    async def get_quote(self, symbol: str) -> Dict[str, float]:
        """Get quote for symbol"""
        pass
    
    def register_fill_callback(self, callback: callable) -> None:
        """Register callback for order fills"""
        self._on_fill_callbacks.append(callback)
    
    def register_error_callback(self, callback: callable) -> None:
        """Register callback for errors"""
        self._on_error_callbacks.append(callback)
    
    async def _notify_fill(self, order: Order, fill_info: Dict[str, Any]) -> None:
        """Notify registered callbacks of fill"""
        for callback in self._on_fill_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(order, fill_info)
                else:
                    callback(order, fill_info)
            except Exception as e:
                logger.error(f"Fill callback error: {e}")
    
    async def _notify_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Notify registered callbacks of error"""
        for callback in self._on_error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error, context)
                else:
                    callback(error, context)
            except Exception as e:
                logger.error(f"Error callback error: {e}")
    
    async def get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        positions_value = sum(
            p.market_value for p in self._positions.values()
        )
        return self._cash + positions_value
    
    async def get_buying_power(self) -> float:
        """Get available buying power"""
        # Base implementation: just cash
        # Subclasses can override for margin
        return self._cash
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            "connected": self._connected,
            "mode": self._mode,
            "positions_count": len(self._positions),
            "pending_orders": len(self._orders),
            "timestamp": datetime.now().isoformat(),
        }


class SimulatedBroker(BaseBroker):
    """
    Simulated Broker for Testing
    
    Full simulation of broker functionality without real connection.
    """
    
    def __init__(
        self,
        initial_cash: float = 1_000_000,
        commission_per_share: float = 0.005,
        slippage_bps: float = 5.0
    ):
        """
        Initialize simulated broker
        
        Args:
            initial_cash: Starting cash
            commission_per_share: Commission per share
            slippage_bps: Simulated slippage in bps
        """
        super().__init__(name="SIMULATED", mode="paper")
        self._cash = initial_cash
        self._commission = commission_per_share
        self._slippage_bps = slippage_bps
        
        # Simulated prices
        self._prices: Dict[str, float] = {}
        self._order_counter = 0
    
    async def connect(self) -> bool:
        """Connect (always succeeds)"""
        self._connected = True
        logger.info("Simulated broker connected")
        return True
    
    async def disconnect(self) -> None:
        """Disconnect"""
        self._connected = False
        logger.info("Simulated broker disconnected")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account info"""
        return {
            "account_id": "SIM-001",
            "type": "simulation",
            "cash": self._cash,
            "portfolio_value": await self.get_portfolio_value(),
            "buying_power": self._cash,
        }
    
    async def submit_order(self, order: Order) -> Order:
        """Submit and immediately fill order (simulation)"""
        # Get price
        price = self._prices.get(order.symbol, order.price or 100.0)
        
        # Apply slippage
        slippage = self._slippage_bps / 10000
        if order.side.value == "BUY":
            fill_price = price * (1 + slippage)
        else:
            fill_price = price * (1 - slippage)
        
        # Calculate cost/proceeds
        value = order.quantity * fill_price
        commission = order.quantity * self._commission
        
        if order.side.value == "BUY":
            total_cost = value + commission
            if total_cost > self._cash:
                order.status = OrderStatus.REJECTED
                return order
            self._cash -= total_cost
            
            # Update position
            if order.symbol in self._positions:
                pos = self._positions[order.symbol]
                new_qty = pos.quantity + order.quantity
                new_cost = (pos.avg_cost * pos.quantity + fill_price * order.quantity) / new_qty
                pos.quantity = new_qty
                pos.avg_cost = new_cost
            else:
                self._positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    avg_cost=fill_price,
                    current_price=fill_price,
                    market_value=value,
                )
        else:  # SELL
            if order.symbol not in self._positions:
                order.status = OrderStatus.REJECTED
                return order
            
            pos = self._positions[order.symbol]
            if pos.quantity < order.quantity:
                order.status = OrderStatus.REJECTED
                return order
            
            self._cash += value - commission
            pos.quantity -= order.quantity
            
            if pos.quantity == 0:
                del self._positions[order.symbol]
        
        # Mark filled
        order.status = OrderStatus.FILLED
        order.broker_order_id = f"SIM-{self._order_counter}"
        self._order_counter += 1
        
        # Notify
        await self._notify_fill(order, {"fill_price": fill_price, "commission": commission})
        
        return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        if order_id in self._orders:
            del self._orders[order_id]
            return True
        return False
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status"""
        if order_id in self._orders:
            return self._orders[order_id].status
        return OrderStatus.FILLED  # Assume filled in simulation
    
    async def get_quote(self, symbol: str) -> Dict[str, float]:
        """Get quote"""
        price = self._prices.get(symbol, 100.0)
        spread = price * 0.001  # 0.1% spread
        
        return {
            "bid": price - spread / 2,
            "ask": price + spread / 2,
            "last": price,
            "volume": 1_000_000,
        }
    
    def set_price(self, symbol: str, price: float) -> None:
        """Set simulated price"""
        self._prices[symbol] = price
        
        # Update position market value
        if symbol in self._positions:
            pos = self._positions[symbol]
            pos.current_price = price
            pos.market_value = pos.quantity * price
            pos.unrealized_pnl = (price - pos.avg_cost) * pos.quantity
