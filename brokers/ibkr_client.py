"""
IBKR Client
============
Interactive Brokers 브로커 클라이언트

ARES-X V110 기반 IBKR 연동 코드.
ib_insync 라이브러리 또는 Client Portal REST API 사용.

환경변수:
- IB_HOST: TWS/Gateway host (default: 127.0.0.1)
- IB_PORT: TWS/Gateway port (default: 7497 paper, 7496 live)
- IB_CLIENT_ID: Client ID (default: 1)
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from brokers.base_broker import BaseBroker
from core.interfaces import (
    Order,
    OrderType,
    OrderStatus,
    Position,
    Side,
    TimeInForce,
)
from core.utils import get_env, get_logger, retry_async

logger = get_logger(__name__)


class IBKRClient(BaseBroker):
    """
    Interactive Brokers Client
    
    Supports:
    - ib_insync library (primary)
    - Client Portal REST API (fallback)
    """
    
    # Side mapping
    SIDE_MAP = {
        Side.BUY: "BUY",
        Side.SELL: "SELL",
    }
    
    # Order type mapping
    ORDER_TYPE_MAP = {
        OrderType.MARKET: "MKT",
        OrderType.LIMIT: "LMT",
        OrderType.STOP: "STP",
        OrderType.STOP_LIMIT: "STP LMT",
        OrderType.MOC: "MOC",
    }
    
    # TIF mapping
    TIF_MAP = {
        TimeInForce.DAY: "DAY",
        TimeInForce.GTC: "GTC",
        TimeInForce.IOC: "IOC",
        TimeInForce.FOK: "FOK",
    }
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
        mode: str = "paper"
    ):
        """
        Initialize IBKR client
        
        Args:
            host: TWS/Gateway host
            port: TWS/Gateway port
            client_id: Client ID
            mode: "paper" or "live"
        """
        super().__init__(name="IBKR", mode=mode)
        
        self._host = host or get_env("IB_HOST", default="127.0.0.1")
        self._port = port or get_env("IB_PORT", default=7497 if mode == "paper" else 7496, cast_type=int)
        self._client_id = client_id or get_env("IB_CLIENT_ID", default=1, cast_type=int)
        
        # ib_insync connection
        self._ib = None
        self._ib_available = False
        
        # REST client (fallback)
        self._rest_client = None
        
        # Try to import ib_insync
        try:
            from ib_insync import IB, Stock, MarketOrder, LimitOrder
            self._IB = IB
            self._Stock = Stock
            self._MarketOrder = MarketOrder
            self._LimitOrder = LimitOrder
            self._ib_available = True
        except ImportError:
            logger.warning("ib_insync not available, will use REST fallback")
            self._ib_available = False
    
    async def connect(self) -> bool:
        """Connect to IBKR"""
        if self._ib_available:
            return await self._connect_insync()
        else:
            return await self._connect_rest()
    
    async def _connect_insync(self) -> bool:
        """Connect using ib_insync"""
        try:
            # ib_insync is synchronous, run in executor
            def _connect():
                ib = self._IB()
                ib.connect(
                    self._host,
                    self._port,
                    clientId=self._client_id,
                    timeout=30
                )
                return ib
            
            loop = asyncio.get_running_loop()
            self._ib = await loop.run_in_executor(None, _connect)
            
            if self._ib.isConnected():
                self._connected = True
                logger.info(f"IBKR connected via ib_insync: {self._host}:{self._port}")
                
                # Get initial account info
                await self._sync_account()
                
                return True
            else:
                logger.error("IBKR connection failed")
                return False
                
        except Exception as e:
            logger.error(f"IBKR connection error: {e}")
            return False
    
    async def _connect_rest(self) -> bool:
        """Connect using REST API (Client Portal)"""
        # REST connection is stateless, just mark as connected
        self._connected = True
        logger.info("IBKR REST mode enabled (ensure Client Portal Gateway is running)")
        return True
    
    async def disconnect(self) -> None:
        """Disconnect from IBKR"""
        if self._ib and self._ib.isConnected():
            def _disconnect():
                self._ib.disconnect()
            
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _disconnect)
        
        self._connected = False
        logger.info("IBKR disconnected")
    
    async def _sync_account(self) -> None:
        """Sync account data from IBKR"""
        if not self._ib:
            return
        
        def _get_account():
            account_values = self._ib.accountValues()
            positions = self._ib.positions()
            return account_values, positions
        
        loop = asyncio.get_running_loop()
        account_values, positions = await loop.run_in_executor(None, _get_account)
        
        # Extract cash
        for av in account_values:
            if av.tag == "TotalCashValue" and av.currency == "USD":
                self._cash = float(av.value)
                break
        
        # Extract positions
        self._positions.clear()
        for pos in positions:
            if pos.position != 0:
                self._positions[pos.contract.symbol] = Position(
                    symbol=pos.contract.symbol,
                    quantity=float(pos.position),
                    avg_cost=float(pos.avgCost),
                    current_price=0.0,  # Will update with quotes
                    market_value=float(pos.position * pos.avgCost),
                )
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        if self._ib:
            await self._sync_account()
        
        return {
            "account_id": f"IBKR-{self._client_id}",
            "type": "margin",
            "mode": self._mode,
            "cash": self._cash,
            "portfolio_value": await self.get_portfolio_value(),
            "positions_count": len(self._positions),
        }
    
    @retry_async(max_retries=3, delay=1.0)
    async def submit_order(self, order: Order) -> Order:
        """Submit order to IBKR"""
        if not self._connected:
            raise ConnectionError("Not connected to IBKR")
        
        if self._ib_available and self._ib:
            return await self._submit_insync(order)
        else:
            return await self._submit_rest(order)
    
    async def _submit_insync(self, order: Order) -> Order:
        """Submit order via ib_insync"""
        def _place_order():
            # Create contract
            contract = self._Stock(
                order.symbol,
                order.exchange or "SMART",
                order.currency or "USD"
            )
            
            # Create order object
            if order.order_type == OrderType.MARKET:
                ib_order = self._MarketOrder(
                    self.SIDE_MAP[order.side],
                    order.quantity
                )
            else:
                ib_order = self._LimitOrder(
                    self.SIDE_MAP[order.side],
                    order.quantity,
                    order.price or 0.0
                )
            
            ib_order.tif = self.TIF_MAP.get(order.time_in_force, "DAY")
            
            # Place order
            trade = self._ib.placeOrder(contract, ib_order)
            self._ib.sleep(1)  # Wait for order acknowledgment
            
            return trade
        
        loop = asyncio.get_running_loop()
        trade = await loop.run_in_executor(None, _place_order)
        
        # Update order status
        if trade.orderStatus.status == "Filled":
            order.status = OrderStatus.FILLED
        elif trade.orderStatus.status == "Submitted":
            order.status = OrderStatus.SUBMITTED
        elif trade.orderStatus.status == "Cancelled":
            order.status = OrderStatus.CANCELLED
        else:
            order.status = OrderStatus.PENDING
        
        order.broker_order_id = str(trade.order.orderId)
        
        # Store order
        self._orders[order.order_id] = order
        
        logger.info(f"IBKR order submitted: {order.order_id} -> {order.broker_order_id}")
        return order
    
    async def _submit_rest(self, order: Order) -> Order:
        """Submit order via REST API (placeholder)"""
        # REST implementation would go here
        logger.warning("REST order submission not fully implemented")
        order.status = OrderStatus.SUBMITTED
        order.broker_order_id = f"REST-{datetime.now().timestamp()}"
        return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        if order_id not in self._orders:
            return False
        
        order = self._orders[order_id]
        
        if self._ib and order.broker_order_id:
            def _cancel():
                self._ib.cancelOrder(int(order.broker_order_id))
            
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _cancel)
        
        order.status = OrderStatus.CANCELLED
        return True
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status"""
        if order_id in self._orders:
            return self._orders[order_id].status
        return OrderStatus.CANCELLED
    
    async def get_quote(self, symbol: str) -> Dict[str, float]:
        """Get quote from IBKR"""
        if not self._ib:
            return {"bid": 0, "ask": 0, "last": 0, "volume": 0}
        
        def _get_quote():
            contract = self._Stock(symbol, "SMART", "USD")
            self._ib.reqMarketDataType(2)  # Frozen data
            ticker = self._ib.reqMktData(contract, "", False, False)
            self._ib.sleep(2)
            
            result = {
                "bid": float(ticker.bid) if ticker.bid else 0,
                "ask": float(ticker.ask) if ticker.ask else 0,
                "last": float(ticker.last) if ticker.last else 0,
                "volume": int(ticker.volume) if ticker.volume else 0,
            }
            
            self._ib.cancelMktData(contract)
            return result
        
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _get_quote)
    
    async def get_historical_data(
        self,
        symbol: str,
        duration: str = "1 Y",
        bar_size: str = "1 day"
    ) -> List[Dict[str, Any]]:
        """Get historical data from IBKR"""
        if not self._ib:
            return []
        
        def _get_history():
            contract = self._Stock(symbol, "SMART", "USD")
            bars = self._ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="ADJUSTED_LAST",
                useRTH=True
            )
            
            return [
                {
                    "date": bar.date,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
                for bar in bars
            ]
        
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _get_history)
