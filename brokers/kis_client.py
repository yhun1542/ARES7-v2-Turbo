"""
KIS Client
===========
한국투자증권 브로커 클라이언트

한국투자증권 Open API 연동.

환경변수:
- KIS_APP_KEY: 앱 키
- KIS_APP_SECRET: 앱 시크릿
- KIS_ACCOUNT_NO: 계좌번호
- KIS_ACCOUNT_CODE: 계좌구분 (01: 주식, 02: 선물옵션)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp

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


class KISClient(BaseBroker):
    """
    한국투자증권 Client
    
    REST API 기반 주문/조회.
    """
    
    # API Endpoints
    ENDPOINTS = {
        "paper": {
            "base": "https://openapivts.koreainvestment.com:29443",
            "ws": "ws://ops.koreainvestment.com:21000",
        },
        "live": {
            "base": "https://openapi.koreainvestment.com:9443",
            "ws": "ws://ops.koreainvestment.com:31000",
        }
    }
    
    # Order type mapping
    ORDER_TYPE_MAP = {
        OrderType.LIMIT: "00",      # 지정가
        OrderType.MARKET: "01",     # 시장가
    }
    
    def __init__(
        self,
        app_key: Optional[str] = None,
        app_secret: Optional[str] = None,
        account_no: Optional[str] = None,
        account_code: str = "01",
        mode: str = "paper"
    ):
        """
        Initialize KIS client
        
        Args:
            app_key: API 앱 키
            app_secret: API 앱 시크릿
            account_no: 계좌번호
            account_code: 계좌구분코드
            mode: "paper" or "live"
        """
        super().__init__(name="KIS", mode=mode)
        
        self._app_key = app_key or get_env("KIS_APP_KEY", required=True)
        self._app_secret = app_secret or get_env("KIS_APP_SECRET", required=True)
        self._account_no = account_no or get_env("KIS_ACCOUNT_NO", required=True)
        self._account_code = account_code or get_env("KIS_ACCOUNT_CODE", default="01")
        
        self._base_url = self.ENDPOINTS[mode]["base"]
        
        # Token management
        self._access_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None
        
        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def connect(self) -> bool:
        """Connect to KIS API"""
        try:
            self._session = aiohttp.ClientSession()
            
            # Get access token
            await self._refresh_token()
            
            if self._access_token:
                self._connected = True
                logger.info(f"KIS connected ({self._mode} mode)")
                
                # Sync account
                await self._sync_account()
                
                return True
            else:
                logger.error("KIS token acquisition failed")
                return False
                
        except Exception as e:
            logger.error(f"KIS connection error: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from KIS"""
        if self._session:
            await self._session.close()
            self._session = None
        
        self._connected = False
        self._access_token = None
        logger.info("KIS disconnected")
    
    async def _refresh_token(self) -> None:
        """Get or refresh access token"""
        # Check if current token is still valid
        if self._access_token and self._token_expires:
            if datetime.now() < self._token_expires - timedelta(minutes=30):
                return
        
        url = f"{self._base_url}/oauth2/tokenP"
        
        payload = {
            "grant_type": "client_credentials",
            "appkey": self._app_key,
            "appsecret": self._app_secret,
        }
        
        async with self._session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                self._access_token = data.get("access_token")
                expires_in = int(data.get("expires_in", 86400))
                self._token_expires = datetime.now() + timedelta(seconds=expires_in)
                logger.info("KIS token refreshed")
            else:
                text = await response.text()
                logger.error(f"KIS token error: {text}")
                raise Exception(f"Token error: {response.status}")
    
    def _get_headers(self, tr_id: str) -> Dict[str, str]:
        """Get API headers"""
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self._access_token}",
            "appkey": self._app_key,
            "appsecret": self._app_secret,
            "tr_id": tr_id,
        }
    
    @retry_async(max_retries=3, delay=1.0)
    async def _request(
        self,
        method: str,
        path: str,
        tr_id: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make API request"""
        await self._refresh_token()
        
        url = f"{self._base_url}{path}"
        headers = self._get_headers(tr_id)
        
        if method == "GET":
            async with self._session.get(url, headers=headers, params=params) as response:
                return await response.json()
        else:
            async with self._session.post(url, headers=headers, json=data) as response:
                return await response.json()
    
    async def _sync_account(self) -> None:
        """Sync account data"""
        # Get balance
        balance = await self._get_balance()
        if balance:
            self._cash = balance.get("cash", 0)
        
        # Get positions
        positions = await self._get_positions_raw()
        self._positions.clear()
        
        for pos in positions:
            symbol = pos.get("pdno")  # 종목코드
            qty = float(pos.get("hldg_qty", 0))
            
            if qty > 0:
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=qty,
                    avg_cost=float(pos.get("pchs_avg_pric", 0)),
                    current_price=float(pos.get("prpr", 0)),
                    market_value=float(pos.get("evlu_amt", 0)),
                    unrealized_pnl=float(pos.get("evlu_pfls_amt", 0)),
                )
    
    async def _get_balance(self) -> Dict[str, Any]:
        """Get account balance"""
        # 모의투자 vs 실전투자 TR_ID 다름
        tr_id = "VTTC8434R" if self._mode == "paper" else "TTTC8434R"
        
        params = {
            "CANO": self._account_no[:8],
            "ACNT_PRDT_CD": self._account_no[8:],
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }
        
        result = await self._request("GET", "/uapi/domestic-stock/v1/trading/inquire-balance", tr_id, params=params)
        
        if result.get("rt_cd") == "0":
            output2 = result.get("output2", [{}])
            if output2:
                return {
                    "cash": float(output2[0].get("dnca_tot_amt", 0)),
                    "total_value": float(output2[0].get("tot_evlu_amt", 0)),
                }
        
        return {}
    
    async def _get_positions_raw(self) -> List[Dict]:
        """Get raw positions data"""
        tr_id = "VTTC8434R" if self._mode == "paper" else "TTTC8434R"
        
        params = {
            "CANO": self._account_no[:8],
            "ACNT_PRDT_CD": self._account_no[8:],
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }
        
        result = await self._request("GET", "/uapi/domestic-stock/v1/trading/inquire-balance", tr_id, params=params)
        
        if result.get("rt_cd") == "0":
            return result.get("output1", [])
        
        return []
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        await self._sync_account()
        
        return {
            "account_id": self._account_no,
            "type": "stock" if self._account_code == "01" else "futures",
            "mode": self._mode,
            "cash": self._cash,
            "portfolio_value": await self.get_portfolio_value(),
            "positions_count": len(self._positions),
        }
    
    async def submit_order(self, order: Order) -> Order:
        """Submit order to KIS"""
        if not self._connected:
            raise ConnectionError("Not connected to KIS")
        
        # TR_ID for order
        if self._mode == "paper":
            tr_id = "VTTC0802U" if order.side == Side.BUY else "VTTC0801U"
        else:
            tr_id = "TTTC0802U" if order.side == Side.BUY else "TTTC0801U"
        
        # Order type
        ord_dvsn = self.ORDER_TYPE_MAP.get(order.order_type, "00")
        
        payload = {
            "CANO": self._account_no[:8],
            "ACNT_PRDT_CD": self._account_no[8:],
            "PDNO": order.symbol,
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(int(order.quantity)),
            "ORD_UNPR": str(int(order.price)) if order.price else "0",
        }
        
        result = await self._request("POST", "/uapi/domestic-stock/v1/trading/order-cash", tr_id, data=payload)
        
        if result.get("rt_cd") == "0":
            order.status = OrderStatus.SUBMITTED
            order.broker_order_id = result.get("output", {}).get("ODNO", "")
            logger.info(f"KIS order submitted: {order.symbol} {order.side.value} {order.quantity}")
        else:
            order.status = OrderStatus.REJECTED
            logger.error(f"KIS order rejected: {result.get('msg1')}")
        
        self._orders[order.order_id] = order
        return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        if order_id not in self._orders:
            return False
        
        order = self._orders[order_id]
        
        # TR_ID for cancel
        tr_id = "VTTC0803U" if self._mode == "paper" else "TTTC0803U"
        
        payload = {
            "CANO": self._account_no[:8],
            "ACNT_PRDT_CD": self._account_no[8:],
            "KRX_FWDG_ORD_ORGNO": "",
            "ORGN_ODNO": order.broker_order_id,
            "ORD_DVSN": "00",
            "RVSE_CNCL_DVSN_CD": "02",  # 취소
            "ORD_QTY": "0",
            "ORD_UNPR": "0",
            "QTY_ALL_ORD_YN": "Y",
        }
        
        result = await self._request("POST", "/uapi/domestic-stock/v1/trading/order-rvsecncl", tr_id, data=payload)
        
        if result.get("rt_cd") == "0":
            order.status = OrderStatus.CANCELLED
            return True
        
        return False
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status"""
        if order_id in self._orders:
            return self._orders[order_id].status
        return OrderStatus.CANCELLED
    
    async def get_quote(self, symbol: str) -> Dict[str, float]:
        """Get current quote"""
        tr_id = "FHKST01010100"
        
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": symbol,
        }
        
        result = await self._request("GET", "/uapi/domestic-stock/v1/quotations/inquire-price", tr_id, params=params)
        
        if result.get("rt_cd") == "0":
            output = result.get("output", {})
            return {
                "bid": float(output.get("stck_sdpr", 0)),   # 매수호가
                "ask": float(output.get("stck_hgpr", 0)),   # 매도호가
                "last": float(output.get("stck_prpr", 0)),  # 현재가
                "volume": int(output.get("acml_vol", 0)),   # 누적거래량
            }
        
        return {"bid": 0, "ask": 0, "last": 0, "volume": 0}
    
    async def get_daily_prices(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> List[Dict[str, Any]]:
        """Get daily price history"""
        tr_id = "FHKST03010100"
        
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": symbol,
            "FID_INPUT_DATE_1": start_date,
            "FID_INPUT_DATE_2": end_date,
            "FID_PERIOD_DIV_CODE": "D",
            "FID_ORG_ADJ_PRC": "1",
        }
        
        result = await self._request("GET", "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice", tr_id, params=params)
        
        if result.get("rt_cd") == "0":
            return [
                {
                    "date": item.get("stck_bsop_date"),
                    "open": float(item.get("stck_oprc", 0)),
                    "high": float(item.get("stck_hgpr", 0)),
                    "low": float(item.get("stck_lwpr", 0)),
                    "close": float(item.get("stck_clpr", 0)),
                    "volume": int(item.get("acml_vol", 0)),
                }
                for item in result.get("output2", [])
            ]
        
        return []
