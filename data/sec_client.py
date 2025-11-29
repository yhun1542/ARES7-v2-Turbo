"""
SEC API Client
===============
SEC 공시 데이터 - 이벤트 리스크 감지

Features Generated:
- filing_count_30d: 최근 30일 공시 수 → 이벤트 리스크
- has_8k_recent: 8-K (material event) 존재 여부 → 리스크 스케일링
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

import aiohttp

from core.interfaces import IDataProvider
from core.utils import get_env, get_logger, retry_async

logger = get_logger(__name__)


class SECClient(IDataProvider):
    """
    SEC API Client (sec-api.io)
    
    Monitors SEC filings for:
    - Material events (8-K)
    - Quarterly/Annual reports (10-Q, 10-K)
    - Institutional holdings (13F)
    """
    
    BASE_URL = "https://api.sec-api.io"
    
    # Filing types and their risk implications
    FILING_RISK_MAP = {
        "8-K": 0.3,      # Material event - moderate risk
        "8-K/A": 0.4,    # Amended material event - higher risk
        "10-K": 0.1,     # Annual report - low risk
        "10-Q": 0.1,     # Quarterly report - low risk
        "4": 0.05,       # Insider trading - low risk
        "SC 13G": 0.05,  # Institutional ownership - low risk
        "S-1": 0.2,      # IPO registration - moderate risk
        "DEF 14A": 0.1,  # Proxy statement - low risk
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize SEC client
        
        Args:
            api_key: SEC-API key (defaults to SEC_API_KEY env var)
        """
        self._api_key = api_key or get_env("SEC_API_KEY", required=True)
        self._session: Optional[aiohttp.ClientSession] = None
        self._connected = False
    
    @property
    def name(self) -> str:
        return "sec"
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    async def connect(self) -> bool:
        """Initialize HTTP session"""
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": self._api_key}
            )
        self._connected = True
        logger.info("SEC client connected")
        return True
    
    async def disconnect(self) -> None:
        """Close HTTP session"""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        logger.info("SEC client disconnected")
    
    async def get_prices(self, *args, **kwargs) -> Any:
        """Not implemented"""
        raise NotImplementedError("SECClient is for filing data")
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Not implemented"""
        raise NotImplementedError("SECClient is for filing data")
    
    @retry_async(max_retries=2, delay=1.0, exceptions=(aiohttp.ClientError,))
    async def _query(
        self,
        query: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute SEC API query"""
        if not self._session:
            await self.connect()
        
        url = self.BASE_URL
        
        async with self._session.post(url, json=query) as response:
            if response.status == 200:
                return await response.json()
            else:
                text = await response.text()
                logger.error(f"SEC API error {response.status}: {text}")
                raise aiohttp.ClientError(f"API error: {response.status}")
    
    async def get_filings(
        self,
        symbol: str,
        filing_types: Optional[List[str]] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get SEC filings for a symbol
        
        Args:
            symbol: Stock ticker
            filing_types: List of form types (e.g., ["8-K", "10-K"])
            from_date: Start date
            to_date: End date
            limit: Maximum results
        
        Returns:
            List of filing records
        """
        # Build query
        must_conditions = [
            {"match": {"ticker": symbol}}
        ]
        
        if filing_types:
            must_conditions.append({
                "terms": {"formType": filing_types}
            })
        
        if from_date:
            must_conditions.append({
                "range": {"filedAt": {"gte": from_date.strftime("%Y-%m-%d")}}
            })
        
        if to_date:
            must_conditions.append({
                "range": {"filedAt": {"lte": to_date.strftime("%Y-%m-%d")}}
            })
        
        query = {
            "query": {
                "bool": {
                    "must": must_conditions
                }
            },
            "from": "0",
            "size": str(limit),
            "sort": [{"filedAt": {"order": "desc"}}]
        }
        
        result = await self._query(query)
        
        filings = []
        for hit in result.get("filings", []):
            filings.append({
                "ticker": hit.get("ticker"),
                "form_type": hit.get("formType"),
                "filed_at": hit.get("filedAt"),
                "description": hit.get("description", ""),
                "url": hit.get("linkToFilingDetails"),
                "accession_number": hit.get("accessionNo"),
            })
        
        return filings
    
    async def get_filing_count(
        self,
        symbol: str,
        days: int = 30,
        filing_types: Optional[List[str]] = None
    ) -> int:
        """
        Count filings in recent period
        
        High filing count may indicate corporate events.
        """
        from_date = datetime.now() - timedelta(days=days)
        
        filings = await self.get_filings(
            symbol=symbol,
            filing_types=filing_types,
            from_date=from_date,
            limit=100
        )
        
        return len(filings)
    
    async def has_material_event(
        self,
        symbol: str,
        days: int = 7
    ) -> bool:
        """
        Check for recent 8-K (material event) filings
        
        Returns True if 8-K filed in recent days.
        This flag affects risk scaling.
        """
        from_date = datetime.now() - timedelta(days=days)
        
        filings = await self.get_filings(
            symbol=symbol,
            filing_types=["8-K", "8-K/A"],
            from_date=from_date,
            limit=5
        )
        
        has_8k = len(filings) > 0
        
        if has_8k:
            logger.info(f"Material event (8-K) detected for {symbol}")
        
        return has_8k
    
    async def calculate_filing_risk_score(
        self,
        symbol: str,
        days: int = 30
    ) -> float:
        """
        Calculate risk score based on filing activity
        
        Returns:
            Risk score (0-1), higher = more risk
        """
        from_date = datetime.now() - timedelta(days=days)
        
        filings = await self.get_filings(
            symbol=symbol,
            from_date=from_date,
            limit=50
        )
        
        if not filings:
            return 0.0
        
        # Calculate weighted risk score
        total_risk = 0.0
        
        for filing in filings:
            form_type = filing.get("form_type", "")
            base_risk = self.FILING_RISK_MAP.get(form_type, 0.05)
            
            # Decay by age
            filed_at = datetime.fromisoformat(filing["filed_at"].replace("Z", "+00:00"))
            age_days = (datetime.now(filed_at.tzinfo) - filed_at).days
            decay = max(0.1, 1.0 - (age_days / days))
            
            total_risk += base_risk * decay
        
        # Normalize
        normalized_risk = min(1.0, total_risk / 3.0)
        
        return normalized_risk
    
    async def get_insider_activity(
        self,
        symbol: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze insider trading activity (Form 4)
        
        Returns:
            Dict with buy/sell counts and net direction
        """
        from_date = datetime.now() - timedelta(days=days)
        
        filings = await self.get_filings(
            symbol=symbol,
            filing_types=["4"],
            from_date=from_date,
            limit=50
        )
        
        # Simple count (actual parsing would need more detailed API)
        return {
            "form_4_count": len(filings),
            "has_recent_insider_activity": len(filings) > 0,
        }
