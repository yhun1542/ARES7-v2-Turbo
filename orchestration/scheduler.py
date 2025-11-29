"""
Trading Scheduler
==================
스케줄링, 재튜닝, 리포팅 관리
"""

from __future__ import annotations

import asyncio
from datetime import datetime, time, timedelta
from typing import Any, Callable, Dict, List, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from core.utils import get_logger

logger = get_logger(__name__)


class TradingScheduler:
    """
    Trading Scheduler
    
    주요 기능:
    - 일일/주간/월간 리밸런싱 스케줄
    - 시장 시간 체크
    - 재튜닝 스케줄
    - 리포트 생성 스케줄
    """
    
    # 시장 시간 (미국 동부시간)
    US_MARKET_OPEN = time(9, 30)
    US_MARKET_CLOSE = time(16, 0)
    US_MARKET_PRE_OPEN = time(4, 0)
    US_MARKET_AFTER_CLOSE = time(20, 0)
    
    # 한국 시장 시간
    KR_MARKET_OPEN = time(9, 0)
    KR_MARKET_CLOSE = time(15, 30)
    
    def __init__(self):
        """Initialize scheduler"""
        self._scheduler = AsyncIOScheduler()
        self._jobs: Dict[str, str] = {}  # job_name -> job_id
    
    def start(self) -> None:
        """Start scheduler"""
        self._scheduler.start()
        logger.info("Trading scheduler started")
    
    def stop(self) -> None:
        """Stop scheduler"""
        self._scheduler.shutdown()
        logger.info("Trading scheduler stopped")
    
    def schedule_daily(
        self,
        name: str,
        func: Callable,
        hour: int,
        minute: int = 0,
        timezone: str = "America/New_York"
    ) -> str:
        """
        Schedule daily task
        
        Args:
            name: Job name
            func: Function to execute
            hour: Hour (0-23)
            minute: Minute (0-59)
            timezone: Timezone
        
        Returns:
            Job ID
        """
        trigger = CronTrigger(hour=hour, minute=minute, timezone=timezone)
        
        job = self._scheduler.add_job(
            func,
            trigger=trigger,
            id=name,
            name=name,
            replace_existing=True
        )
        
        self._jobs[name] = job.id
        logger.info(f"Scheduled daily job '{name}' at {hour:02d}:{minute:02d} {timezone}")
        
        return job.id
    
    def schedule_weekly(
        self,
        name: str,
        func: Callable,
        day_of_week: str,
        hour: int,
        minute: int = 0,
        timezone: str = "America/New_York"
    ) -> str:
        """
        Schedule weekly task
        
        Args:
            name: Job name
            func: Function to execute
            day_of_week: Day of week (mon, tue, wed, thu, fri, sat, sun)
            hour: Hour
            minute: Minute
            timezone: Timezone
        
        Returns:
            Job ID
        """
        trigger = CronTrigger(
            day_of_week=day_of_week,
            hour=hour,
            minute=minute,
            timezone=timezone
        )
        
        job = self._scheduler.add_job(
            func,
            trigger=trigger,
            id=name,
            name=name,
            replace_existing=True
        )
        
        self._jobs[name] = job.id
        logger.info(f"Scheduled weekly job '{name}' on {day_of_week} at {hour:02d}:{minute:02d}")
        
        return job.id
    
    def schedule_monthly(
        self,
        name: str,
        func: Callable,
        day: int,
        hour: int,
        minute: int = 0,
        timezone: str = "America/New_York"
    ) -> str:
        """
        Schedule monthly task
        
        Args:
            name: Job name
            func: Function to execute
            day: Day of month (1-31)
            hour: Hour
            minute: Minute
            timezone: Timezone
        
        Returns:
            Job ID
        """
        trigger = CronTrigger(
            day=day,
            hour=hour,
            minute=minute,
            timezone=timezone
        )
        
        job = self._scheduler.add_job(
            func,
            trigger=trigger,
            id=name,
            name=name,
            replace_existing=True
        )
        
        self._jobs[name] = job.id
        logger.info(f"Scheduled monthly job '{name}' on day {day} at {hour:02d}:{minute:02d}")
        
        return job.id
    
    def remove_job(self, name: str) -> bool:
        """Remove scheduled job"""
        if name in self._jobs:
            self._scheduler.remove_job(self._jobs[name])
            del self._jobs[name]
            logger.info(f"Removed job '{name}'")
            return True
        return False
    
    def get_jobs(self) -> List[Dict[str, Any]]:
        """Get all scheduled jobs"""
        return [
            {
                "name": job.name,
                "id": job.id,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            }
            for job in self._scheduler.get_jobs()
        ]
    
    @staticmethod
    def is_us_market_open(now: Optional[datetime] = None) -> bool:
        """Check if US market is open"""
        import pytz
        
        if now is None:
            now = datetime.now(pytz.timezone("America/New_York"))
        elif now.tzinfo is None:
            now = pytz.timezone("America/New_York").localize(now)
        
        # Weekday check
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Time check
        market_time = now.time()
        return TradingScheduler.US_MARKET_OPEN <= market_time <= TradingScheduler.US_MARKET_CLOSE
    
    @staticmethod
    def is_kr_market_open(now: Optional[datetime] = None) -> bool:
        """Check if Korean market is open"""
        import pytz
        
        if now is None:
            now = datetime.now(pytz.timezone("Asia/Seoul"))
        elif now.tzinfo is None:
            now = pytz.timezone("Asia/Seoul").localize(now)
        
        # Weekday check
        if now.weekday() >= 5:
            return False
        
        # Time check
        market_time = now.time()
        return TradingScheduler.KR_MARKET_OPEN <= market_time <= TradingScheduler.KR_MARKET_CLOSE
    
    @staticmethod
    def next_market_open(market: str = "US") -> datetime:
        """Get next market open time"""
        import pytz
        
        if market == "US":
            tz = pytz.timezone("America/New_York")
            market_open = TradingScheduler.US_MARKET_OPEN
        else:
            tz = pytz.timezone("Asia/Seoul")
            market_open = TradingScheduler.KR_MARKET_OPEN
        
        now = datetime.now(tz)
        
        # If market is open or hasn't opened today, return today's open
        if now.time() < market_open and now.weekday() < 5:
            return now.replace(
                hour=market_open.hour,
                minute=market_open.minute,
                second=0,
                microsecond=0
            )
        
        # Otherwise, find next trading day
        next_day = now + timedelta(days=1)
        while next_day.weekday() >= 5:  # Skip weekends
            next_day += timedelta(days=1)
        
        return next_day.replace(
            hour=market_open.hour,
            minute=market_open.minute,
            second=0,
            microsecond=0
        )
    
    @staticmethod
    def time_to_market_close(market: str = "US") -> Optional[timedelta]:
        """Get time until market close (None if closed)"""
        import pytz
        
        if market == "US":
            tz = pytz.timezone("America/New_York")
            market_close = TradingScheduler.US_MARKET_CLOSE
        else:
            tz = pytz.timezone("Asia/Seoul")
            market_close = TradingScheduler.KR_MARKET_CLOSE
        
        now = datetime.now(tz)
        
        # Check if market is open
        if market == "US":
            is_open = TradingScheduler.is_us_market_open(now)
        else:
            is_open = TradingScheduler.is_kr_market_open(now)
        
        if not is_open:
            return None
        
        close_time = now.replace(
            hour=market_close.hour,
            minute=market_close.minute,
            second=0,
            microsecond=0
        )
        
        return close_time - now


def setup_default_schedule(
    scheduler: TradingScheduler,
    rebalance_func: Callable,
    report_func: Callable,
    retune_func: Optional[Callable] = None
) -> None:
    """
    Setup default trading schedule
    
    Args:
        scheduler: Trading scheduler
        rebalance_func: Daily rebalance function
        report_func: Daily report function
        retune_func: Monthly retune function
    """
    # Daily rebalance at 15:30 ET (30 min before close)
    scheduler.schedule_daily(
        name="daily_rebalance",
        func=rebalance_func,
        hour=15,
        minute=30
    )
    
    # Daily report at 17:00 ET (after close)
    scheduler.schedule_daily(
        name="daily_report",
        func=report_func,
        hour=17,
        minute=0
    )
    
    # Monthly retune on first day
    if retune_func:
        scheduler.schedule_monthly(
            name="monthly_retune",
            func=retune_func,
            day=1,
            hour=6,
            minute=0
        )
    
    logger.info("Default trading schedule configured")
