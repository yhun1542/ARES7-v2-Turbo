#!/bin/bash
# =============================================================================
# ARES-Ultimate KIS Live Trading Script
# =============================================================================
#
# 사용법:
#   ./run_live_kis.sh [--shadow|--live]
#
# 옵션:
#   --shadow  실제 주문 없이 시뮬레이션만 (기본값)
#   --live    실제 주문 실행 (주의!)
#
# =============================================================================

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 스크립트 디렉토리
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 기본 모드
MODE="shadow"
BROKER="kis"

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --shadow)
            MODE="shadow"
            shift
            ;;
        --live)
            MODE="live"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# 환경 확인
echo -e "${GREEN}=== ARES-Ultimate KIS Launcher ===${NC}"
echo ""
echo "Mode: ${YELLOW}${MODE}${NC}"
echo "Broker: ${YELLOW}KIS (한국투자증권)${NC}"
echo "Project: ${PROJECT_DIR}"
echo ""

# 가상환경 활성화
if [ -f "${PROJECT_DIR}/venv/bin/activate" ]; then
    source "${PROJECT_DIR}/venv/bin/activate"
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo -e "${RED}✗ Virtual environment not found${NC}"
    echo "Run: python3.11 -m venv venv && pip install -e ."
    exit 1
fi

# 환경변수 로드
if [ -f "${PROJECT_DIR}/.env" ]; then
    export $(cat "${PROJECT_DIR}/.env" | grep -v '^#' | xargs)
    echo -e "${GREEN}✓ Environment variables loaded${NC}"
else
    echo -e "${YELLOW}⚠ .env file not found${NC}"
fi

# KIS API 키 확인
echo ""
echo "Checking KIS credentials..."

if [ -z "$KIS_APP_KEY" ] || [ -z "$KIS_APP_SECRET" ] || [ -z "$KIS_ACCOUNT_NO" ]; then
    echo -e "${RED}✗ KIS credentials not configured${NC}"
    echo "Please set KIS_APP_KEY, KIS_APP_SECRET, KIS_ACCOUNT_NO in .env"
    exit 1
else
    echo -e "${GREEN}✓ KIS credentials found${NC}"
fi

# Live 모드 확인
if [ "$MODE" = "live" ]; then
    echo ""
    echo -e "${RED}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                    ⚠️  경고 ⚠️                              ║${NC}"
    echo -e "${RED}║                                                           ║${NC}"
    echo -e "${RED}║  실전 모드: 실제 주문이 실행됩니다!                       ║${NC}"
    echo -e "${RED}║  한국투자증권 계좌에서 실제 자금이 사용됩니다.            ║${NC}"
    echo -e "${RED}║                                                           ║${NC}"
    echo -e "${RED}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    read -p "실전 거래를 시작하려면 '확인'을 입력하세요: " confirmation
    
    if [ "$confirmation" != "확인" ]; then
        echo -e "${YELLOW}실전 거래가 취소되었습니다.${NC}"
        exit 0
    fi
fi

# 로그 디렉토리 생성
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"

# 실행
echo ""
echo -e "${GREEN}ARES-Ultimate 시작 중...${NC}"
echo "로그: ${LOG_DIR}/ares_kis_$(date +%Y%m%d).log"
echo ""

cd "$PROJECT_DIR"

# KIS용 오케스트레이터 실행
export ARES_BROKER="kis"

if [ "$MODE" = "live" ]; then
    python -c "
import asyncio
from orchestration.live_orchestrator import LiveOrchestrator, OrchestratorConfig, OrchestratorMode
from engines.ares7_qm_regime.strategy import ARES7QMRegimeStrategy
from risk.aarm_core import TurboAARM
from brokers.kis_client import KISClient
from data.polygon_client import PolygonClient

async def main():
    strategy = ARES7QMRegimeStrategy('config/ares7_qm_turbo_final_251129.yaml')
    broker = KISClient(mode='live')
    data_providers = {'polygon': PolygonClient()}
    risk_manager = TurboAARM()
    
    config = OrchestratorConfig(
        mode=OrchestratorMode.LIVE,
        update_interval_seconds=60,
        rebalance_frequency='daily',
        rebalance_time='14:30',  # 한국시장 기준
    )
    
    orchestrator = LiveOrchestrator(
        strategy=strategy,
        broker=broker,
        data_providers=data_providers,
        risk_manager=risk_manager,
        config=config,
    )
    
    await orchestrator.start()

asyncio.run(main())
" 2>&1 | tee -a "${LOG_DIR}/ares_kis_$(date +%Y%m%d).log"
else
    python -c "
import asyncio
from orchestration.live_orchestrator import LiveOrchestrator, OrchestratorConfig, OrchestratorMode
from engines.ares7_qm_regime.strategy import ARES7QMRegimeStrategy
from risk.aarm_core import TurboAARM
from brokers.kis_client import KISClient
from data.polygon_client import PolygonClient

async def main():
    strategy = ARES7QMRegimeStrategy('config/ares7_qm_turbo_final_251129.yaml')
    broker = KISClient(mode='paper')
    data_providers = {'polygon': PolygonClient()}
    risk_manager = TurboAARM()
    
    config = OrchestratorConfig(
        mode=OrchestratorMode.SHADOW,
        update_interval_seconds=60,
        rebalance_frequency='daily',
        rebalance_time='14:30',
    )
    
    orchestrator = LiveOrchestrator(
        strategy=strategy,
        broker=broker,
        data_providers=data_providers,
        risk_manager=risk_manager,
        config=config,
    )
    
    await orchestrator.start()

asyncio.run(main())
" 2>&1 | tee -a "${LOG_DIR}/ares_kis_$(date +%Y%m%d).log"
fi
