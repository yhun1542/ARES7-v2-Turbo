#!/bin/bash
# =============================================================================
# ARES-Ultimate IBKR Live Trading Script
# =============================================================================
#
# 사용법:
#   ./run_live_ibkr.sh [--shadow|--live]
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
NC='\033[0m' # No Color

# 스크립트 디렉토리
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 기본 모드
MODE="shadow"

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
echo -e "${GREEN}=== ARES-Ultimate IBKR Launcher ===${NC}"
echo ""
echo "Mode: ${YELLOW}${MODE}${NC}"
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

# IBKR 연결 확인
echo ""
echo "Checking IBKR connection..."
IB_HOST=${IB_HOST:-127.0.0.1}
IB_PORT=${IB_PORT:-7497}

if nc -z "$IB_HOST" "$IB_PORT" 2>/dev/null; then
    echo -e "${GREEN}✓ IBKR Gateway reachable at ${IB_HOST}:${IB_PORT}${NC}"
else
    echo -e "${RED}✗ Cannot connect to IBKR Gateway at ${IB_HOST}:${IB_PORT}${NC}"
    echo "Please ensure TWS or IB Gateway is running with API enabled."
    exit 1
fi

# Live 모드 확인
if [ "$MODE" = "live" ]; then
    echo ""
    echo -e "${RED}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                    ⚠️  WARNING ⚠️                           ║${NC}"
    echo -e "${RED}║                                                           ║${NC}"
    echo -e "${RED}║  LIVE MODE: Real orders will be placed!                   ║${NC}"
    echo -e "${RED}║  This will use REAL MONEY from your IBKR account.         ║${NC}"
    echo -e "${RED}║                                                           ║${NC}"
    echo -e "${RED}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    read -p "Type 'CONFIRM' to proceed with LIVE trading: " confirmation
    
    if [ "$confirmation" != "CONFIRM" ]; then
        echo -e "${YELLOW}Live trading cancelled.${NC}"
        exit 0
    fi
fi

# 로그 디렉토리 생성
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"

# 실행
echo ""
echo -e "${GREEN}Starting ARES-Ultimate...${NC}"
echo "Logs: ${LOG_DIR}/ares_$(date +%Y%m%d).log"
echo ""

cd "$PROJECT_DIR"

if [ "$MODE" = "live" ]; then
    python -m orchestration.live_orchestrator --live 2>&1 | tee -a "${LOG_DIR}/ares_$(date +%Y%m%d).log"
else
    python -m orchestration.live_orchestrator 2>&1 | tee -a "${LOG_DIR}/ares_$(date +%Y%m%d).log"
fi
