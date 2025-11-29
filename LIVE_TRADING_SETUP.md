# ARES-Ultimate Live Trading Setup Guide
## 라이브 트레이딩 설정 및 Shadow Mode 가이드

**날짜**: 2025-11-29  
**버전**: 1.0  
**EC2 IP**: 3.35.141.47

---

## 📋 개요

ARES-Ultimate를 라이브 트레이딩 환경에서 실행하기 위한 설정 가이드입니다.

**지원 브로커:**
1. **Interactive Brokers (IBKR)** - 미국 주식
2. **한국투자증권 (KIS)** - 한국 주식

**운영 모드:**
1. **Shadow Mode**: 실제 주문 없이 시그널만 생성 (추천)
2. **Paper Trading**: 가상 계좌에서 실제 주문 테스트
3. **Live Trading**: 실제 계좌에서 실제 주문 (신중)

---

## 🔧 Phase 5: 라이브 트레이딩 설정

### Step 1: 브로커 설정

#### Option A: Interactive Brokers (IBKR)

**1. TWS (Trader Workstation) 설치**
```bash
# EC2에 TWS 설치 (GUI 필요)
# 또는 IB Gateway 사용 (headless)
wget https://download2.interactivebrokers.com/installers/ibgateway/latest-standalone/ibgateway-latest-standalone-linux-x64.sh
chmod +x ibgateway-latest-standalone-linux-x64.sh
./ibgateway-latest-standalone-linux-x64.sh
```

**2. IB Gateway 설정**
- Port: 7497 (Paper Trading) 또는 7496 (Live)
- API 활성화: TWS Settings > API > Enable ActiveX and Socket Clients
- Trusted IPs: 127.0.0.1 추가

**3. 환경변수 설정**
```bash
# ~/.bashrc 또는 .env에 추가
export IBKR_HOST="127.0.0.1"
export IBKR_PORT="7497"  # Paper: 7497, Live: 7496
export IBKR_CLIENT_ID="1"
export IBKR_ACCOUNT_ID="DU123456"  # 실제 계좌 ID로 변경
```

**4. 연결 테스트**
```bash
cd /home/ubuntu/workspace/ARES-Ultimate-251129
source venv/bin/activate
python -c "
from brokers.ibkr_client import IBKRClient
import asyncio

async def test():
    client = IBKRClient()
    connected = await client.connect()
    print(f'IBKR Connected: {connected}')
    if connected:
        positions = await client.get_positions()
        print(f'Positions: {positions}')
        await client.disconnect()

asyncio.run(test())
"
```

#### Option B: 한국투자증권 (KIS)

**1. API 신청**
- 한국투자증권 홈페이지 > Open API 신청
- App Key, App Secret 발급

**2. 환경변수 설정**
```bash
# ~/.bashrc 또는 .env에 추가
export KIS_APP_KEY="your_app_key"
export KIS_APP_SECRET="your_app_secret"
export KIS_ACCOUNT_NO="12345678-01"
export KIS_CANO="12345678"
export KIS_ACNT_PRDT_CD="01"
```

**3. 연결 테스트**
```bash
cd /home/ubuntu/workspace/ARES-Ultimate-251129
source venv/bin/activate
python -c "
from brokers.kis_client import KISClient
import asyncio

async def test():
    client = KISClient()
    connected = await client.connect()
    print(f'KIS Connected: {connected}')
    if connected:
        balance = await client.get_account_balance()
        print(f'Balance: {balance}')
        await client.disconnect()

asyncio.run(test())
"
```

---

### Step 2: Shadow Mode 설정

Shadow Mode는 **실제 주문 없이** 시그널만 생성하고 로깅하는 안전한 모드입니다.

**1. Shadow Mode 설정 파일 생성**
```yaml
# config/live_trading_shadow.yaml
mode: "shadow"  # shadow, paper, live

orchestrator:
  interval_seconds: 300  # 5분마다 실행
  timezone: "America/New_York"
  market_hours_only: true

strategy:
  name: "ARES7_QM_REGIME_TURBO"
  config_file: "config/ares7_qm_turbo_final_251129.yaml"

risk:
  max_portfolio_leverage: 1.5
  max_position_size: 0.10  # 10% per position
  max_daily_loss: 0.02  # 2% daily loss limit

broker:
  name: "ibkr"  # or "kis"
  paper_trading: true

logging:
  level: "INFO"
  file: "logs/live_trading_shadow.log"
  rotation: "daily"

alerts:
  email: "your_email@example.com"  # 선택사항
  slack_webhook: ""  # 선택사항
```

**2. Shadow Mode 실행 스크립트**
```bash
#!/bin/bash
# scripts/run_shadow_mode.sh

cd /home/ubuntu/workspace/ARES-Ultimate-251129
source venv/bin/activate
source setup_env_vars.sh

# Shadow Mode 실행
python scripts/main.py live --config config/live_trading_shadow.yaml --shadow

# 또는 백그라운드 실행
# nohup python scripts/main.py live --config config/live_trading_shadow.yaml --shadow > logs/shadow_mode.out 2>&1 &
```

**3. Shadow Mode 시작**
```bash
cd /home/ubuntu/workspace/ARES-Ultimate-251129
chmod +x scripts/run_shadow_mode.sh
./scripts/run_shadow_mode.sh
```

**4. 로그 모니터링**
```bash
# 실시간 로그 확인
tail -f logs/live_trading_shadow.log

# 시그널 확인
grep "Signal generated" logs/live_trading_shadow.log

# 에러 확인
grep "ERROR" logs/live_trading_shadow.log
```

---

### Step 3: Paper Trading 설정

Paper Trading은 가상 계좌에서 실제 주문을 테스트하는 모드입니다.

**1. Paper Trading 설정**
```yaml
# config/live_trading_paper.yaml
mode: "paper"

broker:
  name: "ibkr"
  paper_trading: true
  port: 7497  # Paper trading port

# 나머지 설정은 shadow와 동일
```

**2. Paper Trading 실행**
```bash
python scripts/main.py live --config config/live_trading_paper.yaml
```

**3. 성능 모니터링**
```bash
# 포지션 확인
python scripts/main.py status --broker ibkr

# 수익률 확인
python scripts/main.py performance --start-date 2025-11-29
```

---

### Step 4: Live Trading 설정 (신중!)

⚠️ **경고**: Live Trading은 실제 자금을 사용합니다. 충분한 테스트 후 진행하세요.

**사전 체크리스트:**
- [ ] Shadow Mode 최소 1주일 실행
- [ ] Paper Trading 최소 1개월 실행
- [ ] 모든 성능 지표 목표 달성
- [ ] 리스크 관리 파라미터 검증
- [ ] 긴급 중단 절차 숙지
- [ ] 계좌 잔고 확인 (최소 $10,000 권장)

**1. Live Trading 설정**
```yaml
# config/live_trading_live.yaml
mode: "live"

broker:
  name: "ibkr"
  paper_trading: false
  port: 7496  # Live trading port

risk:
  max_portfolio_leverage: 1.2  # 보수적으로 설정
  max_position_size: 0.05  # 5% per position
  max_daily_loss: 0.01  # 1% daily loss limit
  circuit_breaker:
    enabled: true
    trigger_loss: -0.03  # -3% 손실 시 자동 중단
    cooldown_hours: 24

alerts:
  email: "your_email@example.com"  # 필수!
  slack_webhook: "https://hooks.slack.com/..."  # 필수!
```

**2. Live Trading 시작**
```bash
# 수동 확인 후 시작
python scripts/main.py live --config config/live_trading_live.yaml

# 확인 프롬프트
# > WARNING: Live trading mode! Real money will be used.
# > Type 'YES' to confirm: YES
```

**3. 긴급 중단**
```bash
# 모든 포지션 청산 및 중단
python scripts/main.py emergency-stop --broker ibkr

# 또는 프로세스 강제 종료
pkill -f "main.py live"
```

---

## 📊 모니터링 및 알림

### 1. 시스템 모니터링

**실시간 대시보드**
```bash
# 웹 대시보드 시작 (포트 8000)
python scripts/main.py dashboard --port 8000

# 브라우저에서 접속
# http://3.35.141.47:8000
```

**주요 지표:**
- 현재 포지션 및 비중
- 일일 수익률
- 드로다운
- 시그널 생성 빈도
- API 연결 상태

### 2. 알림 설정

**이메일 알림**
```python
# config/alerts.yaml
email:
  enabled: true
  smtp_server: "smtp.gmail.com"
  smtp_port: 587
  from_email: "ares-bot@example.com"
  to_email: "your_email@example.com"
  password: "your_app_password"
  
  triggers:
    - signal_generated
    - position_opened
    - position_closed
    - daily_loss_limit
    - circuit_breaker_triggered
```

**Slack 알림**
```python
# Slack Webhook URL 설정
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

# 알림 테스트
python scripts/test_alerts.py
```

---

## 🔒 보안 및 백업

### 1. API 키 보안

```bash
# .env 파일 권한 설정
chmod 600 .env
chmod 600 setup_env_vars.sh

# Git에서 제외
echo ".env" >> .gitignore
echo "setup_env_vars.sh" >> .gitignore
```

### 2. 로그 백업

```bash
# 일일 로그 백업 (cron)
0 0 * * * tar -czf /home/ubuntu/backups/logs_$(date +\%Y\%m\%d).tar.gz /home/ubuntu/workspace/ARES-Ultimate-251129/logs/
```

### 3. 데이터 백업

```bash
# 주간 데이터 캐시 백업
0 0 * * 0 tar -czf /home/ubuntu/backups/data_cache_$(date +\%Y\%m\%d).tar.gz /home/ubuntu/workspace/ARES-Ultimate-251129/data_cache/
```

---

## 🚨 트러블슈팅

### 문제 1: IBKR 연결 실패

**증상**: `Connection refused` 에러

**해결책**:
1. IB Gateway가 실행 중인지 확인
2. API 설정에서 Socket Clients 활성화 확인
3. Trusted IPs에 127.0.0.1 추가
4. 포트 번호 확인 (Paper: 7497, Live: 7496)

### 문제 2: 데이터 로딩 느림

**증상**: Polygon API rate limit 초과

**해결책**:
1. 캐싱 활용: `use_cache=True`
2. 데이터 로딩 간격 조정
3. 필요한 심볼만 로드

### 문제 3: 메모리 부족

**증상**: `MemoryError` 발생

**해결책**:
1. 데이터 캐시 정리: `rm -rf data_cache/*`
2. 로그 파일 정리: `find logs/ -mtime +7 -delete`
3. EC2 인스턴스 업그레이드 고려

---

## 📈 성능 목표

| 지표 | 목표 | Shadow Mode | Paper Trading | Live Trading |
|------|------|-------------|---------------|--------------|
| **Sharpe Ratio** | 3.86+ | 모니터링 | 검증 필요 | 달성 필수 |
| **Max Drawdown** | <-12.63% | 모니터링 | 검증 필요 | 달성 필수 |
| **Win Rate** | 55%+ | 모니터링 | 검증 필요 | 달성 필수 |
| **Uptime** | 99%+ | 달성 필수 | 달성 필수 | 달성 필수 |

---

## ✅ 체크리스트

### Shadow Mode 체크리스트
- [ ] 환경변수 설정 완료
- [ ] 브로커 연결 테스트 성공
- [ ] Shadow Mode 설정 파일 생성
- [ ] Shadow Mode 실행 성공
- [ ] 로그 파일 생성 확인
- [ ] 시그널 생성 확인
- [ ] 최소 1주일 안정적 실행

### Paper Trading 체크리스트
- [ ] Shadow Mode 1주일 이상 실행
- [ ] Paper Trading 계좌 준비
- [ ] Paper Trading 설정 파일 생성
- [ ] Paper Trading 실행 성공
- [ ] 포지션 생성/청산 확인
- [ ] 성능 지표 모니터링
- [ ] 최소 1개월 안정적 실행

### Live Trading 체크리스트
- [ ] Paper Trading 1개월 이상 실행
- [ ] 모든 성능 지표 목표 달성
- [ ] 리스크 관리 파라미터 검증
- [ ] 긴급 중단 절차 숙지
- [ ] 알림 시스템 테스트 완료
- [ ] 계좌 잔고 충분 (최소 $10,000)
- [ ] Live Trading 설정 파일 생성
- [ ] 최종 승인 (사용자 확인)

---

**작성자**: Manus AI  
**날짜**: 2025-11-29  
**버전**: 1.0

**주의**: 라이브 트레이딩은 실제 자금 손실 위험이 있습니다. 충분한 테스트와 검증 후 신중하게 진행하세요.
