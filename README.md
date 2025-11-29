# ARES-Ultimate-251129

## Level 6 자율 금융 시스템

ARES7 QM Regime Turbo 전략을 핵심 알파 엔진으로, IBKR/KIS 브로커 연동을 통한 자율 거래 시스템.

### 성능 목표

| 지표 | In-Sample | Out-of-Sample |
|------|-----------|---------------|
| Sharpe Ratio | 3.86 | 4.37 |
| Max Drawdown | -12.63% | -10.10% |

### 아키텍처

```
ARES-Ultimate-251129/
├── config/                 # 설정 파일
│   ├── ares7_qm_turbo_final_251129.yaml
│   ├── brokers.yaml
│   └── data_sources.yaml
├── core/                   # 핵심 인터페이스 & 유틸리티
│   ├── interfaces.py       # IBroker, IStrategy, IRiskManager
│   └── utils.py
├── data/                   # 데이터 클라이언트
│   ├── polygon_client.py   # 가격 데이터
│   ├── sf1_client.py       # 펀더멘탈 (Sharadar)
│   ├── fred_client.py      # 거시경제
│   ├── tavily_client.py    # AI 뉴스 검색
│   └── sec_client.py       # SEC 공시
├── engines/                # 전략 엔진
│   ├── ares7_qm_regime/    # QM Regime Turbo
│   └── aresx_v110/         # 실행 엔진
├── risk/                   # 리스크 관리
│   ├── regime_filter.py    # 시장 레짐 판별
│   ├── aarm_core.py        # Turbo AARM
│   └── cvar_utils.py       # CVaR 계산
├── ensemble/               # 앙상블 전략
│   ├── dynamic_ensemble.py # 레짐별 블렌딩
│   └── turbo_aarm.py       # 전체 파이프라인
├── brokers/                # 브로커 클라이언트
│   ├── ibkr_client.py      # Interactive Brokers
│   └── kis_client.py       # 한국투자증권
├── orchestration/          # 오케스트레이션
│   ├── live_orchestrator.py
│   └── scheduler.py
├── backtest/               # 백테스트
│   ├── run_backtest.py
│   └── metrics.py
├── deployment/             # 배포
│   ├── aws_gpu_setup.md
│   └── scripts/
└── scripts/                # 실행 스크립트
    └── main.py
```

### 핵심 컴포넌트

#### 1. ARES7 QM Regime Strategy
- **Quality Factors**: ROE, ROIC, Gross Margin, Current Ratio, D/E
- **Momentum Factors**: 6개월/12개월 수익률
- **QM Overlay**: Top 20% +4%, Bottom 20% -4%
- **PIT Compliance**: 90일 지연

#### 2. Regime Filter
- **BULL**: SPX > MA200, VIX < 25, 6M/12M 수익률 양수
- **BEAR**: SPX < MA200, 6M 수익률 음수
- **HIGH_VOL**: VIX >= 30 또는 이벤트 리스크
- **NEUTRAL**: 기타

#### 3. Turbo AARM
- **Volatility Targeting**: 목표 변동성 18%
- **Drawdown Scaling**: -5%부터 단계적 축소
- **Circuit Breaker**: -6% 트리거, 40% 축소
- **레버리지**: 기본 1.2x, 최대 1.8x

#### 4. Dynamic Ensemble
| 레짐 | QM | Defensive |
|------|-----|-----------|
| BULL | 100% | 0% |
| BEAR | 60% | 40% |
| HIGH_VOL | 50% | 50% |
| NEUTRAL | 30% | 70% |

### 설치

```bash
# 클론
git clone <repo> ARES-Ultimate-251129
cd ARES-Ultimate-251129

# 가상환경
python3.11 -m venv venv
source venv/bin/activate

# 설치
pip install -e .
```

### 환경변수

```bash
# .env 파일 생성
POLYGON_API_KEY=your_key
NASDAQ_DATA_LINK_API_KEY=your_key
FRED_API_KEY=your_key
TAVILY_API_KEY=your_key
SEC_API_KEY=your_key

# IBKR
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1

# KIS
KIS_APP_KEY=your_key
KIS_APP_SECRET=your_secret
KIS_ACCOUNT_NO=your_account
```

### 실행

#### 백테스트
```bash
python scripts/main.py backtest --synthetic
```

#### 라이브 (Shadow Mode)
```bash
# IBKR
python scripts/main.py live-ibkr

# KIS
python scripts/main.py live-kis
```

#### 라이브 (Real Orders - 주의!)
```bash
python scripts/main.py live-ibkr --live
```

### 데이터 소스

| 소스 | 용도 | 기능 |
|------|------|------|
| Polygon.io | 가격 | OHLCV, 모멘텀 팩터 |
| SF1/Sharadar | 펀더멘탈 | Quality 팩터 |
| FRED | 매크로 | VIX, 스프레드, 레짐 |
| Tavily | 뉴스 | 센티먼트, 이벤트 리스크 |
| SEC API | 공시 | 8-K 이벤트 |

### 브로커 연동

#### IBKR
- TWS 또는 IB Gateway 실행 필요
- API 포트: 7497 (paper), 7496 (live)
- ib_insync 라이브러리 사용

#### KIS (한국투자증권)
- REST API 기반
- 모의투자/실전투자 지원
- 토큰 자동 갱신

### 리스크 관리

1. **Position Limits**: 종목당 최대 10%
2. **Leverage Limits**: 최대 1.8x
3. **Drawdown Scaling**: 단계적 포지션 축소
4. **Circuit Breaker**: 6% 손실 시 활성화
5. **Event Risk**: 뉴스/공시 기반 리스크 플래그

### AWS 배포

상세 가이드: `deployment/aws_gpu_setup.md`

```bash
# 권장 인스턴스
- 백테스트: g4dn.xlarge (T4 GPU)
- 라이브: c6i.xlarge
```

### 라이선스

Private - All Rights Reserved

### 연락처

Jason - Quantitative Finance Developer
