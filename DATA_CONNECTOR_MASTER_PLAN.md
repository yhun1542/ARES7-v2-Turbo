# ARES 통합 데이터 커넥터 마스터 플랜
**목표**: 5개 데이터 소스를 통합하여 알파/베타 생성에 유기적으로 연결

---

## 1. Polygon (Massive.com) - 가격 및 시장 데이터

### 우선순위 1: 핵심 가격 데이터
- ✅ **Aggregates (Bars)**: 일/분/시간 OHLCV 데이터
- ✅ **Quotes**: 실시간 호가 데이터
- ✅ **Trades**: 실시간 체결 데이터
- ✅ **Snapshots**: 현재 시장 상태

### 우선순위 2: 기술 지표
- **SMA (Simple Moving Average)**: 이동평균
- **EMA (Exponential Moving Average)**: 지수이동평균
- **RSI (Relative Strength Index)**: 상대강도지수
- **MACD**: 이동평균수렴확산
- **Bollinger Bands**: 볼린저밴드

### 우선순위 3: 펀더멘탈 데이터
- **Balance Sheets**: 재무상태표
- **Cash Flow Statements**: 현금흐름표
- **Income Statements**: 손익계산서
- **Dividends**: 배당 정보
- **Splits**: 주식분할 정보

### 우선순위 4: 시장 구조 데이터
- **Short Interest**: 공매도 잔고
- **Short Volume**: 공매도 거래량
- **Insider Transactions**: 내부자 거래
- **Market Movers**: 시장 주도주

### 우선순위 5: 옵션 데이터
- **Options Contracts**: 옵션 계약 정보
- **Options Snapshot**: 옵션 스냅샷
- **Implied Volatility**: 내재변동성

---

## 2. Nasdaq (Sharadar) - 펀더멘탈 데이터

### 우선순위 1: SF1 Core Fundamentals
- **ROE (Return on Equity)**: 자기자본이익률
- **ROA (Return on Assets)**: 총자산이익률
- **ROIC (Return on Invested Capital)**: 투하자본이익률
- **Profit Margin**: 이익률
- **Revenue Growth**: 매출 성장률
- **EPS Growth**: 주당순이익 성장률
- **Debt to Equity**: 부채비율
- **Current Ratio**: 유동비율
- **Quick Ratio**: 당좌비율

### 우선순위 2: SF1 Derived Metrics
- **Quality Score**: 품질 점수 (ROE, ROA, ROIC 조합)
- **Growth Score**: 성장 점수 (매출/EPS 성장률)
- **Value Score**: 가치 점수 (P/E, P/B, P/S)
- **Financial Health**: 재무건전성 (부채비율, 유동비율)

### 우선순위 3: TICKERS Metadata
- **Sector**: 섹터 정보
- **Industry**: 산업 정보
- **Market Cap**: 시가총액
- **IPO Date**: 상장일

---

## 3. FRED - 매크로 경제 지표

### 우선순위 1: 금리 및 통화
- **DFF (Federal Funds Rate)**: 연방기금금리
- **DGS10 (10-Year Treasury)**: 10년물 국채 수익률
- **DGS2 (2-Year Treasury)**: 2년물 국채 수익률
- **T10Y2Y (Yield Curve)**: 수익률 곡선 (10Y-2Y)
- **DTWEXBGS (Dollar Index)**: 달러 인덱스

### 우선순위 2: 변동성 및 리스크
- **VIXCLS (VIX)**: 변동성 지수
- **BAMLH0A0HYM2 (High Yield Spread)**: 하이일드 스프레드
- **T10YIE (Inflation Expectations)**: 인플레이션 기대

### 우선순위 3: 경제 활동
- **UNRATE (Unemployment Rate)**: 실업률
- **CPIAUCSL (CPI)**: 소비자물가지수
- **GDP (Real GDP)**: 실질 GDP
- **INDPRO (Industrial Production)**: 산업생산지수
- **RSXFS (Retail Sales)**: 소매판매

### 우선순위 4: 신용 및 유동성
- **M2SL (M2 Money Supply)**: 통화량 M2
- **TOTRESNS (Total Reserves)**: 총 지준금
- **WALCL (Fed Balance Sheet)**: 연준 대차대조표

---

## 4. Alpha Vantage - 기술 지표 및 대체 데이터

### 우선순위 1: 고급 기술 지표
- **ADX (Average Directional Index)**: 평균방향지수
- **AROON**: 아룬 지표
- **BBANDS (Bollinger Bands)**: 볼린저밴드
- **CCI (Commodity Channel Index)**: 상품채널지수
- **STOCH (Stochastic)**: 스토캐스틱

### 우선순위 2: 뉴스 및 감성 분석
- **News Sentiment**: 뉴스 감성 분석
- **Insider Transactions**: 내부자 거래 (SEC Form 3/4/5)

### 우선순위 3: 크립토 데이터
- **Crypto Quotes**: 암호화폐 시세
- **Crypto Technical Indicators**: 암호화폐 기술 지표

---

## 5. SEC - 공시 및 규제 데이터

### 우선순위 1: 핵심 공시
- **10-K (Annual Report)**: 연간 보고서
- **10-Q (Quarterly Report)**: 분기 보고서
- **8-K (Current Report)**: 수시 보고서
- **Form 3/4/5 (Insider Trading)**: 내부자 거래

### 우선순위 2: 소유권 정보
- **13F (Institutional Holdings)**: 기관 보유 현황
- **13D/G (Beneficial Ownership)**: 대량보유 보고

### 우선순위 3: 텍스트 분석
- **Full-Text Search**: 전문 검색
- **Content Extraction**: 내용 추출
- **XBRL to JSON**: 재무제표 데이터

---

## 알파/베타 생성 파이프라인

### Alpha Signals (초과수익 추구)

#### 1. Quality Alpha (Sharadar SF1)
```
Quality Score = 0.4 * ROE + 0.3 * ROA + 0.3 * ROIC
Financial Health = 0.5 * Current_Ratio + 0.5 * (1 / Debt_to_Equity)
Quality Alpha = 0.6 * Quality_Score + 0.4 * Financial_Health
```

#### 2. Momentum Alpha (Polygon)
```
Price Momentum = (Price_now / Price_3m_ago) - 1
Volume Momentum = (Volume_now / Volume_3m_avg) - 1
Momentum Alpha = 0.7 * Price_Momentum + 0.3 * Volume_Momentum
```

#### 3. Technical Alpha (Polygon + Alpha Vantage)
```
RSI_Signal = (RSI - 50) / 50  # Normalized
MACD_Signal = MACD / Price
BB_Signal = (Price - BB_Middle) / (BB_Upper - BB_Lower)
Technical Alpha = 0.4 * RSI_Signal + 0.3 * MACD_Signal + 0.3 * BB_Signal
```

#### 4. Value Alpha (Sharadar SF1)
```
PE_Score = 1 / (P/E Ratio)  # Lower is better
PB_Score = 1 / (P/B Ratio)
PS_Score = 1 / (P/S Ratio)
Value Alpha = 0.4 * PE_Score + 0.3 * PB_Score + 0.3 * PS_Score
```

#### 5. Sentiment Alpha (Alpha Vantage + SEC)
```
News_Sentiment = News_Sentiment_Score  # -1 to 1
Insider_Signal = (Insider_Buys - Insider_Sells) / Total_Shares
Sentiment Alpha = 0.6 * News_Sentiment + 0.4 * Insider_Signal
```

### Beta Exposure (시장 리스크)

#### 1. Market Beta (Polygon)
```
Market_Beta = Cov(Stock_Returns, SPY_Returns) / Var(SPY_Returns)
```

#### 2. Sector Beta (Sharadar)
```
Sector_Beta = Cov(Stock_Returns, Sector_Returns) / Var(Sector_Returns)
```

#### 3. Macro Beta (FRED)
```
Rate_Beta = Correlation(Stock_Returns, DGS10_Changes)
Dollar_Beta = Correlation(Stock_Returns, DTWEXBGS_Changes)
VIX_Beta = Correlation(Stock_Returns, VIX_Changes)
Macro_Beta = 0.4 * Rate_Beta + 0.3 * Dollar_Beta + 0.3 * VIX_Beta
```

### Regime Detection (FRED)

```python
def detect_regime(vix, yield_curve, unemployment):
    """
    BULL: VIX < 20, Yield Curve > 0, Unemployment < 5%
    BEAR: VIX > 30, Yield Curve < 0, Unemployment > 6%
    HIGH_VOL: VIX > 25
    NEUTRAL: Otherwise
    """
    if vix > 30 or yield_curve < 0:
        return "BEAR"
    elif vix > 25:
        return "HIGH_VOL"
    elif vix < 20 and yield_curve > 0 and unemployment < 5:
        return "BULL"
    else:
        return "NEUTRAL"
```

### Combined Alpha/Beta Strategy

```python
# ARES7 QM Regime Strategy
Quality_Weight = 0.6
Momentum_Weight = 0.4

QM_Alpha = Quality_Weight * Quality_Alpha + Momentum_Weight * Momentum_Alpha

# Regime-adjusted weights
if regime == "BULL":
    QM_Alpha *= 1.2  # Increase exposure
elif regime == "BEAR":
    QM_Alpha *= 0.5  # Reduce exposure
elif regime == "HIGH_VOL":
    QM_Alpha *= 0.7  # Moderate exposure

# Beta neutralization (optional)
Neutralized_Alpha = QM_Alpha - Market_Beta * Market_Return

# Final signal
Final_Signal = Neutralized_Alpha
```

---

## 구현 우선순위

### Phase 1: 핵심 데이터 (1-2일)
1. ✅ Polygon Aggregates (가격 데이터)
2. ✅ Sharadar SF1 (펀더멘탈 데이터)
3. ✅ FRED VIX, Yield Curve (레짐 감지)

### Phase 2: 알파 시그널 (2-3일)
4. ✅ Quality Alpha 계산
5. ✅ Momentum Alpha 계산
6. ✅ QM Overlay 통합

### Phase 3: 고급 기능 (3-5일)
7. ✅ Technical Alpha (RSI, MACD, BB)
8. ✅ Value Alpha (P/E, P/B, P/S)
9. ✅ Sentiment Alpha (뉴스, 내부자 거래)

### Phase 4: 베타 관리 (1-2일)
10. ✅ Market Beta 계산
11. ✅ Sector Beta 계산
12. ✅ Macro Beta 계산

### Phase 5: 최적화 (1-2일)
13. ✅ GPU 가속
14. ✅ 데이터 캐싱
15. ✅ 병렬 처리

---

**총 예상 기간**: 8-14일  
**목표 성능**: Sharpe 3.86 (OOS 4.37)
