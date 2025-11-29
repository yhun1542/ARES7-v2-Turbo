# ARES7 v2 Turbo: 최종 배포 가이드 (v5)

**문서 버전:** 5.0
**배포일:** 2025년 11월 29일
**작성자:** Manus AI

---

## 📋 Executive Summary

이 문서는 ARES7 v2 Turbo의 **최종 프로덕션 버전(v5)** 배포 및 운영을 위한 가이드입니다. 모든 최적화가 완료된 시스템을 다운로드하고, 버전을 관리하며, 운영하는 방법을 기술합니다.

**핵심 산출물:**
- **백업 패키지:** `ARES7-v5-FINAL-20251129.tar.gz` (527MB)
- **Git 태그:** `ARES7-v5-FINAL`

**최종 성능 (v5):**
- **Sharpe Ratio:** 3.08
- **Annual Return:** 44.6%
- **Annual Volatility:** 14.5%
- **Max Drawdown:** -6.3%
- **Calmar Ratio:** 7.06

---

## 1. 최종 버전 다운로드

최종 버전은 완전한 백업 패키지 형태로 제공됩니다. 아래 첨부 파일을 다운로드하여 로컬 PC, 외장 하드, 개인 클라우드 등 안전한 곳에 **최소 2곳 이상** 보관하십시오.

- **파일명:** `ARES7-v5-FINAL-20251129.tar.gz`
- **크기:** 527MB
- **포함 내역:**
  - 전체 소스코드 (`ARES-Ultimate-251129/`)
  - 모든 리포트 (`ARES7_Final_Performance_Report_v5.md` 등)
  - 검증/최적화 결과 데이터

---

## 2. 버전 관리 (Git)

모든 변경 사항은 Git 저장소에 기록되었으며, 최종 버전에 태그가 지정되었습니다.

- **최종 태그:** `ARES7-v5-FINAL`
- **최종 커밋 해시:** `fb45c23`

### 특정 버전으로 복원하는 방법

1.  **Git 저장소 확인:**
    ```bash
    cd /path/to/ARES-Ultimate-251129
    git log --oneline -5
    ```

2.  **최종 버전(v5)으로 체크아웃:**
    ```bash
    git checkout ARES7-v5-FINAL
    ```

3.  **이전 버전(v3)으로 복원 (필요 시):**
    ```bash
    git checkout ARES-Ultimate-251129-FINAL
    ```

---

## 3. 운영 및 자동화

시스템은 주간/월간 자동화 스케줄을 통해 스스로를 검증하고 리포트를 생성합니다. 운영자는 이 리포트를 주기적으로 확인하기만 하면 됩니다.

### 3.1. Cron Job 설정

EC2 또는 운영 서버에 아래 Cron Job을 설정하여 자동화를 활성화하십시오.

```bash
# crontab -e

# [주간] 캐파시티 확장 검증 (매주 월요일 오전 10시)
0 10 * * 1 cd /home/ubuntu/ARES-Ultimate-251129 && python3 automation/weekly_capacity_check.py >> logs/weekly_capacity.log 2>&1

# [월간] 통계적 신뢰도(DSR/SPA) 검증 (매월 1일 오전 9시)
0 9 1 * * cd /home/ubuntu/ARES-Ultimate-251129 && python3 automation/monthly_validation_schedule.py >> logs/monthly_validation.log 2>&1
```

### 3.2. 모니터링 대시보드

실시간 시스템 상태는 모니터링 API를 통해 확인할 수 있습니다.

1.  **API 서버 실행:**
    ```bash
    cd /home/ubuntu/ARES-Ultimate-251129
    uvicorn monitoring.api:app --host 0.0.0.0 --port 8000
    ```

2.  **대시보드 접속:**
    - 브라우저에서 `http://<서버_IP>:8000`으로 접속합니다.
    - 실시간 차트, 포지션, 거래 내역, 킬 스위치 등을 확인할 수 있습니다.

---

## 4. 결론

본 가이드를 통해 ARES7 v2 Turbo (v5)의 최종 버전을 안전하게 보관하고, 효율적으로 운영할 수 있습니다. 시스템은 이제 모든 최적화와 자동화가 완료된, 완전한 프로덕션 그레이드 상태입니다.

**시스템 상태:** ✅ **FINAL PRODUCTION VERSION (v5)**
