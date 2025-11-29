# ARES7 v2 Turbo: EC2 배포 가이드 (v5.1)

**문서 버전:** 5.1
**배포일:** 2025년 11월 29일
**작성자:** Manus AI

---

## 📋 Executive Summary

이 문서는 ARES7 v2 Turbo의 **최종 프로덕션 버전(v5.1)**을 Amazon EC2 또는 유사한 Ubuntu 환경에 배포하는 방법을 상세히 기술합니다. `install.sh` 스크립트를 통해 모든 과정을 자동화할 수 있습니다.

**배포 목표:** GitHub 리포지토리에서 코드를 복제하고, 단일 스크립트를 실행하여 완전한 운영 환경을 구축합니다.

---

## 1. 사전 준비 사항

1.  **EC2 인스턴스 생성:**
    - **OS:** Ubuntu 22.04 LTS 이상
    - **인스턴스 타입:** `t3.medium` 이상 권장 (최소 2 vCPU, 4GB RAM)
    - **보안 그룹:** SSH(22), HTTP(8000) 포트 허용

2.  **API 키 준비:**
    - Polygon.io, IBKR/KIS 등 필요한 API 키를 미리 준비합니다.

---

## 2. 자동 설치 (One-Click Deployment)

EC2 인스턴스에 SSH로 접속한 후, 아래 3단계만 실행하면 모든 설치가 완료됩니다.

### 2.1. GitHub 리포지토리 복제

```bash
git clone https://github.com/yhun1542/ARES7-v2-Turbo.git
cd ARES7-v2-Turbo
```

### 2.2. 설치 스크립트 실행

`install.sh` 스크립트는 시스템 업데이트, 의존성 설치, 가상 환경 생성, 디렉토리 생성, 환경 변수 설정까지 모든 과정을 자동으로 처리합니다.

```bash
./install.sh
```

### 2.3. 환경 변수 설정

설치가 완료되면 `.env` 파일이 자동으로 생성됩니다. 이 파일에 준비해 둔 API 키와 계정 정보를 입력합니다.

```bash
nano .env
```

- **필수:** `POLYGON_API_KEY`, 브로커(IBKR 또는 KIS) 정보
- **선택:** OpenAI, Slack, AWS 등

---

## 3. 시스템 검증 및 실행

### 3.1. 가상 환경 활성화

모든 명령어는 가상 환경 내에서 실행해야 합니다.

```bash
source venv/bin/activate
```

### 3.2. 최종 검증 백테스트

시스템이 올바르게 설치되었는지 최종 검증 백테스트를 실행하여 성능을 확인합니다.

```bash
python3 backtest/run_final_validation.py
```

- **예상 결과:** Sharpe 3.0 이상, Vol 14.5% 수준의 성능 지표가 출력되어야 합니다.

### 3.3. 모니터링 API 서버 실행

실시간 시스템 상태를 확인하기 위해 모니터링 API 서버를 실행합니다.

```bash
uvicorn monitoring.api:app --host 0.0.0.0 --port 8000
```

- **접속:** 웹 브라우저에서 `http://<EC2_Public_IP>:8000`으로 접속하여 대시보드를 확인합니다.

---

## 4. 운영 자동화 (Cron Job)

시스템의 자가 검증 및 리포팅을 위해 Cron Job을 설정합니다.

1.  **Crontab 편집기 열기:**
    ```bash
    crontab -e
    ```

2.  **스케줄 추가:**
    `install.sh` 실행 시 출력된 Cron Job 설정을 복사하여 붙여넣습니다.

    ```bash
    # Weekly Capacity Check (Monday 10:00 AM)
    0 10 * * 1 cd /home/ubuntu/ARES7-v2-Turbo && source venv/bin/activate && python3 automation/weekly_capacity_check.py >> logs/weekly_capacity.log 2>&1

    # Monthly Validation (1st of month, 9:00 AM)
    0 9 1 * * cd /home/ubuntu/ARES7-v2-Turbo && source venv/bin/activate && python3 automation/monthly_validation_schedule.py >> logs/monthly_validation.log 2>&1
    ```

---

## 5. 결론

본 가이드를 통해 ARES7 v2 Turbo (v5.1)를 EC2에 성공적으로 배포하고, 완전한 자동 운영 환경을 구축할 수 있습니다. 모든 설정과 스크립트가 GitHub에 포함되어 있어, 언제든지 재현 가능한 배포가 가능합니다.

**시스템 상태:** ✅ **READY FOR EC2 DEPLOYMENT**
