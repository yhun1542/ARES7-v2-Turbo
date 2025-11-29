# AWS GPU Deployment Guide

## ARES-Ultimate-251129 AWS 배포 가이드

### 권장 인스턴스

| 용도 | 인스턴스 | vCPU | RAM | GPU | 비용/월 |
|------|----------|------|-----|-----|---------|
| 개발/테스트 | t3.xlarge | 4 | 16GB | - | ~$120 |
| 백테스트 | g4dn.xlarge | 4 | 16GB | T4 | ~$380 |
| 라이브 트레이딩 | c6i.xlarge | 4 | 8GB | - | ~$123 |
| 고성능 백테스트 | g4dn.2xlarge | 8 | 32GB | T4 | ~$680 |

### 1. EC2 인스턴스 설정

```bash
# Ubuntu 22.04 LTS AMI 사용
# Security Group: 22 (SSH), 7497 (TWS Paper), 7496 (TWS Live)

# 인스턴스 시작 후 SSH 접속
ssh -i your-key.pem ubuntu@<instance-ip>
```

### 2. 시스템 패키지 설치

```bash
# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# 필수 패키지
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    tmux \
    htop \
    redis-server \
    postgresql \
    postgresql-contrib

# Docker (선택사항)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
```

### 3. GPU 드라이버 설치 (g4dn 인스턴스)

```bash
# NVIDIA 드라이버
sudo apt install -y nvidia-driver-535

# CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run --toolkit --silent

# 환경변수
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 확인
nvidia-smi
```

### 4. Python 환경 설정

```bash
# 프로젝트 클론
cd ~
git clone <your-repo> ARES-Ultimate-251129
cd ARES-Ultimate-251129

# 가상환경 생성
python3.11 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install --upgrade pip
pip install -e .

# GPU용 PyTorch (선택사항)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 5. 환경변수 설정

```bash
# .env 파일 생성
cat > .env << 'EOF'
# Polygon.io
POLYGON_API_KEY=your_polygon_key

# NASDAQ Data Link (SF1)
NASDAQ_DATA_LINK_API_KEY=your_nasdaq_key

# FRED
FRED_API_KEY=your_fred_key

# Tavily
TAVILY_API_KEY=your_tavily_key

# SEC API
SEC_API_KEY=your_sec_key

# IBKR (TWS/Gateway)
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1

# KIS (한국투자증권)
KIS_APP_KEY=your_kis_key
KIS_APP_SECRET=your_kis_secret
KIS_ACCOUNT_NO=your_account_no

# Telegram Alerts
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
EOF

# 환경변수 로드
export $(cat .env | xargs)
```

### 6. TWS/IB Gateway 설정 (IBKR)

```bash
# IB Gateway 다운로드
wget https://download2.interactivebrokers.com/installers/ibgateway/latest-standalone/ibgateway-latest-standalone-linux-x64.sh

# 설치
chmod +x ibgateway-latest-standalone-linux-x64.sh
./ibgateway-latest-standalone-linux-x64.sh

# Xvfb (가상 디스플레이)
sudo apt install -y xvfb
Xvfb :1 -screen 0 1024x768x16 &
export DISPLAY=:1

# IB Gateway 실행
~/Jts/ibgateway/*/ibgateway &
```

### 7. Systemd 서비스 등록

```bash
# 서비스 파일 생성
sudo cat > /etc/systemd/system/ares-ultimate.service << 'EOF'
[Unit]
Description=ARES-Ultimate Trading System
After=network.target redis-server.service postgresql.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ARES-Ultimate-251129
Environment="PATH=/home/ubuntu/ARES-Ultimate-251129/venv/bin"
EnvironmentFile=/home/ubuntu/ARES-Ultimate-251129/.env
ExecStart=/home/ubuntu/ARES-Ultimate-251129/venv/bin/python -m orchestration.live_orchestrator
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 서비스 활성화
sudo systemctl daemon-reload
sudo systemctl enable ares-ultimate
sudo systemctl start ares-ultimate

# 상태 확인
sudo systemctl status ares-ultimate
```

### 8. 로그 모니터링

```bash
# 실시간 로그
sudo journalctl -u ares-ultimate -f

# 애플리케이션 로그
tail -f ~/ARES-Ultimate-251129/logs/ares.log
```

### 9. 백업 설정

```bash
# 일일 백업 스크립트
cat > ~/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR=/home/ubuntu/backups
DATE=$(date +%Y%m%d)

# 설정 백업
tar -czf $BACKUP_DIR/config_$DATE.tar.gz ~/ARES-Ultimate-251129/config/

# 데이터베이스 백업
pg_dump ares_db > $BACKUP_DIR/db_$DATE.sql

# S3 업로드 (선택사항)
# aws s3 cp $BACKUP_DIR/config_$DATE.tar.gz s3://your-bucket/backups/
EOF

chmod +x ~/backup.sh

# Cron 등록 (매일 새벽 3시)
(crontab -l 2>/dev/null; echo "0 3 * * * /home/ubuntu/backup.sh") | crontab -
```

### 10. 모니터링 대시보드

```bash
# Prometheus + Grafana (Docker)
docker run -d --name prometheus -p 9090:9090 prom/prometheus
docker run -d --name grafana -p 3000:3000 grafana/grafana
```

---

## 문제 해결

### GPU 메모리 부족
```bash
# GPU 메모리 확인
nvidia-smi

# PyTorch 메모리 해제
import torch
torch.cuda.empty_cache()
```

### IBKR 연결 실패
```bash
# TWS/Gateway 포트 확인
netstat -tlnp | grep 749

# API 설정 확인 (TWS > Edit > Global Configuration > API)
# - Enable ActiveX and Socket Clients
# - Socket port: 7497 (paper) / 7496 (live)
```

### 권한 문제
```bash
# 로그 디렉토리 권한
sudo chown -R ubuntu:ubuntu ~/ARES-Ultimate-251129/logs/

# 소켓 권한
sudo chmod 777 /var/run/postgresql/.s.PGSQL.5432
```
