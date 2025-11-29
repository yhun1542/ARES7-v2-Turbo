"""
Monthly Governance Report Generator
월간 거버넌스 리포트 자동 생성
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    월간 거버넌스 리포트 자동 생성기
    
    ARES7 시스템의 운영 성과, 안정성, 감사 추적 정보를 종합하여
    매월 자동으로 리포트를 생성합니다.
    """
    
    def __init__(self, project_root: str = "/home/ubuntu/ARES-Ultimate-Final"):
        self.project_root = Path(project_root)
        self.state_dir = self.project_root / "state"
        self.log_dir = self.project_root / "logs"
        self.audit_dir = self.project_root / "audit_logs"
        self.report_dir = self.project_root / "reports"
        self.report_dir.mkdir(exist_ok=True)
        
        self.report_template = self._load_template()
    
    def _load_template(self) -> str:
        """리포트 템플릿 로드"""
        return """# ARES7 월간 거버넌스 리포트

**리포트 기간:** {start_date} ~ {end_date}
**생성일:** {generation_date}

---

## 1. 요약 (Executive Summary)

이번 달 ARES7 시스템은 **{overall_status}** 상태로 운영되었습니다.

주요 성과 지표는 다음과 같습니다:

| 지표 (Metric) | 값 (Value) | 벤치마크 (Benchmark) |
|---|---|---|
| **총수익률 (Total Return)** | `{total_return:.2%}` | `{benchmark_return:.2%}` |
| **연환산 수익률 (Annualized Return)** | `{annual_return:.2%}` | - |
| **변동성 (Volatility)** | `{volatility:.2%}` | `{target_volatility:.2%}` |
| **샤프 비율 (Sharpe Ratio)** | `{sharpe_ratio:.2f}` | `{target_sharpe:.2f}` |
| **최대 낙폭 (Max Drawdown)** | `{max_drawdown:.2%}` | `{max_drawdown_limit:.2%}` |

![Equity Curve]({equity_curve_path})

---

## 2. 안정성 및 리스크 관리 (Stability & Risk Management)

### 2.1. 가드레일 시스템 (Guardrails V3)

- **총 가드레일 경고:** `{guardrail_alerts} 건`
- **주요 경고 내역:**
{guardrail_details}

### 2.2. Dead-man Switch

- **비상 종료 이벤트:** `{emergency_events} 건`
- **주요 이벤트 내역:**
{emergency_details}

### 2.3. Autopilot 자동화 시스템

- **총 게이트 평가:** `{autopilot_evaluations} 회`
- **Canary 증액/롤백:** `{autopilot_actions} 건`
- **주요 활동 내역:**
{autopilot_details}

---

## 3. 통계적 유의성 검증 (Statistical Validation)

탐색 편향을 보정한 전략의 통계적 유의성을 검증했습니다.

| 검증 항목 | 결과 | p-value | 유의성 (95%) |
|---|---|---|---|
| **Deflated Sharpe Ratio (DSR)** | `{dsr:.4f}` | `{dsr_pvalue:.4f}` | `{dsr_significant}` |
| **Superior Predictive Ability (SPA)** | `{spa_statistic:.4f}` | `{spa_pvalue:.4f}` | `{spa_superior}` |

**종합 결론:** `{overall_validation}`

---

## 4. 감사 추적성 (Audit Trail)

이번 리포트 생성에 사용된 환경 및 설정 정보입니다.

{audit_section}

---

## 5. 첨부 산출물 (Attachments)

- **상세 거래 내역:** `trades_{report_period}.csv`
- **일별 수익률 데이터:** `returns_{report_period}.csv`
- **감사 로그 전문:** `audit_{report_period}.json`

"""
    
    def generate_report(self, year: int, month: int):
        """
        지정된 연월의 리포트 생성
        
        Args:
            year: 연도
            month: 월
        """
        start_date = datetime(year, month, 1)
        end_date = (start_date + timedelta(days=31)).replace(day=1) - timedelta(days=1)
        
        report_period = f"{year}{month:02d}"
        report_path = self.report_dir / f"ARES7_Governance_Report_{report_period}.md"
        image_dir = self.report_dir / "images"
        image_dir.mkdir(exist_ok=True)
        
        logger.info(f"Generating report for {year}-{month}...")
        
        # 1. 데이터 수집
        data = self._collect_data(start_date, end_date)
        
        # 2. 시각화 생성
        equity_curve_path = self._generate_equity_curve(
            data["returns"], 
            image_dir / f"equity_curve_{report_period}.png"
        )
        data["equity_curve_path"] = equity_curve_path
        
        # 3. 감사 정보 주입
        data["audit_section"] = self._get_audit_section()
        
        # 4. 템플릿 채우기
        report_content = self.report_template.format(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            generation_date=datetime.now().strftime("%Y-%m-%d"),
            report_period=report_period,
            **data
        )
        
        # 5. 리포트 저장
        with open(report_path, "w") as f:
            f.write(report_content)
        
        logger.info(f"Report generated: {report_path}")
        
        # 6. 첨부 파일 생성 (예시)
        self._create_attachments(report_period)
    
    def _collect_data(self, start_date: datetime, end_date: datetime) -> dict:
        """리포트 데이터 수집"""
        # 실제 구현에서는 각 모듈의 로그/상태 파일에서 데이터를 읽어옵니다.
        # 여기서는 테스트를 위해 더미 데이터를 생성합니다.
        
        # 수익률 데이터
        returns = pd.Series(np.random.randn(30) * 0.01, index=pd.date_range(start=start_date, periods=30))
        benchmark_returns = pd.Series(np.random.randn(30) * 0.008, index=returns.index)
        
        # 성과 지표
        total_return = (1 + returns).prod() - 1
        annual_return = total_return * (252 / len(returns))
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        equity_curve = (1 + returns).cumprod()
        max_drawdown = (equity_curve / equity_curve.cummax() - 1).min()
        
        return {
            "overall_status": "정상",
            "returns": returns,
            "total_return": total_return,
            "benchmark_return": (1 + benchmark_returns).prod() - 1,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "target_volatility": 0.12,
            "target_sharpe": 2.0,
            "max_drawdown_limit": -0.08,
            "guardrail_alerts": 0,
            "guardrail_details": "- 경고 없음",
            "emergency_events": 0,
            "emergency_details": "- 이벤트 없음",
            "autopilot_evaluations": 12 * 24 * 30, # 5분 주기
            "autopilot_actions": 2,
            "autopilot_details": "- Canary 증액 (50% -> 60%)\n- Canary 증액 (60% -> 100%)",
            "dsr": 2.15, "dsr_pvalue": 0.015, "dsr_significant": "Yes",
            "spa_statistic": 2.8, "spa_pvalue": 0.002, "spa_superior": "Yes",
            "overall_validation": "✅ 통계적으로 유의미함"
        }
    
    def _generate_equity_curve(self, returns: pd.Series, save_path: Path) -> str:
        """자산 곡선 차트 생성"""
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 6))
        
        equity_curve = (1 + returns).cumprod()
        ax.plot(equity_curve.index, equity_curve, label="ARES7", color="#007ACC")
        
        ax.set_title("Equity Curve", fontsize=16)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        
        logger.info(f"Equity curve saved: {save_path}")
        return str(save_path.relative_to(self.report_dir))
    
    def _get_audit_section(self) -> str:
        """감사 섹션 생성"""
        # 실제로는 audit_trail.py를 사용하여 생성
        return """### 실행 환경
- **타임스탬프:** 2025-12-01T10:00:00Z
- **Python 버전:** 3.11.0
- **Git Commit:** a1b2c3d4

### 의존성 해시
- **Config SHA256:** `e3b0c442...`
- **Requirements SHA256:** `f1d2d2f9...`
"""
    
    def _create_attachments(self, report_period: str):
        """첨부 파일 생성 (더미)"""
        # 거래 내역
        trades_df = pd.DataFrame({
            "timestamp": [datetime.now()],
            "symbol": ["AAPL"],
            "action": ["BUY"],
            "quantity": [100],
            "price": [150.0]
        })
        trades_df.to_csv(self.report_dir / f"trades_{report_period}.csv", index=False)
        
        # 수익률 데이터
        returns_df = pd.DataFrame({"returns": np.random.randn(30) * 0.01})
        returns_df.to_csv(self.report_dir / f"returns_{report_period}.csv", index=False)
        
        # 감사 로그
        audit_log = {"operation": "report_generation", "timestamp": datetime.now().isoformat()}
        with open(self.report_dir / f"audit_{report_period}.json", "w") as f:
            json.dump(audit_log, f, indent=2)
        
        logger.info(f"Attachments created for {report_period}")


# 테스트 코드
if __name__ == "__main__":
    print("=" * 60)
    print("Monthly Governance Report Generator Test")
    print("=" * 60)
    print()
    
    # 리포트 생성기 초기화
    generator = ReportGenerator()
    
    # 2025년 11월 리포트 생성
    generator.generate_report(year=2025, month=11)
    
    print()
    print("✅ Report generation complete!")
    print(f"Check the reports directory: {generator.report_dir}")
