"""
Monthly Validation Schedule
DSR/SPA 통계적 검증 월간 자동 실행
"""

import sys
from pathlib import Path
import subprocess
import logging
from datetime import datetime

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_monthly_validation():
    """월간 검증 실행"""
    
    logger.info("=" * 80)
    logger.info("Monthly Validation - DSR/SPA Statistical Tests")
    logger.info("=" * 80)
    logger.info(f"Execution time: {datetime.now().isoformat()}")
    logger.info("")
    
    # 1. 종합 검증 실행
    logger.info("Running comprehensive validation suite...")
    
    validation_script = project_root / "validation" / "comprehensive_validation_suite.py"
    
    try:
        result = subprocess.run(
            ["python3", str(validation_script)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            logger.info("✅ Comprehensive validation completed successfully")
            logger.info("")
            logger.info("Output:")
            logger.info(result.stdout)
        else:
            logger.error("❌ Comprehensive validation failed")
            logger.error(result.stderr)
            return False
        
    except subprocess.TimeoutExpired:
        logger.error("❌ Validation timeout (5 minutes)")
        return False
    
    except Exception as e:
        logger.error(f"❌ Validation error: {e}")
        return False
    
    # 2. 월간 거버넌스 리포트 생성
    logger.info("Generating monthly governance report...")
    
    from governance.report_generator import ReportGenerator
    
    try:
        now = datetime.now()
        generator = ReportGenerator()
        report_path = generator.generate_report(now.year, now.month)
        
        logger.info(f"✅ Monthly report generated: {report_path}")
        
    except Exception as e:
        logger.error(f"❌ Report generation error: {e}")
        return False
    
    # 3. 결과 요약
    logger.info("")
    logger.info("=" * 80)
    logger.info("Monthly Validation Summary")
    logger.info("=" * 80)
    logger.info("✅ DSR/SPA statistical tests completed")
    logger.info("✅ Monthly governance report generated")
    logger.info("✅ All validations passed")
    logger.info("=" * 80)
    
    return True


def setup_cron_job():
    """
    Cron job 설정 가이드
    
    매월 1일 오전 9시 실행:
    0 9 1 * * cd /home/ubuntu/ARES-Ultimate-251129 && python3 automation/monthly_validation_schedule.py >> logs/monthly_validation.log 2>&1
    """
    
    print("=" * 80)
    print("Monthly Validation Cron Job Setup")
    print("=" * 80)
    print()
    print("Add the following line to your crontab:")
    print()
    print("0 9 1 * * cd /home/ubuntu/ARES-Ultimate-251129 && python3 automation/monthly_validation_schedule.py >> logs/monthly_validation.log 2>&1")
    print()
    print("This will run the validation on the 1st of every month at 9:00 AM")
    print()
    print("To edit crontab:")
    print("  $ crontab -e")
    print()
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monthly Validation Schedule")
    parser.add_argument(
        "--setup-cron",
        action="store_true",
        help="Show cron job setup instructions"
    )
    
    args = parser.parse_args()
    
    if args.setup_cron:
        setup_cron_job()
    else:
        success = run_monthly_validation()
        sys.exit(0 if success else 1)
