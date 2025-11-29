#!/bin/bash
# ARES7 v2 Turbo - Automated Installation Script
# This script sets up a complete production environment on EC2 or any Ubuntu system

set -e  # Exit on error

echo "============================================================================"
echo "ARES7 v2 Turbo - Automated Installation"
echo "============================================================================"
echo ""

# ============================================================================
# 1. System Update
# ============================================================================
echo "[1/8] Updating system packages..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq

# ============================================================================
# 2. Install System Dependencies
# ============================================================================
echo "[2/8] Installing system dependencies..."
sudo apt-get install -y -qq \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev

# ============================================================================
# 3. Create Virtual Environment
# ============================================================================
echo "[3/8] Creating Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# ============================================================================
# 4. Install Python Dependencies
# ============================================================================
echo "[4/8] Installing Python packages..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# ============================================================================
# 5. Create Directories
# ============================================================================
echo "[5/8] Creating necessary directories..."
mkdir -p logs
mkdir -p data/cache
mkdir -p rebalance_logs
mkdir -p reports
mkdir -p reports/images
mkdir -p capacity_checks
mkdir -p backtest/results
mkdir -p optimization/results

# ============================================================================
# 6. Setup Environment Variables
# ============================================================================
echo "[6/8] Setting up environment variables..."
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.template .env
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file and add your API keys!"
    echo "   nano .env"
    echo ""
else
    echo ".env file already exists, skipping..."
fi

# ============================================================================
# 7. Setup Cron Jobs (Optional)
# ============================================================================
echo "[7/8] Cron job setup..."
echo ""
echo "To enable automated validation and capacity checks, add these to crontab:"
echo ""
echo "# Weekly Capacity Check (Monday 10:00 AM)"
echo "0 10 * * 1 cd $(pwd) && source venv/bin/activate && python3 automation/weekly_capacity_check.py >> logs/weekly_capacity.log 2>&1"
echo ""
echo "# Monthly Validation (1st of month, 9:00 AM)"
echo "0 9 1 * * cd $(pwd) && source venv/bin/activate && python3 automation/monthly_validation_schedule.py >> logs/monthly_validation.log 2>&1"
echo ""
echo "Run 'crontab -e' to add these lines."
echo ""

# ============================================================================
# 8. Verification
# ============================================================================
echo "[8/8] Verifying installation..."
python3 -c "import numpy, pandas, scipy; print('✅ Core packages OK')"
python3 -c "import yaml; print('✅ YAML OK')"

echo ""
echo "============================================================================"
echo "✅ Installation Complete!"
echo "============================================================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env file with your API keys:"
echo "     nano .env"
echo ""
echo "  2. Test the system:"
echo "     source venv/bin/activate"
echo "     python3 backtest/run_final_validation.py"
echo ""
echo "  3. Start monitoring API:"
echo "     uvicorn monitoring.api:app --host 0.0.0.0 --port 8000"
echo ""
echo "  4. Setup cron jobs for automation (see above)"
echo ""
echo "============================================================================"
