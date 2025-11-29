"""
ARES7 QM Regime 통합 백테스트 엔진
===================================
통합 데이터 커넥터 + ARES7 전략 + Turbo AARM

목표 성능:
- Full Sharpe: 3.86
- OOS Sharpe: 4.37
- 연율화 수익률: 67.74%
- MDD: -12.63%
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging

# Import unified data connector
from data.unified_data_connector import UnifiedDataConnector

# Import ARES7 components
from risk.adaptive_asymmetric_risk_manager import AdaptiveAsymmetricRiskManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ARES7IntegratedBacktest:
    """ARES7 QM Regime 통합 백테스트"""
    
    def __init__(
        self,
        universe: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 1_000_000.0
    ):
        self.universe = universe
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        
        # Components
        self.data_connector = UnifiedDataConnector()
        self.risk_manager = None  # Will be initialized later
        
        # Data storage
        self.price_data = {}
        self.alpha_signals = {}
        self.beta_exposure = {}
        self.regime_data = None
        
        # Backtest results
        self.portfolio_values = []
        self.positions = []
        self.trades = []
        
    async def load_data(self):
        """데이터 로딩"""
        logger.info(f"Loading data for {len(self.universe)} symbols...")
        
        # Load price data and alpha signals for all symbols
        tasks = []
        for symbol in self.universe:
            tasks.append(self._load_symbol_data(symbol))
        
        results = await asyncio.gather(*tasks)
        
        for symbol, (price_df, alpha_df, beta_df) in zip(self.universe, results):
            if not price_df.empty:
                self.price_data[symbol] = price_df
                self.alpha_signals[symbol] = alpha_df
                self.beta_exposure[symbol] = beta_df
        
        # Load macro data and regime
        logger.info("Loading macro data and detecting regime...")
        macro_data = await self.data_connector.get_macro_indicators(
            self.start_date,
            self.end_date
        )
        self.regime_data = self.data_connector.detect_regime(macro_data)
        
        logger.info(f"✅ Data loaded for {len(self.price_data)} symbols")
        
    async def _load_symbol_data(self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """개별 종목 데이터 로딩"""
        try:
            # Price data
            price_df = await self.data_connector.get_polygon_aggregates(
                symbol,
                self.start_date,
                self.end_date
            )
            
            # Alpha signals
            alpha_df = await self.data_connector.generate_regime_adjusted_signals(
                symbol,
                self.start_date,
                self.end_date
            )
            
            # Beta exposure
            beta_df = await self.data_connector.calculate_beta_exposure(
                symbol,
                self.start_date,
                self.end_date
            )
            
            return price_df, alpha_df, beta_df
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def initialize_risk_manager(self):
        """리스크 매니저 초기화"""
        self.risk_manager = AdaptiveAsymmetricRiskManager(
            base_leverage=1.2,
            max_leverage=1.8,
            target_vol=0.18,
            circuit_breaker_dd=-0.06,
            progressive_dd_scaling=True
        )
    
    def calculate_portfolio_weights(
        self,
        current_date: pd.Timestamp,
        current_regime: str
    ) -> Dict[str, float]:
        """
        포트폴리오 가중치 계산
        
        ARES7 QM Regime 전략:
        1. Quality Alpha + Momentum Alpha (60/40)
        2. Regime adjustment
        3. Rank-based weighting
        4. Long-only
        """
        # Get alpha signals for current date
        alpha_scores = {}
        
        for symbol in self.universe:
            if symbol in self.alpha_signals:
                alpha_df = self.alpha_signals[symbol]
                
                # Get regime-adjusted alpha
                if current_date in alpha_df.index:
                    alpha_scores[symbol] = alpha_df.loc[current_date, "regime_adjusted_alpha"]
        
        if not alpha_scores:
            return {}
        
        # Convert to Series and rank
        alpha_series = pd.Series(alpha_scores)
        alpha_series = alpha_series.dropna()
        
        # Rank-based weighting (top 20%)
        n_top = max(1, int(len(alpha_series) * 0.2))
        top_symbols = alpha_series.nlargest(n_top)
        
        # Equal weight for top symbols
        weights = {}
        weight_per_symbol = 1.0 / len(top_symbols)
        
        for symbol in top_symbols.index:
            weights[symbol] = weight_per_symbol
        
        return weights
    
    def apply_risk_management(
        self,
        weights: Dict[str, float],
        current_portfolio_value: float,
        current_drawdown: float
    ) -> Dict[str, float]:
        """
        리스크 관리 적용 (Turbo AARM)
        
        Args:
            weights: 목표 가중치
            current_portfolio_value: 현재 포트폴리오 가치
            current_drawdown: 현재 드로다운
        
        Returns:
            조정된 가중치
        """
        if self.risk_manager is None:
            return weights
        
        # Calculate leverage based on drawdown
        leverage = self.risk_manager.calculate_leverage(current_drawdown)
        
        # Apply leverage to weights
        adjusted_weights = {
            symbol: weight * leverage
            for symbol, weight in weights.items()
        }
        
        return adjusted_weights
    
    def run_backtest(self) -> pd.DataFrame:
        """백테스트 실행"""
        logger.info("Starting backtest...")
        
        # Get all trading dates
        all_dates = sorted(set.union(*[set(df.index) for df in self.price_data.values()]))
        
        # Initialize
        current_capital = self.initial_capital
        current_positions = {}
        portfolio_value = self.initial_capital
        peak_value = self.initial_capital
        
        results = []
        
        for date in all_dates:
            # Get current regime
            if date in self.regime_data.index:
                current_regime = self.regime_data.loc[date]
            else:
                current_regime = "NEUTRAL"
            
            # Calculate current portfolio value
            portfolio_value = current_capital
            for symbol, shares in current_positions.items():
                if symbol in self.price_data and date in self.price_data[symbol].index:
                    price = self.price_data[symbol].loc[date, "close"]
                    portfolio_value += shares * price
            
            # Update peak and calculate drawdown
            peak_value = max(peak_value, portfolio_value)
            current_drawdown = (portfolio_value - peak_value) / peak_value
            
            # Calculate target weights
            target_weights = self.calculate_portfolio_weights(date, current_regime)
            
            # Apply risk management
            adjusted_weights = self.apply_risk_management(
                target_weights,
                portfolio_value,
                current_drawdown
            )
            
            # Rebalance portfolio
            target_positions = {}
            for symbol, weight in adjusted_weights.items():
                if symbol in self.price_data and date in self.price_data[symbol].index:
                    price = self.price_data[symbol].loc[date, "close"]
                    target_value = portfolio_value * weight
                    target_shares = target_value / price
                    target_positions[symbol] = target_shares
            
            # Execute trades
            for symbol in set(list(current_positions.keys()) + list(target_positions.keys())):
                current_shares = current_positions.get(symbol, 0)
                target_shares = target_positions.get(symbol, 0)
                
                if abs(target_shares - current_shares) > 0.01:  # Threshold to avoid tiny trades
                    trade_shares = target_shares - current_shares
                    
                    if symbol in self.price_data and date in self.price_data[symbol].index:
                        price = self.price_data[symbol].loc[date, "close"]
                        trade_value = trade_shares * price
                        
                        # Update capital and positions
                        current_capital -= trade_value
                        current_positions[symbol] = target_shares
                        
                        # Record trade
                        self.trades.append({
                            "date": date,
                            "symbol": symbol,
                            "shares": trade_shares,
                            "price": price,
                            "value": trade_value
                        })
            
            # Record daily results
            results.append({
                "date": date,
                "portfolio_value": portfolio_value,
                "cash": current_capital,
                "regime": current_regime,
                "drawdown": current_drawdown,
                "n_positions": len([s for s in current_positions.values() if abs(s) > 0.01])
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.set_index("date")
        
        logger.info("✅ Backtest completed")
        
        return results_df
    
    def calculate_performance_metrics(self, results_df: pd.DataFrame) -> Dict:
        """성능 지표 계산"""
        # Returns
        returns = results_df["portfolio_value"].pct_change().dropna()
        
        # Sharpe Ratio
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Max Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Total Return
        total_return = (results_df["portfolio_value"].iloc[-1] / results_df["portfolio_value"].iloc[0]) - 1
        
        # Annualized Return
        n_years = len(results_df) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Win Rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        # Calmar Ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        metrics = {
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "win_rate": win_rate,
            "calmar_ratio": calmar_ratio,
            "n_trades": len(self.trades)
        }
        
        return metrics
    
    def calculate_in_sample_out_of_sample(
        self,
        results_df: pd.DataFrame,
        split_date: str
    ) -> Tuple[Dict, Dict]:
        """In-Sample / Out-of-Sample 분리 분석"""
        split_date = pd.Timestamp(split_date)
        
        # Split data
        is_df = results_df[results_df.index < split_date]
        oos_df = results_df[results_df.index >= split_date]
        
        # Calculate metrics for each period
        is_metrics = self.calculate_performance_metrics(is_df)
        oos_metrics = self.calculate_performance_metrics(oos_df)
        
        return is_metrics, oos_metrics


async def main():
    """메인 실행 함수"""
    # S&P 100 유니버스 (예시)
    universe = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH", "JNJ",
        "V", "XOM", "WMT", "JPM", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
        "KO", "PEP", "COST", "AVGO", "ADBE", "CRM", "ACN", "MCD", "CSCO", "TMO",
        "ABT", "LIN", "DHR", "NKE", "NEE", "VZ", "TXN", "WFC", "CMCSA", "DIS",
        "PM", "UPS", "RTX", "QCOM", "ORCL", "INTC", "AMGN", "HON", "INTU", "IBM"
    ]
    
    # Backtest period
    start_date = "2020-01-01"
    end_date = "2024-12-31"
    split_date = "2023-01-01"  # In-Sample / Out-of-Sample split
    
    # Initialize backtest
    backtest = ARES7IntegratedBacktest(
        universe=universe[:20],  # Start with top 20 for testing
        start_date=start_date,
        end_date=end_date,
        initial_capital=1_000_000.0
    )
    
    # Load data
    await backtest.load_data()
    
    # Initialize risk manager
    backtest.initialize_risk_manager()
    
    # Run backtest
    results_df = backtest.run_backtest()
    
    # Calculate performance metrics
    full_metrics = backtest.calculate_performance_metrics(results_df)
    is_metrics, oos_metrics = backtest.calculate_in_sample_out_of_sample(results_df, split_date)
    
    # Print results
    print("\n" + "="*80)
    print("ARES7 QM REGIME BACKTEST RESULTS")
    print("="*80)
    print()
    
    print("FULL PERIOD:")
    print(f"  Sharpe Ratio:        {full_metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:        {full_metrics['max_drawdown']:.2%}")
    print(f"  Total Return:        {full_metrics['total_return']:.2%}")
    print(f"  Annualized Return:   {full_metrics['annualized_return']:.2%}")
    print(f"  Win Rate:            {full_metrics['win_rate']:.2%}")
    print(f"  Calmar Ratio:        {full_metrics['calmar_ratio']:.2f}")
    print(f"  Number of Trades:    {full_metrics['n_trades']}")
    print()
    
    print("IN-SAMPLE (< 2023-01-01):")
    print(f"  Sharpe Ratio:        {is_metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:        {is_metrics['max_drawdown']:.2%}")
    print(f"  Total Return:        {is_metrics['total_return']:.2%}")
    print()
    
    print("OUT-OF-SAMPLE (>= 2023-01-01):")
    print(f"  Sharpe Ratio:        {oos_metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:        {oos_metrics['max_drawdown']:.2%}")
    print(f"  Total Return:        {oos_metrics['total_return']:.2%}")
    print()
    
    print("="*80)
    print()
    
    # Save results
    results_df.to_csv("/home/ubuntu/ARES-Ultimate-251129/results/ares7_backtest_results.csv")
    print("✅ Results saved to results/ares7_backtest_results.csv")


if __name__ == "__main__":
    asyncio.run(main())
