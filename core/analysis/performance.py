# core/analysis/performance.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta


class PerformanceAnalyzer:
    def __init__(self, config):
        """初始化性能分析器"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics = {}

    def calculate_returns(self, portfolio_values: pd.Series) -> pd.Series:
        """计算收益率"""
        try:
            returns = portfolio_values.pct_change()
            returns.fillna(0, inplace=True)
            return returns
        except Exception as e:
            self.logger.error(f"Error calculating returns: {e}")
            return pd.Series()

    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """计算夏普比率"""
        try:
            excess_returns = returns - self.config.RISK_FREE_RATE / 252  # 转换为日化收益率
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            return sharpe_ratio
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """计算最大回撤"""
        try:
            rolling_max = portfolio_values.expanding().max()
            drawdowns = portfolio_values / rolling_max - 1
            max_drawdown = drawdowns.min()
            return max_drawdown
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}")
            return 0.0

    def calculate_metrics(self, portfolio_values: pd.Series) -> Dict:
        """计算所有性能指标"""
        try:
            returns = self.calculate_returns(portfolio_values)

            self.metrics = {
                'total_return': (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1),
                'annualized_return': (1 + returns.mean()) ** 252 - 1,
                'sharpe_ratio': self.calculate_sharpe_ratio(returns),
                'max_drawdown': self.calculate_max_drawdown(portfolio_values),
                'volatility': returns.std() * np.sqrt(252),
                'win_rate': (returns > 0).mean(),
                'avg_return': returns.mean(),
                'avg_win': returns[returns > 0].mean(),
                'avg_loss': returns[returns < 0].mean(),
            }

            return self.metrics
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}

    def generate_report(self, portfolio_values: pd.Series, trades: List[Dict]) -> Dict:
        """生成完整的性能报告"""
        try:
            metrics = self.calculate_metrics(portfolio_values)

            report = {
                'performance_metrics': metrics,
                'trading_summary': {
                    'total_trades': len(trades),
                    'profitable_trades': len([t for t in trades if t.get('profit', 0) > 0]),
                    'loss_trades': len([t for t in trades if t.get('profit', 0) <= 0]),
                    'total_profit': sum(t.get('profit', 0) for t in trades),
                    'total_commission': sum(t.get('commission', 0) for t in trades),
                },
                'timestamps': {
                    'start_date': portfolio_values.index[0],
                    'end_date': portfolio_values.index[-1],
                    'duration_days': (portfolio_values.index[-1] - portfolio_values.index[0]).days
                }
            }

            return report
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return {}

    def plot_equity_curve(self, portfolio_values: pd.Series) -> Optional[Dict]:
        """生成权益曲线图表数据"""
        try:
            equity_curve = {
                'dates': portfolio_values.index.strftime('%Y-%m-%d').tolist(),
                'values': portfolio_values.values.tolist(),
                'drawdown': self.calculate_drawdown_series(portfolio_values).values.tolist()
            }
            return equity_curve
        except Exception as e:
            self.logger.error(f"Error generating equity curve plot: {e}")
            return None

    def calculate_drawdown_series(self, portfolio_values: pd.Series) -> pd.Series:
        """计算回撤序列"""
        try:
            rolling_max = portfolio_values.expanding().max()
            drawdowns = portfolio_values / rolling_max - 1
            return drawdowns
        except Exception as e:
            self.logger.error(f"Error calculating drawdown series: {e}")
            return pd.Series()

    def _calculate_risk_metrics(
            self,
            returns: pd.Series,
            benchmark_returns: pd.Series
    ) -> Dict:
        """计算风险指标"""
        try:
            # 计算波动率
            volatility = returns.std() * np.sqrt(252)

            # 计算最大回撤
            cum_returns = (1 + returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdowns = (cum_returns - running_max) / running_max
            max_drawdown = drawdowns.min()

            # 计算夏普比率
            excess_returns = returns - self.config.RISK_FREE_RATE / 252
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std()

            # 计算Sortino比率
            downside_returns = returns[returns < 0]
            sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std()

            # 计算信息比率
            tracking_error = (returns - benchmark_returns).std() * np.sqrt(252)
            information_ratio = (returns - benchmark_returns).mean() * np.sqrt(252) / tracking_error

            return {
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'information_ratio': information_ratio
            }

        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {}

    def _calculate_performance_metrics(
            self,
            returns: pd.Series,
            benchmark_returns: pd.Series
    ) -> Dict:
        """计算绩效指标"""
        try:
            # 计算累积收益
            cum_returns = (1 + returns).cumprod() - 1
            benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1

            # 计算年化收益
            total_days = (returns.index[-1] - returns.index[0]).days
            annual_return = (1 + cum_returns.iloc[-1]) ** (365 / total_days) - 1

            # 计算胜率
            winning_days = (returns > 0).sum()
            total_days = len(returns)
            win_rate = winning_days / total_days

            # 计算盈亏比
            avg_gain = returns[returns > 0].mean()
            avg_loss = abs(returns[returns < 0].mean())
            profit_loss_ratio = avg_gain / avg_loss if avg_loss != 0 else np.inf

            # 计算Alpha和Beta
            covariance = returns.cov(benchmark_returns)
            beta = covariance / benchmark_returns.var()
            alpha = (annual_return - self.config.RISK_FREE_RATE) - (
                    beta * (benchmark_returns.mean() * 252 - self.config.RISK_FREE_RATE))

            return {
                'total_return': cum_returns.iloc[-1],
                'annual_return': annual_return,
                'win_rate': win_rate,
                'profit_loss_ratio': profit_loss_ratio,
                'alpha': alpha,
                'beta': beta
            }

        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}

    def generate_performance_report(
            self,
            portfolio_history: List[Dict],
            benchmark_data: pd.DataFrame
    ) -> Dict:
        """生成综合绩效报告"""
        try:
            # 基本分析
            analysis = self.analyze_portfolio_performance(
                portfolio_history, benchmark_data)

            # 添加时间序列分析
            time_series_analysis = self._analyze_time_series(
                portfolio_history, benchmark_data)

            # 添加归因分析
            attribution_analysis = self._perform_attribution_analysis(
                portfolio_history, benchmark_data)

            # 生成综合报告
            report = {
                **analysis,
                'time_series_analysis': time_series_analysis,
                'attribution_analysis': attribution_analysis,
                'report_generated': datetime.now().isoformat()
            }

            return report

        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return {}