# core/backtesting/engine.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
import logging
from datetime import datetime
from ..analysis.performance import PerformanceAnalyzer


class BacktestEngine:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.performance_analyzer = PerformanceAnalyzer(config)

    def run_backtest(
            self,
            strategy: Callable,
            data: pd.DataFrame,
            initial_capital: float,
            benchmark_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """运行回测"""
        try:
            # 初始化回测状态
            self.portfolio = self._initialize_portfolio(initial_capital)
            self.trades = []
            self.portfolio_history = []

            # 运行回测
            for timestamp, row in data.iterrows():
                # 更新市场数据
                market_data = self._prepare_market_data(row)

                # 运行策略
                signals = strategy(market_data, self.portfolio)

                # 执行交易
                self._execute_signals(signals, market_data, timestamp)

                # 更新组合状态
                self._update_portfolio(market_data, timestamp)

                # 记录历史
                self._record_history(timestamp)

            # 生成回测报告
            report = self._generate_backtest_report(benchmark_data)

            return report

        except Exception as e:
            self.logger.error(f"Backtest error: {e}")
            return {}

    def _initialize_portfolio(self, initial_capital: float) -> Dict:
        """初始化投资组合"""
        return {
            'cash': initial_capital,
            'positions': {},
            'total_value': initial_capital
        }

    def _prepare_market_data(self, row: pd.Series) -> Dict:
        """准备市场数据"""
        return {
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume'],
            'additional_data': row.get('additional_data', {})
        }

    def _execute_signals(
            self,
            signals: Dict,
            market_data: Dict,
            timestamp: datetime
    ) -> None:
        """执行交易信号"""
        for symbol, signal in signals.items():
            if signal == 0:  # 不交易
                continue

            # 计算交易量
            quantity = self._calculate_position_size(
                signal, market_data['close'], symbol)

            if quantity == 0:
                continue

            # 执行交易
            self._execute_trade(symbol, quantity, market_data['close'], timestamp)

    def _calculate_position_size(
            self,
            signal: float,
            price: float,
            symbol: str
    ) -> int:
        """计算持仓规模"""
        try:
            # 计算目标持仓价值
            portfolio_value = self.portfolio['total_value']
            target_value = portfolio_value * self.config.POSITION_SIZE * abs(signal)

            # 计算股数
            quantity = int(target_value / price)

            # 检查是否超过最大持仓限制
            max_quantity = int(portfolio_value * self.config.MAX_POSITION_SIZE / price)
            quantity = min(quantity, max_quantity)

            return quantity

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0

    def _execute_trade(
            self,
            symbol: str,
            quantity: int,
            price: float,
            timestamp: datetime
    ) -> None:
        """执行交易"""
        try:
            # 计算交易成本
            transaction_cost = self._calculate_transaction_cost(quantity, price)
            total_cost = quantity * price + transaction_cost

            # 检查是否有足够资金
            if total_cost > self.portfolio['cash'] and quantity > 0:
                return

            # 更新现金和持仓
            self.portfolio['cash'] -= total_cost

            if symbol not in self.portfolio['positions']:
                self.portfolio['positions'][symbol] = 0
            self.portfolio['positions'][symbol] += quantity

            # 记录交易
            self.trades.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'cost': transaction_cost
            })

        except Exception as e:
            self.logger.error(f"Trade execution error: {e}")

    def _calculate_transaction_cost(self, quantity: int, price: float) -> float:
        """计算交易成本"""
        # 计算佣金
        commission = max(
            quantity * price * self.config.COMMISSION_RATE,
            self.config.MIN_COMMISSION
        )

        # 计算滑点成本
        slippage = quantity * price * self.config.SLIPPAGE_RATE

        return commission + slippage

    def _update_portfolio(self, market_data: Dict, timestamp: datetime) -> None:
        """更新组合状态"""
        try:
            total_value = self.portfolio['cash']

            # 更新持仓市值
            for symbol, quantity in self.portfolio['positions'].items():
                position_value = quantity * market_data['close']
                total_value += position_value

            self.portfolio['total_value'] = total_value

        except Exception as e:
            self.logger.error(f"Portfolio update error: {e}")

    def _record_history(self, timestamp: datetime) -> None:
        """记录历史数据"""
        self.portfolio_history.append({
            'timestamp': timestamp,
            'total_value': self.portfolio['total_value'],
            'cash': self.portfolio['cash'],
            'positions': self.portfolio['positions'].copy()
        })

    def _generate_backtest_report(
            self,
            benchmark_data: Optional[pd.DataFrame]
    ) -> Dict:
        """生成回测报告"""
        try:
            # 计算基本指标
            performance_metrics = self.performance_analyzer.analyze_portfolio_performance(
                self.portfolio_history,
                benchmark_data
            )

            # 添加交易统计
            trade_statistics = self._calculate_trade_statistics()

            # 生成完整报告
            report = {
                'performance_metrics': performance_metrics,
                'trade_statistics': trade_statistics,
                'portfolio_history': self.portfolio_history,
                'trades': self.trades
            }

            return report

        except Exception as e:
            self.logger.error(f"Report generation error: {e}")
            return {}

    def _calculate_trade_statistics(self) -> Dict:
        """计算交易统计数据"""
        try:
            if not self.trades:
                return {}

            trades_df = pd.DataFrame(self.trades)

            # 计算基本统计
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['quantity'] > 0])
            losing_trades = total_trades - winning_trades

            # 计算盈亏统计
            total_profit = trades_df['quantity'].mul(trades_df['price']).sum()
            total_cost = trades_df['cost'].sum()
            net_profit = total_profit - total_cost

            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'total_profit': total_profit,
                'total_cost': total_cost,
                'net_profit': net_profit
            }

        except Exception as e:
            self.logger.error(f"Trade statistics calculation error: {e}")
            return {}