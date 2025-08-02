# core/analysis/technical.py

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD, ADXIndicator, CCIIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import logging
from typing import Dict, Optional, Union
import plotly.graph_objs as go
from datetime import datetime


class TechnicalAnalyzer:
    def __init__(self, config):
        """初始化技术分析器"""
        self.config = config
        self.logger = logging.getLogger(__name__)

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据的有效性"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        if data.empty:
            raise ValueError("Empty data provided")
        if data['close'].isnull().any():
            raise ValueError("Data contains missing values in 'close' column")
        return True

    def _calculate_ma(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculates moving averages robustly."""
        ma_results = {}
        for period in [5, 10, 20, 30, 60]:
            # The `ta` library handles `min_periods` internally, but we ensure fillna=True
            ma = SMAIndicator(close=data['close'], window=period, fillna=True)
            ma_results[f'ma_{period}'] = ma.sma_indicator()
        return ma_results

    def _calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """Calculates RSI robustly."""
        rsi = RSIIndicator(close=data['close'], window=14, fillna=True)
        return rsi.rsi()

    def _calculate_macd(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculates MACD robustly."""
        macd_ind = MACD(close=data['close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
        return {
            'macd': macd_ind.macd(),
            'macd_signal': macd_ind.macd_signal(),
            'macd_diff': macd_ind.macd_diff()
        }

    def _calculate_bollinger(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculates Bollinger Bands robustly."""
        bb = BollingerBands(close=data['close'], window=20, window_dev=2, fillna=True)
        return {
            'bb_upper': bb.bollinger_hband(),
            'bb_lower': bb.bollinger_lband(),
            'bb_middle': bb.bollinger_mavg()
        }


    def _calculate_ma_trend(self, data: pd.DataFrame) -> str:
        """计算移动平均线趋势（严格使用字典访问）"""
        try:
            ma_data = self._calculate_ma(data)
            ma_20 = ma_data.get('ma_20', pd.Series(dtype=float))

            if ma_20.empty or len(ma_20) < 2:
                return "数据不足"

            current = ma_20.iloc[-1]
            previous = ma_20.iloc[-2]

            if pd.isna(current) or pd.isna(previous):
                return "数据不足"

            if current > previous * 1.01:  # 1%上涨阈值
                return "上升"
            elif current < previous * 0.99:  # 1%下跌阈值
                return "下降"
            else:
                return "横盘"
        except Exception as e:
            self.logger.error(f"MA趋势分析错误: {str(e)}")
            return "未知"

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        [CLEAN VERSION] Executes a full technical analysis by orchestrating
        robust, low-level calculation methods.
        """
        try:
            self._validate_data(data)
            result_df = data.copy()

            # --- Combine results from robust helper methods ---
            all_indicators = {}
            all_indicators.update(self._calculate_ma(result_df))
            all_indicators.update(self._calculate_macd(result_df))
            all_indicators.update(self._calculate_bollinger(result_df))

            # Add single-series indicators
            all_indicators['rsi'] = self._calculate_rsi(result_df)

            # Use `ta` library for other indicators, ensuring fillna=True
            adx_ind = ADXIndicator(high=result_df['high'], low=result_df['low'], close=result_df['close'], window=14,
                                   fillna=True)
            all_indicators['adx'] = adx_ind.adx()

            atr_ind = AverageTrueRange(high=result_df['high'], low=result_df['low'], close=result_df['close'],
                                       window=14, fillna=True)
            all_indicators['atr'] = atr_ind.average_true_range()

            cci_ind = CCIIndicator(high=result_df['high'], low=result_df['low'], close=result_df['close'], window=20,
                                   fillna=True)
            all_indicators['cci'] = cci_ind.cci()

            # Assign all calculated indicators to the DataFrame
            for col_name, series in all_indicators.items():
                result_df[col_name] = series

            return result_df

        except Exception as e:
            self.logger.error(f"Technical analysis failed: {str(e)}", exc_info=True)
            return pd.DataFrame()

    def generate_signals(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """生成交易信号（完全兼容原有逻辑）"""
        try:
            self._validate_data(data)
            analysis = self.analyze(data)
            signals = pd.DataFrame(index=data.index)

            # RSI信号
            signals['rsi_signal'] = np.select(
                [analysis['rsi'] < 30, analysis['rsi'] > 70],
                [1, -1],
                default=0
            )

            # MACD信号
            signals['macd_signal'] = np.select(
                [analysis['macd'] > analysis['macd_signal'],
                 analysis['macd'] < analysis['macd_signal']],
                [1, -1],
                default=0
            )

            # 布林带信号
            signals['bb_signal'] = np.select(
                [data['close'] < analysis['bb_lower'],
                 data['close'] > analysis['bb_upper']],
                [1, -1],
                default=0
            )

            # MA信号（显式类型检查）
            ma_20 = analysis.get('ma_20', pd.Series())
            if isinstance(ma_20, dict):  # 双重保险
                ma_20 = ma_20.get('ma_20', pd.Series())

            signals['ma_signal'] = np.select(
                [data['close'] > ma_20,
                 data['close'] < ma_20],
                [1, -1],
                default=0
            )

            # 综合信号
            weights = {
                'rsi_signal': 0.25,
                'macd_signal': 0.25,
                'bb_signal': 0.25,
                'ma_signal': 0.25
            }
            signals['combined_signal'] = sum(
                signals[col] * weight
                for col, weight in weights.items()
            )

            return signals

        except Exception as e:
            self.logger.error(f"信号生成失败: {str(e)}")
            return None

    def plot_indicators(self, data: pd.DataFrame) -> go.Figure:
        """绘制技术指标图表（完全兼容）"""
        try:
            analysis = self.analyze(data)
            fig = go.Figure()

            # K线图
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='价格'
            ))

            # 移动平均线（动态遍历字典）
            colors = ['blue', 'green', 'red', 'purple', 'orange']
            periods = [5, 10, 20, 30, 60]
            for period, color in zip(periods, colors):
                ma_key = f'ma_{period}'
                if ma_key in analysis.get('ma', {}):
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=analysis['ma'][ma_key],
                        name=f'MA{period}',
                        line=dict(color=color, width=1)
                    ))

            # 布林带
            if all(k in analysis for k in ['bb_upper', 'bb_lower', 'bb_middle']):
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=analysis['bb_upper'],
                    name='布林上轨',
                    line=dict(color='gray', dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=analysis['bb_lower'],
                    name='布林下轨',
                    line=dict(color='gray', dash='dash'),
                    fill='tonexty'
                ))

            # 图表布局
            fig.update_layout(
                title='技术分析图表',
                yaxis_title='价格',
                xaxis_title='日期',
                template='plotly_white',
                showlegend=True,
                height=600
            )
            return fig

        except Exception as e:
            self.logger.error(f"图表绘制失败: {str(e)}")
            raise