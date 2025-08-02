# core/strategy/unified_strategy.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback
from typing import Optional, Dict, Any, List
from core.translate import translator
import threading
import time

# --- 统一的依赖导入 ---
try:
    from .ml_strategy import MLStrategy, SKLEARN_AVAILABLE, TENSORFLOW_AVAILABLE
except ImportError as e:
    MLStrategy, SKLEARN_AVAILABLE, TENSORFLOW_AVAILABLE = None, False, False
    logging.warning(f"MLStrategy 无法导入 ({e})，机器学习功能将不可用。")

logger = logging.getLogger(__name__)


class UnifiedStrategy:
    """
    统一策略模块，作为策略展示、回测和信号融合的协调中心。
    """

    def __init__(self, system: Any):
        """
        [最终修复版] 初始化统一策略模块。
        """
        self.system = system
        self.backtest_results: Dict[str, Any] = {}

        # --- 初始化 MLStrategy 实例 ---
        # 1. 先将实例属性初始化为 None
        self.ml_strategy_instance: Optional[MLStrategy] = None

        # 2. 检查所有前置条件是否满足
        # (ML库可用，system对象有效，且system对象已成功初始化config和data_manager)
        can_init_ml = (SKLEARN_AVAILABLE or TENSORFLOW_AVAILABLE) and \
                      MLStrategy is not None and \
                      hasattr(self.system, 'config') and self.system.config is not None and \
                      hasattr(self.system, 'data_manager') and self.system.data_manager is not None

        if can_init_ml:
            # 3. 只有在所有条件都满足时，才尝试创建实例
            try:
                # 将 system.data_manager 明确地传递给 MLStrategy
                self.ml_strategy_instance = MLStrategy(
                    config=self.system.config,
                    data_manager_ref=self.system.data_manager
                )
                logger.info("MLStrategy instance (with its TextFeatureExtractor) was successfully created.")
            except Exception as e:
                logger.error(f"Failed to create MLStrategy instance: {e}", exc_info=True)
                # 创建失败，保持 self.ml_strategy_instance 为 None
                self.ml_strategy_instance = None
        else:
            # 4. 如果前置条件不满足，记录清晰的警告
            logger.warning("ML features are disabled because one or more prerequisites are not met:")
            if not (SKLEARN_AVAILABLE or TENSORFLOW_AVAILABLE):
                logger.warning("- ML libraries (Sklearn/TensorFlow) are not available.")
            if not hasattr(self.system, 'config') or self.system.config is None:
                logger.warning("- System.config is not available.")
            if not hasattr(self.system, 'data_manager') or self.system.data_manager is None:
                logger.warning("- System.data_manager is not available.")

    # 辅助属性，方便地访问 text_feature_extractor
    @property
    def text_feature_extractor(self):
        if self.ml_strategy_instance:
            return self.ml_strategy_instance.text_feature_extractor
        return None

    def _prepare_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        准备用于技术分析和简单回测的特征数据。
        注意：此方法与 MLStrategy 中的 prepare_features 不同，后者更复杂且用于ML。
        """
        if data is None or data.empty or 'close' not in data.columns:
            logger.warning("数据为空或缺少'close'列，无法准备特征。")
            return None
        if len(data) < 20:  # 需要足够数据计算MA20
            logger.warning(f"数据行数 ({len(data)}) 过少，无法准备所有技术指标。")
            # return None # 或者只计算可计算的指标

        try:
            df = data.copy()

            # 计算移动平均线
            df['MA5'] = df['close'].rolling(window=5, min_periods=1).mean()
            df['MA10'] = df['close'].rolling(window=10, min_periods=1).mean()
            df['MA20'] = df['close'].rolling(window=20, min_periods=1).mean()

            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0.0).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(window=14, min_periods=1).mean()

            # 避免除以零
            rs = gain / loss.replace(0, 1e-9)  # 用一个极小的数替换0
            df['RSI'] = 100.0 - (100.0 / (1.0 + rs))
            df['RSI'] = df['RSI'].fillna(50)  # RSI无法计算时（例如loss全为0），设为中性50

            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()  # 通常称为Signal Line

            # 布林带
            df['BB_Mid'] = df['close'].rolling(window=20, min_periods=1).mean()
            std_20 = df['close'].rolling(window=20, min_periods=1).std().fillna(0)  # std为NaN时填充0
            df['BB_Upper'] = df['BB_Mid'] + (std_20 * 2)
            df['BB_Lower'] = df['BB_Mid'] - (std_20 * 2)

            # 成交量相关 (如果volume列存在)
            if 'volume' in df.columns:
                df['Volume_MA20'] = df['volume'].rolling(window=20, min_periods=1).mean()

            # df = df.dropna() # 移除此行，让调用者处理NaN，或者在计算指标时用min_periods
            return df
        except Exception as e:
            logger.error(f"为技术分析准备特征数据时出错: {e}", exc_info=True)
            return data  # 返回原始数据，让调用者处理

    def get_technical_signals(self, data: pd.DataFrame, strategy_type: str = "ma_crossover") -> Dict:
        """获取技术分析信号"""

        processed_data = self._prepare_features(data)

        if processed_data is None or processed_data.empty:
            return {"signal": "ERROR", "reason": "无法处理输入数据以生成技术信号。", "indicators": {}, "data": data}

        # 使用处理后的数据
        df = processed_data.copy()  # 创建副本以防修改原始处理数据

        # 确保最新的数据行有计算好的指标值
        if df.empty or any(pd.isna(df.iloc[-1].get(col)) for col in
                           ['MA5', 'MA20', 'RSI', 'MACD', 'Signal_Line', 'BB_Upper', 'BB_Lower']):
            # 如果关键指标在最后一行是NaN，可能意味着数据不足或处理问题
            return {"signal": "HOLD", "reason": "最新的技术指标数据不足或无效。",
                    "indicators": df.iloc[-1].to_dict() if not df.empty else {}, "data": df}

        signal = "HOLD"
        reason = "无明确信号"
        latest_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) >= 2 else latest_row  # 安全获取前一行

        try:
            if strategy_type == "ma_crossover":
                if prev_row['MA5'] < prev_row['MA20'] and latest_row['MA5'] > latest_row['MA20']:
                    signal = "BUY"
                    reason = "MA5上穿MA20 (金叉)"
                elif prev_row['MA5'] > prev_row['MA20'] and latest_row['MA5'] < latest_row['MA20']:
                    signal = "SELL"
                    reason = "MA5下穿MA20 (死叉)"

            elif strategy_type == "rsi":
                if latest_row['RSI'] < 30:
                    signal = "BUY"
                    reason = f"RSI ({latest_row['RSI']:.2f}) 进入超卖区 (<30)"
                elif latest_row['RSI'] > 70:
                    signal = "SELL"
                    reason = f"RSI ({latest_row['RSI']:.2f}) 进入超买区 (>70)"

            elif strategy_type == "macd":
                if prev_row['MACD'] < prev_row['Signal_Line'] and latest_row['MACD'] > latest_row['Signal_Line']:
                    signal = "BUY"
                    reason = "MACD线上穿信号线 (金叉)"
                elif prev_row['MACD'] > prev_row['Signal_Line'] and latest_row['MACD'] < latest_row['Signal_Line']:
                    signal = "SELL"
                    reason = "MACD线下穿信号线 (死叉)"

            elif strategy_type == "bollinger":
                if latest_row['close'] < latest_row['BB_Lower']:
                    signal = "BUY"
                    reason = "价格触及布林带下轨"
                elif latest_row['close'] > latest_row['BB_Upper']:
                    signal = "SELL"
                    reason = "价格触及布林带上轨"

            indicators = latest_row.to_dict()  # 获取最后一行的所有指标值

            return {
                "signal": signal,
                "reason": reason,
                "indicators": indicators,
                "data": df  # 返回包含所有计算指标的DataFrame
            }
        except KeyError as e:
            error_msg = f"计算技术信号时出错: 缺少列 {e}。请确保数据已正确处理。"
            logger.error(error_msg, exc_info=True)
            return {"signal": "ERROR", "reason": error_msg, "indicators": {}, "data": df}
        except Exception as e:
            error_msg = f"计算技术信号时发生未知错误: {e}"
            logger.error(error_msg, exc_info=True)
            return {"signal": "ERROR", "reason": error_msg, "indicators": {}, "data": df}

    def _backtest_technical_strategy(self,
                                     symbol: str,
                                     data: pd.DataFrame,
                                     strategy_type: str,
                                     initial_capital: float,
                                     commission_rate: float) -> Dict:
        """
        [最终修复版] 回测传统技术指标策略，并调用统一的指标计算方法。
        """
        logger.info(f"--- Starting Technical Backtest for {symbol} ({strategy_type}) ---")

        tech_signal_result = self.get_technical_signals(data.copy(), strategy_type)
        if tech_signal_result.get("signal") == "ERROR":
            return {"success": False, "message": tech_signal_result.get('reason')}

        backtest_df = tech_signal_result["data"].copy()

        # --- 生成 Signal_Action (逻辑不变) ---
        # ...

        # --- 交易循环 (逻辑不变) ---
        portfolio = {'cash': initial_capital, 'shares': 0.0}
        history, trades_log = [], []
        current_position = 0

        for i in range(len(backtest_df)):
            current_price = backtest_df['close'].iloc[i]
            signal_action = backtest_df['Signal_Action'].iloc[i]
            if pd.isna(current_price): continue

            if signal_action == "BUY" and current_position == 0:
                shares_to_buy = np.floor(portfolio['cash'] / (current_price * (1 + commission_rate)))
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + commission_rate)
                    portfolio['cash'] -= cost;
                    portfolio['shares'] += shares_to_buy;
                    current_position = 1
                    trades_log.append(
                        {'date': backtest_df.index[i], 'type': 'BUY', 'price': current_price, 'shares': shares_to_buy})
            elif signal_action == "SELL" and current_position == 1:
                shares_to_sell = portfolio['shares']
                proceeds = shares_to_sell * current_price * (1 - commission_rate)
                portfolio['cash'] += proceeds;
                portfolio['shares'] = 0;
                current_position = 0
                trades_log.append(
                    {'date': backtest_df.index[i], 'type': 'SELL', 'price': current_price, 'shares': shares_to_sell})

            total_value = portfolio['cash'] + (portfolio['shares'] * current_price)
            history.append({'timestamp': backtest_df.index[i], 'total_value': total_value, 'price': current_price})

        # --- 调用统一的指标计算方法 ---
        return self._calculate_backtest_stats(history, trades_log, initial_capital)

    def _calculate_backtest_stats(self, history: List[Dict], trades_log: List[Dict], initial_capital: float,
                                  alpha_scores: Optional[List[float]] = None) -> Dict:
        """
        [新增/统一] 计算全面的回测性能指标。
        """
        if not history: return {"success": False, "message": "回测历史为空，无法计算指标。"}

        history_df = pd.DataFrame(history).set_index('timestamp')

        # --- 基础指标 ---
        final_value = history_df['total_value'].iloc[-1]
        total_return = (final_value / initial_capital) - 1
        buy_hold_return = (history_df['price'].iloc[-1] / history_df['price'].iloc[0]) - 1

        # --- 风险指标 ---
        daily_returns = history_df['total_value'].pct_change().fillna(0)
        annual_volatility = daily_returns.std() * np.sqrt(252)
        max_drawdown = (history_df['total_value'] / history_df['total_value'].cummax() - 1.0).min()

        # --- 风险调整后收益指标 ---
        risk_free_rate = getattr(self.system.config, 'RISK_FREE_RATE', 0.03)
        # 修复年化收益率计算
        num_days = len(history_df)
        annual_return = (1 + total_return) ** (252.0 / num_days) - 1 if num_days > 0 else 0

        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.inf

        # --- 交易指标 (修复了 UnboundLocalError) ---
        profits, losses = [], []
        if trades_log and len(trades_log) > 0:
            for i in range(len(trades_log)):
                if trades_log[i]['type'] == 'SELL':
                    # 简化逻辑：找到这笔卖出对应的上一笔交易（必须是买入）
                    if i > 0 and trades_log[i - 1]['type'] == 'BUY':
                        # 确保交易的股数一致或部分卖出
                        trade_shares = min(trades_log[i]['shares'], trades_log[i - 1]['shares'])
                        profit = (trades_log[i]['price'] - trades_log[i - 1]['price']) * trade_shares
                        if profit > 0:
                            profits.append(profit)
                        else:
                            losses.append(profit)

        win_rate = len(profits) / (len(profits) + len(losses)) if (len(profits) + len(losses)) > 0 else 0.0
        avg_profit = np.mean(profits) if profits else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else np.inf

        stats = {
            "initial_capital": initial_capital, "final_value": final_value,
            "total_return": total_return, "buy_hold_return": buy_hold_return,
            "annual_return": annual_return, "annual_volatility": annual_volatility,
            "max_drawdown": max_drawdown, "sharpe_ratio": sharpe_ratio,
            "calmar_ratio": calmar_ratio, "trades": len(trades_log),
            "win_rate": win_rate, "profit_loss_ratio": profit_loss_ratio,
        }

        # 将交易记录添加到 history_df 以便绘图
        if trades_log:
            trades_df = pd.DataFrame(trades_log).set_index('date')
            history_df['buy_signal'] = trades_df[trades_df['type'] == 'BUY']['price']
            history_df['sell_signal'] = trades_df[trades_df['type'] == 'SELL']['price']
        else:
            history_df['buy_signal'] = np.nan;
            history_df['sell_signal'] = np.nan

        return {
            "success": True, "message": f"回测成功完成，共产生 {len(trades_log)} 笔交易。",
            "stats": stats, "history_df": history_df,
            "alpha_scores": alpha_scores
        }

    def backtest_strategy(self,
                          symbol: str,
                          data: pd.DataFrame,
                          strategy_config: Dict,
                          initial_capital: float = 10000.0,
                          commission_rate: float = 0.0003) -> Dict:
        """
        [最终修复版] 统一回测引擎，确保数据流畅通。
        """
        strategy_type = strategy_config.get("type")
        logger.info(f"--- 开始统一回测 for {symbol} | Strategy: {strategy_type} ---")

        if data is None or data.empty:
            return {"success": False, "message": "回测数据无效。"}

        try:
            # --- 1. 数据准备和特征计算 ---
            st.info("正在准备回测数据和计算特征...")

            # 确保数据有足够的历史
            if len(data) < 150:  # 需要足够的lookback
                return {"success": False, "message": f"数据不足，仅有{len(data)}行"}

            # 计算所有技术特征
            if hasattr(self.system, 'technical_analyzer'):
                features_df = self.system.technical_analyzer.analyze(data.copy())
            else:
                features_df = self._prepare_features(data.copy())

            if features_df.empty:
                return {"success": False, "message": "特征计算失败"}

            # --- 2. 根据策略类型生成信号 ---
            if strategy_type == "ml_quant":
                return self._backtest_ml_strategy(symbol, features_df, strategy_config,
                                                  initial_capital, commission_rate)
            elif strategy_type == "technical":
                return self._backtest_technical_strategy(symbol, features_df,
                                                         strategy_config.get("strategy_type", "ma_crossover"),
                                                         initial_capital, commission_rate)
            else:
                return {"success": False, "message": f"不支持的策略类型: {strategy_type}"}

        except Exception as e:
            logger.error(f"回测过程中出错: {e}", exc_info=True)
            return {"success": False, "message": f"回测失败: {str(e)}"}

    def _backtest_ml_strategy(self, symbol: str, features_df: pd.DataFrame,
                              strategy_config: Dict, initial_capital: float,
                              commission_rate: float) -> Dict:
        """ML策略的回测逻辑"""

        model_to_use = strategy_config.get('ml_model_name')
        if not model_to_use or not self.ml_strategy_instance:
            return {"success": False, "message": "ML模型配置无效"}

        # 1. 设置活动模型
        if not self.ml_strategy_instance.set_active_model(model_to_use):
            return {"success": False, "message": f"加载模型 '{model_to_use}' 失败"}

        st.info(f"使用模型 '{model_to_use}' 进行批量预测...")

        # 2. 如果启用LLM，准备文本特征
        if strategy_config.get('use_llm', False):
            features_df = self._prepare_text_features_for_backtest(features_df, symbol)

        # 3. 进行批量预测
        predictions = self.ml_strategy_instance.predict_for_backtest(features_df, symbol)

        if predictions is None or predictions.empty:
            return {"success": False, "message": "批量预测失败"}

        logger.info(f"批量预测成功，获得 {len(predictions)} 个预测值")

        # 4. 执行回测循环
        return self._execute_backtest_loop(features_df, predictions, strategy_config,
                                           initial_capital, commission_rate)

    def _prepare_text_features_for_backtest(self, features_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """为回测准备文本特征"""
        if not self.text_feature_extractor or not self.text_feature_extractor.is_available:
            logger.warning("文本特征提取器不可用，跳过文本特征")
            # 初始化默认的文本特征
            for col in self.ml_strategy_instance.text_feature_columns:
                if col not in features_df.columns:
                    features_df[col] = 0.0
            return features_df

        try:
            st.info("正在获取和分析文本特征...")

            # 调用文本特征提取器的批量方法
            text_features = self.text_feature_extractor.get_and_extract_features_for_backtest(
                symbol, features_df.index
            )

            if text_features is not None and not text_features.empty:
                # 合并文本特征
                features_df = features_df.join(text_features, how='left')

                # 填充缺失值
                for col in self.ml_strategy_instance.text_feature_columns:
                    if col in features_df.columns:
                        features_df[col] = features_df[col].fillna(0.0)
                    else:
                        features_df[col] = 0.0

                logger.info("文本特征合并成功")
            else:
                logger.warning("文本特征获取失败，使用默认值")
                for col in self.ml_strategy_instance.text_feature_columns:
                    features_df[col] = 0.0

        except Exception as e:
            logger.error(f"文本特征准备失败: {e}")
            # 使用默认值
            for col in self.ml_strategy_instance.text_feature_columns:
                features_df[col] = 0.0

        return features_df

    def _execute_backtest_loop(self, features_df: pd.DataFrame, predictions: pd.Series,
                               strategy_config: Dict, initial_capital: float,
                               commission_rate: float) -> Dict:
        """执行回测交易循环"""

        # 1. 对齐预测数据和特征数据
        common_index = features_df.index.intersection(predictions.index)
        if len(common_index) == 0:
            return {"success": False, "message": "预测数据与特征数据索引不匹配"}

        backtest_df = features_df.loc[common_index].copy()
        aligned_predictions = predictions.loc[common_index]

        logger.info(f"对齐后的回测数据: {len(backtest_df)} 行")

        # 2. 初始化交易状态
        portfolio = {'cash': initial_capital, 'shares': 0.0}
        history, trades_log = [], []
        alpha_scores = []

        # 3. 执行交易循环
        alpha_threshold = strategy_config.get('alpha_threshold', 0.1)

        for i, (date, row) in enumerate(backtest_df.iterrows()):
            current_price = row['close']
            if pd.isna(current_price):
                continue

            # 获取当前的alpha分数
            if date in aligned_predictions.index:
                current_alpha = aligned_predictions.loc[date]
            else:
                current_alpha = 0.0

            alpha_scores.append(current_alpha)

            # 交易决策
            if current_alpha > alpha_threshold and portfolio['shares'] == 0:
                # 买入
                shares_to_buy = np.floor(portfolio['cash'] / (current_price * (1 + commission_rate)))
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + commission_rate)
                    portfolio['cash'] -= cost
                    portfolio['shares'] += shares_to_buy
                    trades_log.append({
                        'date': date, 'type': 'BUY', 'price': current_price, 'shares': shares_to_buy
                    })

            elif current_alpha < -alpha_threshold and portfolio['shares'] > 0:
                # 卖出
                shares_to_sell = portfolio['shares']
                proceeds = shares_to_sell * current_price * (1 - commission_rate)
                portfolio['cash'] += proceeds
                portfolio['shares'] = 0
                trades_log.append({
                    'date': date, 'type': 'SELL', 'price': current_price, 'shares': shares_to_sell
                })

            # 记录组合价值
            total_value = portfolio['cash'] + (portfolio['shares'] * current_price)
            history.append({
                'timestamp': date, 'total_value': total_value, 'price': current_price
            })

        # 4. 计算最终性能指标
        return self._calculate_backtest_stats(history, trades_log, initial_capital, alpha_scores)

    def _plot_backtest_chart_with_trades(self, history_df: pd.DataFrame, initial_capital: float, title: str):
        """
        [完整修复版] 统一的回测图表绘制函数，包含价格K线、交易点标记、权益曲线和回撤。
        """
        if history_df is None or history_df.empty:
            st.warning(translator.t('warning_no_data_for_backtest_chart', fallback="无回测历史数据可供绘图。"))
            return

        # --- 创建带有两个Y轴的子图 ---
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,  # 减小垂直间距
            row_heights=[0.7, 0.3],
            specs=[[{"secondary_y": True}],  # 第一行有两个Y轴
                   [{"secondary_y": False}]]  # 第二行只有一个Y轴
        )

        # --- 图 1: 价格、交易点和权益曲线 ---

        # 1a. 在主Y轴 (y1) 上绘制价格。使用收盘价线图作为背景。
        fig.add_trace(go.Scatter(
            x=history_df.index,
            y=history_df['price'],
            mode='lines',
            name=translator.t('price', fallback='Price'),
            line=dict(color='lightgrey', width=1.5)
        ), secondary_y=False, row=1, col=1)

        # 1b. 在主Y轴 (y1) 上标记买入信号
        if 'buy_signal' in history_df.columns:
            fig.add_trace(go.Scatter(
                x=history_df.index,
                y=history_df['buy_signal'],  # Y值是交易发生时的价格
                mode='markers',
                name=translator.t('buy', fallback='Buy'),
                marker=dict(symbol='triangle-up', size=10, color='green', line=dict(width=1, color='DarkSlateGrey'))
            ), secondary_y=False, row=1, col=1)

        # 1c. 在主Y轴 (y1) 上标记卖出信号
        if 'sell_signal' in history_df.columns:
            fig.add_trace(go.Scatter(
                x=history_df.index,
                y=history_df['sell_signal'],
                mode='markers',
                name=translator.t('sell', fallback='Sell'),
                marker=dict(symbol='triangle-down', size=10, color='red', line=dict(width=1, color='DarkSlateGrey'))
            ), secondary_y=False, row=1, col=1)

        # 1d. 在次Y轴 (y2) 上绘制策略权益曲线
        fig.add_trace(go.Scatter(
            x=history_df.index,
            y=history_df['total_value'],
            mode='lines',
            name=translator.t('strategy_equity', fallback='Strategy Equity'),
            line=dict(color='blue', width=2)
        ), secondary_y=True, row=1, col=1)

        # --- 图 2: 回撤 ---
        roll_max = history_df['total_value'].cummax()
        drawdown = (history_df['total_value'] / roll_max - 1.0) * 100  # 转换为百分比
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown,
            mode='lines',
            name=translator.t('drawdown', fallback='Drawdown'),
            fill='tozeroy',  # 填充到0线
            line=dict(color='rgba(255, 82, 82, 0.7)')  # 使用半透明红色
        ), row=2, col=1)

        # --- 图表布局和样式 ---
        fig.update_layout(
            title_text=title,
            height=700,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,  # 放在图表顶部
                xanchor="right",
                x=1
            ),
            xaxis_rangeslider_visible=False,  # 隐藏K线图的范围滑块
            margin=dict(t=80, b=50, l=50, r=50)  # 调整边距
        )

        # 设置Y轴标题和样式
        fig.update_yaxes(title_text=translator.t('price_axis_label', fallback="Price ($)"), row=1, col=1,
                         secondary_y=False)
        fig.update_yaxes(title_text=translator.t('equity_axis_label', fallback="Portfolio Value ($)"), row=1, col=1,
                         secondary_y=True, showgrid=False)  # 次Y轴不显示网格线
        fig.update_yaxes(title_text=translator.t('drawdown_axis_label', fallback="Drawdown (%)"), row=2, col=1)

        # 设置X轴标题（只在最下面的子图显示）
        fig.update_xaxes(title_text=translator.t('date', fallback="Date"), row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

    def render_strategy_ui(self, system):  # system 参数现在是 TradingSystem 的实例
        """渲染策略交易UI"""
        st.header(translator.t('strategy_page_main_title', fallback="📊 策略交易"))

        # 确保 system 和 system.config 存在
        if not system or not hasattr(system, 'config'):
            st.error("系统或配置未正确初始化，无法渲染策略UI。")
            return

        tech_analysis_tab_label = translator.t('tech_analysis_tab_title', fallback="技术分析策略")
        ml_strategy_tab_label = translator.t('ml_strategy_tab_title', fallback="机器学习策略")
        backtest_tab_label = translator.t('backtest_tab_title', fallback="策略回测")

        tabs = st.tabs([tech_analysis_tab_label, ml_strategy_tab_label, backtest_tab_label])

        with tabs[0]:
            self._render_technical_analysis_tab(system)

        with tabs[1]:
            self._render_machine_learning_tab(system)  # system 将被传递

        with tabs[2]:
            self._render_backtest_tab(system)

    def _render_technical_analysis_tab(self, system: Any):
        """
        [最终优化版] 渲染技术分析标签页，采用“状态驱动”模式，避免不必要的 Rerun。
        """
        st.subheader(translator.t('tech_analysis_tab', fallback="📈 技术分析驱动信号"))

        col_input, col_result = st.columns([1, 2])
        with col_input:
            symbol_tech = st.text_input(translator.t('stock_symbol'), "AAPL", key="tech_symbol_input").upper()
            strategy_options_map = {
                "ma_crossover": translator.t('strat_ma_crossover', fallback="均线交叉"),
                "rsi": translator.t('strat_rsi', fallback="RSI"),
                "macd": translator.t('strat_macd', fallback="MACD"),
                "bollinger": translator.t('strat_bollinger', fallback="布林带")
            }
            selected_strategy_key = st.selectbox(
                translator.t('tech_select_strategy'), options=list(strategy_options_map.keys()),
                format_func=lambda x: strategy_options_map.get(x, x), key="tech_strategy_select"
            )
            qty_tech_trade = st.number_input(
                translator.t('trade_quantity_tech'), min_value=1, value=10, step=1,
                key="qty_tech_trade_input", help=translator.t('trade_quantity_tech_help')
            )

        signal_state_key = f"tech_signal_result_{symbol_tech}_{selected_strategy_key}"

        if col_input.button(translator.t('tech_analyze_button'), key="analyze_tech_btn"):
            st.session_state[signal_state_key] = "LOADING"
            try:
                data = system.data_manager.get_historical_data(symbol_tech, days=252)
                if data is not None and not data.empty:
                    # 调用统一的、返回 DataFrame 的 analyze 方法
                    features_df = system.technical_analyzer.analyze(data)
                    # get_technical_signals 现在应该基于这个 features_df 来工作
                    st.session_state[signal_state_key] = self.get_technical_signals(features_df, selected_strategy_key)
                else:
                    st.session_state[signal_state_key] = {
                        "error": translator.t('error_fetching_data_for_analysis').format(symbol=symbol_tech)}
            except Exception as e:
                logger.error(f"Error during technical analysis for {symbol_tech}: {e}", exc_info=True)
                st.session_state[signal_state_key] = {"error": str(e)}
            # 按钮点击后 Streamlit 会自动 rerun 来显示新状态

        # --- 结果展示 ---
        signal_data = st.session_state.get(signal_state_key)

        if signal_data == "LOADING":
            with col_result:
                st.info("正在分析...")
        elif isinstance(signal_data, dict):
            with col_result:
                if "error" in signal_data:
                    st.error(signal_data["error"])
                elif signal_data.get("signal") == "ERROR":
                    st.error(translator.t('error_analysis_failed').format(reason=signal_data.get('reason', '')))
                else:
                    strategy_name = strategy_options_map.get(selected_strategy_key)
                    st.markdown(f"#### {symbol_tech} - {strategy_name}")

                    signal_val = signal_data.get("signal", "HOLD")
                    signal_color = {"BUY": "green", "SELL": "red", "HOLD": "gray"}.get(signal_val, "black")
                    translated_signal = translator.t(signal_val.lower(), fallback=signal_val)

                    st.markdown(
                        f"**{translator.t('tech_current_signal')}: <span style='color:{signal_color};'>{translated_signal}</span>**",
                        unsafe_allow_html=True)
                    st.caption(f"{translator.t('tech_reason')}: {signal_data.get('reason', '')}")

                    st.markdown(f"##### {translator.t('tech_latest_indicators')}:")
                    indicators = signal_data.get("indicators", {})
                    sub_cols = st.columns(2)
                    idx = 0
                    for k, v in indicators.items():
                        if isinstance(v, (int, float)) and pd.notna(v):
                            sub_cols[idx % 2].metric(label=str(k), value=f"{v:.2f}")
                            idx += 1

                    # --- 手动交易按钮 ---
                    if signal_val in ["BUY", "SELL"]:
                        btn_label_key = 'execute_buy_tech_signal' if signal_val == "BUY" else 'execute_sell_tech_signal'
                        qty_to_sell = system.portfolio.get('positions', {}).get(symbol_tech, {}).get('quantity', 0)
                        btn_fallback = f"执行买入 ({qty_tech_trade} 股)" if signal_val == "BUY" else f"执行卖出 ({qty_to_sell} 股)"
                        btn_label = translator.t(btn_label_key, fallback=btn_fallback).format(symbol=symbol_tech)

                        if st.button(btn_label, key=f"exec_tech_{signal_state_key}_btn"):
                            with st.spinner(translator.t('executing_trade_spinner')):
                                price_data = system.data_manager.get_realtime_price(symbol_tech)
                                exec_price = price_data['price'] if price_data else indicators.get('close')

                                if not exec_price:
                                    st.error(translator.t('error_cannot_get_price_for_trade'))
                                else:
                                    order_data = {
                                        "symbol": symbol_tech,
                                        "quantity": qty_tech_trade if signal_val == "BUY" else qty_to_sell,
                                        "price": exec_price, "direction": signal_val.capitalize(),
                                        "order_type": "Market Order"
                                    }
                                    if order_data["quantity"] > 0:
                                        trade_result = system.execute_trade(order_data)
                                        if trade_result and trade_result.get('success'):
                                            st.success(translator.t('trade_executed_successfully'))
                                            # 清除信号缓存以避免重复操作
                                            st.session_state.pop(signal_state_key, None)
                                            st.rerun()
                                        else:
                                            st.error(translator.t('trade_execution_failed_msg').format(
                                                message=trade_result.get('message', '')))
                                    else:
                                        st.warning("无可交易数量。")

            # --- 图表显示 (在主列) ---
            if isinstance(signal_data, dict) and signal_data.get("data") is not None and not signal_data["data"].empty:
                st.markdown("---")
                self._plot_technical_chart(
                    signal_data["data"], selected_strategy_key, symbol_tech,
                    translator.t(signal_data.get("signal", "HOLD").lower(), fallback=signal_data.get("signal", "HOLD"))
                )

    def _render_machine_learning_tab(self, system: Any):
        """
        [最终优化版] 渲染机器学习标签页，将预测和训练拆分为独立的辅助方法，并优化模型切换逻辑。
        """
        st.subheader(translator.t('ml_predict_tab', fallback="💡 机器学习预测与训练"))
        if not self.ml_strategy_instance:
            st.warning(translator.t('warning_ml_module_unavailable'));
            return

        # --- 1. 模型和股票选择 ---
        col1, col2 = st.columns(2)
        with col1:
            available_models = getattr(system.config, 'AVAILABLE_ML_MODELS', {})
            available_models_display_names = list(available_models.keys())
            default_model_name = getattr(system.config, 'DEFAULT_ML_MODEL_NAME', None)

            # 使用 session_state 追踪当前选择的模型，以在 reruns 之间保持状态
            if 'ml_selected_model' not in st.session_state:
                st.session_state.ml_selected_model = default_model_name if default_model_name in available_models_display_names else (
                    available_models_display_names[0] if available_models_display_names else None)

            new_model_option = translator.t('enter_new_model_option', fallback="(输入新模型名称)")
            options = available_models_display_names + [new_model_option]

            # 确保当前 session_state 中的选择在 options 列表中，以正确设置 index
            try:
                current_index = options.index(st.session_state.ml_selected_model)
            except (ValueError, IndexError):
                current_index = 0

            selected_model_display_name = st.selectbox(
                translator.t('select_ml_model', fallback="选择或输入模型名称:"),
                options,
                index=current_index,
                key="ml_model_selector_v10"
            )

            # 当 selectbox 的值发生变化时，更新 session_state
            if selected_model_display_name != st.session_state.ml_selected_model:
                st.session_state.ml_selected_model = selected_model_display_name
                # Streamlit selectbox 变化后会自动 rerun, 无需手动调用

            actual_model_name_to_use = st.session_state.ml_selected_model
            is_new_model_scenario = (actual_model_name_to_use == new_model_option)

            if is_new_model_scenario:
                actual_model_name_to_use = st.text_input(
                    translator.t('ml_new_model_name_prompt', fallback="新模型名称:"),
                    key="ml_new_model_input_v10"
                ).strip()

        with col2:
            symbol_ml = st.text_input(translator.t('stock_symbol'), "AAPL", key="ml_symbol_v10").upper()
            company_name_ml = st.text_input(translator.t('company_name_for_llm'), symbol_ml, key="ml_company_name_v10")

        # --- 更新活动模型 (在每次 Rerun 时检查是否需要加载) ---
        is_handler_invalid = self.ml_strategy_instance.active_model_handler is None or \
                             getattr(self.ml_strategy_instance.active_model_handler, 'model', None) is None

        # 只有在需要切换模型时才显示加载状态
        if actual_model_name_to_use and not is_new_model_scenario and \
                self.ml_strategy_instance.current_model_name != actual_model_name_to_use:

            with st.spinner(translator.t('loading_model_spinner', model_name=actual_model_name_to_use)):
                success = self.ml_strategy_instance.set_active_model(actual_model_name_to_use)
                if not success:
                    st.error(f"加载模型 '{actual_model_name_to_use}' 失败。")

        st.markdown("---")

        # --- 2. 动作选择 ---
        predict_label = translator.t('predict', fallback="预测")
        train_label = translator.t('train_model', fallback="训练模型")
        action_ml = st.radio(
            translator.t('select_action', fallback="选择操作:"),
            [predict_label, train_label],
            key="ml_action_radio_v10",
            horizontal=True
        )

        # --- 3. 渲染对应的 UI 和执行逻辑 ---
        if not symbol_ml: return

        if is_new_model_scenario and not actual_model_name_to_use and action_ml == train_label:
            st.warning(translator.t('info_enter_new_model_name'));
            return

        if action_ml == predict_label:
            self._render_ml_predict_ui(system, symbol_ml, company_name_ml)
        elif action_ml == train_label:
            self._render_ml_train_ui(system, symbol_ml, actual_model_name_to_use, is_new_model_scenario)

    def _render_ml_predict_ui(self, system: Any, symbol_ml: str, company_name: str):
        """
        [最终优化版] 渲染机器学习“预测”UI，使用后台线程，结果持久化在 session_state。
        """
        # --- 预测配置 ---
        use_llm_analysis = st.checkbox(
            translator.t('enable_llm_analysis_predict', fallback="结合 Gemini 新闻分析进行预测"),
            key="ml_use_llm_checkbox"
        )

        # --- 为后台任务定义一个唯一的 session_state 键 ---
        # 键包含了所有输入参数，确保每次不同预测都有独立的状态
        predict_state_key = f"ml_predict_state_{symbol_ml}_{self.ml_strategy_instance.current_model_name or 'none'}_{use_llm_analysis}"

        # --- 后台预测执行函数 ---
        def run_prediction_in_background():
            """这个函数将在一个独立的线程中运行，不会阻塞 UI。"""
            try:
                # 1. 更新状态：正在运行
                st.session_state[predict_state_key] = {"status": "running"}

                # 2. 检查模型
                if not self.ml_strategy_instance or not self.ml_strategy_instance.active_model_handler or getattr(
                        self.ml_strategy_instance.active_model_handler, 'model', None) is None:
                    raise ValueError(
                        translator.t('error_model_not_loaded_for_predict', fallback="模型未加载，无法预测。"))

                # 3. 获取数据
                # 为预测获取足够的回溯数据，例如 252 天（约一年）
                stock_data = system.data_manager.get_historical_data(symbol_ml, days=252)
                if stock_data is None or len(stock_data) < 65:  # 65 是一个保守的最小特征计算所需行数
                    raise ValueError(translator.t('error_insufficient_data_for_predict').format(symbol=symbol_ml))

                # 4. 获取量化预测
                quant_pred = self.ml_strategy_instance.predict(stock_data, symbol=symbol_ml)

                # 5. 获取文本分析 (如果启用)
                text_pred = None
                if use_llm_analysis:
                    text_extractor = self.text_feature_extractor
                    if text_extractor and text_extractor.is_available:
                        model_name = getattr(system.config, 'GEMINI_DEFAULT_MODEL', 'gemini-1.5-flash')
                        # get_and_extract_features 返回 (features, full_analysis)
                        _, text_pred = text_extractor.get_and_extract_features(symbol_ml, company_name, model_name)
                    else:
                        # 可以在结果中附加一个警告
                        text_pred = {"warning": translator.t('llm_module_unavailable_warning')}

                # 6. 将最终结果存入 session_state
                st.session_state[predict_state_key] = {"status": "completed",
                                                       "result": {"quant": quant_pred, "text": text_pred}}

            except Exception as e:
                logger.error(f"后台预测线程出错: {e}", exc_info=True)
                st.session_state[predict_state_key] = {"status": "failed", "error": str(e)}

        # --- UI 交互：按钮 ---
        if st.button(translator.t('ml_predict_button', fallback="执行预测"), key=f"predict_btn_{predict_state_key}",
                     use_container_width=True):
            # 点击按钮后，只做两件事：设置初始状态，然后启动线程
            st.session_state[predict_state_key] = {"status": "started"}
            thread = threading.Thread(target=run_prediction_in_background, daemon=True)
            thread.start()
            st.rerun()  # 立即刷新，UI 将捕捉到 "started" 状态并显示加载提示

        # --- 状态监控和结果显示 ---
        current_state = st.session_state.get(predict_state_key)

        if current_state:
            status = current_state.get("status")

            # 如果正在进行中，显示加载提示
            if status in ["started", "running"]:
                st.info(translator.t('ml_predicting_spinner_combined', fallback="正在执行量化模型预测与新闻分析..."))

            # 如果已完成，渲染结果
            elif status == "completed":
                results = current_state.get("result", {})
                quant_pred = results.get('quant') or {}  # 使用 or {} 避免 None
                text_pred = results.get('text') or {}  # 使用 or {} 避免 None

                res_col1, res_col2 = st.columns(2)

                # 在左侧显示量化模型结果
                with res_col1:
                    st.subheader(translator.t('quant_model_prediction', fallback="量化模型预测"))
                    if 'message' not in quant_pred and quant_pred:
                        direction_map = {-1: "持有", 0: "卖出", 1: "买入"}
                        direction_text = direction_map.get(quant_pred.get('direction', -1), "未知")
                        st.metric("模型信号 (Direction)", direction_text)
                        if 'probability_up' in quant_pred:
                            prob_up = quant_pred['probability_up']
                            st.progress(prob_up, text=f"上涨概率: {prob_up:.2%}")
                        if 'predicted_alpha' in quant_pred:
                            st.metric("预测 Alpha", f"{quant_pred['predicted_alpha']:.4f}")
                        if quant_pred.get('feature_importance'):
                            with st.expander("查看特征重要性"):
                                st.dataframe(pd.DataFrame(list(quant_pred['feature_importance'].items()),
                                                          columns=['Feature', 'Importance']))
                    elif status == "failed":
                                st.error(f"预测失败: {current_state.get('error')}")

                # 在右侧显示 Gemini 分析结果
                with res_col2:
                    st.subheader(translator.t('gemini_news_analysis', fallback="Gemini 新闻分析"))
                    if 'error' in text_pred:
                        st.error(f"Gemini Error: {text_pred['error']}")
                    elif 'warning' in text_pred:
                        st.warning(text_pred['warning'])
                    elif text_pred:
                        st.metric(translator.t('gemini_aggregated_sentiment'),
                                  f"{text_pred.get('aggregated_sentiment_score', 0.0):.2f}")
                        st.info(
                            f"**{translator.t('gemini_key_summary', fallback='核心摘要')}:** {text_pred.get('key_summary', 'N/A')}")
                        with st.expander(translator.t('gemini_analyzed_articles', fallback="查看分析的新闻源")):
                            for article in text_pred.get('analyzed_articles', []):
                                st.markdown(f"**[{article.get('title', 'No Title')}]({article.get('url', '#')})**")
                                st.caption(
                                    f"**摘要:** {article.get('summary', 'N/A')} | **情绪分:** {article.get('sentiment_score', 0.0):.2f}")
                    else:
                        st.info("未启用或未进行 Gemini 分析。")

                # --- 最终决策融合与自动化交易 ---
                st.markdown("---")
                st.subheader(translator.t('final_decision_and_auto_trade', fallback="最终决策与自动化"))

                final_col1, final_col2 = st.columns([1, 2])
                with final_col1:
                    quant_dir = quant_pred.get('direction', -1)
                    text_senti = text_pred.get('aggregated_sentiment_score', 0.0)

                    final_decision = "中性/持有"
                    if quant_dir == 1 and text_senti > 0.15:
                        final_decision = "强烈买入"
                    elif quant_dir == 0 and text_senti < -0.15:
                        final_decision = "强烈卖出"
                    elif quant_dir == 1:
                        final_decision = "弱买入"
                    elif quant_dir == 0:
                        final_decision = "弱卖出"
                    st.metric("综合建议", final_decision)
                with final_col2:
                    st.write(translator.t('ml_auto_trade_settings', fallback="自动化交易参数"))
                    auto_col1, auto_col2 = st.columns(2)
                    with auto_col1:
                        qty_ml_trade = st.number_input(translator.t('trade_quantity_ml'), min_value=1, value=10, step=1,
                                                       key="qty_ml_trade_v7")
                    with auto_col2:
                        ml_confidence_trade_threshold = st.slider(translator.t('ml_confidence_threshold_trade'), 0.50,
                                                                  0.99,
                                                                  0.65, 0.01, key="ml_confidence_slider_v7")

                    auto_trade_id = f"ml_{symbol_ml}_{self.ml_strategy_instance.current_model_name or 'default'}"
                    current_enabled = st.session_state.get('auto_trading_enabled', {}).get(auto_trade_id, False)
                    new_enabled = st.toggle(translator.t('enable_auto_trade_for_ml_strategy'), value=current_enabled,
                                            key=f"auto_toggle_{auto_trade_id}")

                    if new_enabled != current_enabled:
                        st.session_state.setdefault('auto_trading_enabled', {})[auto_trade_id] = new_enabled
                        if new_enabled:
                            st.session_state.setdefault('auto_trading_config', {})[auto_trade_id] = {
                                "name": f"ML: {self.ml_strategy_instance.current_model_name} for {symbol_ml}",
                                "type": "ml",
                                "symbol": symbol_ml, "ml_model_name": self.ml_strategy_instance.current_model_name,
                                "ml_confidence_threshold": ml_confidence_trade_threshold,
                                "trade_quantity": qty_ml_trade,
                                "interval_seconds": int(getattr(system.config, 'AUTO_TRADE_ML_INTERVAL', 300))
                            }
                            st.success(translator.t('auto_trade_enabled_for_strategy'))
                        else:
                            st.session_state.setdefault('auto_trading_config', {}).pop(auto_trade_id, None)
                            st.info(translator.t('auto_trade_disabled_for_strategy'))
                        st.rerun()

    def _render_ml_train_ui(self, system: Any, symbol_ml: str, model_name_to_use: str, is_new_model: bool):
        """
        [最终优化版] 渲染机器学习“训练”部分UI，使用后台线程执行训练，避免 UI 冻结。
        """
        st.write(translator.t('ml_train_settings_label', fallback="**训练参数**"))
        days_for_training = st.number_input(
            translator.t('ml_training_days', fallback="用于训练的数据天数:"),
            min_value=252, max_value=5000, value=1260, step=32,
            key="ml_training_days_v9"
        )

        # --- 准备按钮文本 ---
        label_key = 'ml_train_button_new' if is_new_model else 'ml_train_button_update'
        fallback_text = f"训练新模型 '{model_name_to_use}'" if is_new_model else f"训练并更新 '{model_name_to_use}'"
        button_text = translator.t(label_key, fallback=fallback_text).format(model_name=model_name_to_use)

        # --- 为后台任务定义 session_state 键 ---
        training_state_key = f"training_state_{symbol_ml}_{model_name_to_use}_{days_for_training}"

        # --- 后台训练执行函数 ---
        def run_training_in_background():
            """这个函数将在一个独立的线程中运行。"""
            try:
                # 1. 更新状态：正在获取数据
                st.session_state[training_state_key] = {
                    "status": "fetching_data", "progress": 0.1,
                    "message": translator.t('fetching_data_for_training_spinner', fallback="正在获取训练数据...")
                }

                # a. 动态计算需要的前置数据量
                available_models = getattr(self.ml_strategy_instance.config, 'AVAILABLE_ML_MODELS', {})
                model_filename = available_models.get(model_name_to_use)
                if is_new_model:
                    # 为新模型猜测文件名以获取类型
                    if "transformer" in model_name_to_use.lower():
                        model_filename = "new_model.h5"
                    elif "lstm" in model_name_to_use.lower():
                        model_filename = "new_model.h5"
                    else:
                        model_filename = "new_model.joblib"
                if not model_filename: raise ValueError(f"无法确定模型 '{model_name_to_use}' 的文件名。")

                model_type_temp = self.ml_strategy_instance._get_model_type(model_filename)
                hyperparams = self.ml_strategy_instance.config.ML_HYPERPARAMETERS.get(model_type_temp, {})
                lookback_buffer = hyperparams.get('lookback', 60) + 65  # 模型回溯期 + 特征计算最大回溯期

                # b. 获取数据
                training_data = system.data_manager.get_historical_data(
                    symbol_ml, days=(days_for_training + lookback_buffer), interval="1d"
                )
                if training_data is None or len(training_data) < lookback_buffer:
                    raise ValueError(translator.t('error_insufficient_training_data',
                                                  count=len(training_data) if training_data is not None else 0))

                # 2. 更新状态：正在训练
                st.session_state[training_state_key] = {
                    "status": "training", "progress": 0.3,
                    "message": translator.t('ml_training_in_progress_spinner_message',
                                            fallback=f"模型 '{model_name_to_use}' 正在训练中，请稍候...")
                }

                # 3. 调用耗时的训练方法
                train_result = self.ml_strategy_instance.train(
                    data=training_data,
                    symbol=symbol_ml,
                    model_display_name_to_save=model_name_to_use
                )

                # 4. 更新最终状态
                st.session_state[training_state_key] = {"status": "completed", "result": train_result,
                                                        "progress": 1.0}
            except Exception as e:
                logger.error(f"后台训练线程出错: {e}", exc_info=True)
                st.session_state[training_state_key] = {"status": "failed", "error": str(e)}

        # --- UI 交互逻辑 ---
        if st.button(button_text, key=f"train_btn_{training_state_key}", use_container_width=True):
            if not model_name_to_use:
                st.error(translator.t('error_model_name_empty', fallback="模型名称不能为空。"));
                return

            # 点击按钮后，设置初始状态并启动后台线程
            st.session_state[training_state_key] = {"status": "started", "progress": 0.0,
                                                    "message": "已启动训练任务..."}
            training_thread = threading.Thread(target=run_training_in_background, daemon=True)
            training_thread.start()
            st.rerun()

        # --- 状态监控和结果显示 ---
        current_state = st.session_state.get(training_state_key)
        if current_state:
            status = current_state.get("status")

            if status in ["started", "fetching_data", "training"]:
                st.progress(current_state.get("progress", 0), text=current_state.get("message"))
                st.info(translator.t('info_training_in_background',
                                     fallback="模型正在后台训练，您可以自由浏览其他页面或执行预测。"))

            elif status == "completed":
                train_result = current_state.get("result")
                if train_result and train_result.get('success'):
                    st.success(train_result.get('message', translator.t('ml_training_completed_default',
                                                                        fallback="模型训练完成。")))

                    res_cols = st.columns(3)
                    if 'train_score' in train_result:  # Sklearn
                        res_cols[0].metric(translator.t('ml_train_acc'),
                                           f"{train_result.get('train_score', 0):.2%}")
                        res_cols[1].metric(translator.t('ml_test_acc'), f"{train_result.get('test_score', 0):.2%}")
                        res_cols[2].metric(translator.t('ml_samples_used'), f"{train_result.get('n_samples', 0)}")
                    elif 'validation_metric' in train_result:  # DL
                        metric_name = "Validation Accuracy" if "acc" in str(
                            train_result.get('history', {})).lower() else "Validation Loss"
                        res_cols[0].metric(metric_name, f"{train_result.get('validation_metric', 0):.4f}")
                        res_cols[1].metric(translator.t('ml_samples_used'),
                                           f"{train_result.get('n_samples', 'N/A')}")

                    feat_importance = self.ml_strategy_instance.get_feature_importance()
                    if feat_importance:
                        st.subheader(
                            translator.t('ml_view_feature_importance_trained', fallback="已训练模型的特征重要性"))
                        importance_df = pd.DataFrame(list(feat_importance.items()),
                                                     columns=['Feature', 'Importance']).sort_values(by='Importance',
                                                                                                    ascending=False).head(
                            15)
                        fig = go.Figure(
                            [go.Bar(x=importance_df['Importance'], y=importance_df['Feature'], orientation='h')])
                        fig.update_layout(title="模型特征重要性", yaxis=dict(autorange="reversed"), height=500,
                                          margin=dict(l=120))
                        st.plotly_chart(fig, use_container_width=True)

                elif train_result:
                    st.error(translator.t('ml_training_failed_ui',
                                          error=train_result.get('message', translator.t('unknown_error'))))

                if st.button(translator.t('clear_results_button', fallback="清除结果"),
                             key=f"clear_btn_{training_state_key}"):
                    del st.session_state[training_state_key]
                    st.rerun()

            elif status == "failed":
                st.error(translator.t('ml_training_failed_ui', error=current_state.get('error', '未知错误')))
                if st.button(translator.t('clear_error_button', fallback="清除错误"),
                             key=f"clear_err_{training_state_key}"):
                    del st.session_state[training_state_key]
                    st.rerun()

    def _render_backtest_tab(self, system: Any):
        """
        [最终优化版] 渲染回测标签页，使用后台线程和 st.container 实现流畅的、无重置感的交互体验。
        """
        st.subheader(translator.t('backtest_tab', fallback="🔍 策略回测"))

        # --- 1. UI 输入元素 ---
        col_bt1, col_bt2 = st.columns([2, 1])
        with col_bt1:
            symbol_bt = st.text_input(translator.t('stock_symbol'), "AAPL", key="backtest_symbol_v4").upper()
            backtest_type_options = {
                "ml_quant": translator.t('backtest_type_ml', fallback="ML量化策略 (Alpha驱动)"),
                "technical": translator.t('backtest_type_tech', fallback="传统技术指标策略")
            }
            selected_backtest_type = st.radio(
                translator.t('backtest_type_label', fallback="选择回测类型:"),
                options=list(backtest_type_options.keys()),
                format_func=lambda x: backtest_type_options.get(x, x),
                key="backtest_type_radio_v4", horizontal=True
            )
        with col_bt2:
            initial_capital_bt = st.number_input(translator.t('backtest_initial_capital'), 1000.0, 1000000.0,
                                                 10000.0, key="backtest_capital_v4")
            days_for_backtest = st.number_input(translator.t('backtest_days'), 100, 5000, 730,
                                                key="backtest_days_v4")
            commission_bt = st.number_input(translator.t('backtest_commission'), 0.0, 0.01, 0.0003, 0.0001,
                                            format="%.4f", key="backtest_commission_v4")

        st.markdown("---")

        # --- 2. 根据类型显示不同的策略配置 ---
        strategy_config = {}
        strategy_display_name = ""

        if selected_backtest_type == "ml_quant":
            st.markdown(f"**{translator.t('ml_strategy_configuration', fallback='ML量化策略配置')}**")
            if not self.ml_strategy_instance:
                st.error("ML模块未初始化，无法进行ML策略回测。");
                return

            available_models = list(getattr(system.config, 'AVAILABLE_ML_MODELS', {}).keys())
            if not available_models: st.warning("配置中没有可用的ML模型。"); return

            selected_ml_model_bt = st.selectbox(
                translator.t('select_ml_model_for_backtest', fallback="选择用于回测的ML模型:"),
                options=available_models, key="backtest_ml_model_select_v4"
            )
            strategy_display_name = f"{backtest_type_options['ml_quant']} ({selected_ml_model_bt})"

            ml_param_col1, ml_param_col2 = st.columns(2)
            with ml_param_col1:
                st.write(translator.t('signal_generation_header', fallback="**信号生成**"))
                use_llm_backtest = st.checkbox(translator.t('incorporate_gemini_features'),
                                               key="backtest_use_llm_checkbox")
                llm_weight = st.slider(
                    translator.t('llm_signal_weight'), 0.0, 1.0, 0.3, 0.05, key="backtest_llm_weight",
                    disabled=not use_llm_backtest, help=translator.t('llm_alpha_weight_help')
                )
            with ml_param_col2:
                st.write(translator.t('threshold_and_risk_header', fallback="**交易阈值与风控**"))
                alpha_threshold_bt = st.slider(
                    translator.t('alpha_signal_threshold'), 0.01, 0.5, 0.1, 0.01, key="bt_alpha_threshold_v4"
                )
                vol_veto_bt = st.checkbox(translator.t('use_volatility_veto_checkbox'), value=True,
                                          key="bt_vol_veto_v4")

            strategy_config = {
                "name": f"ML_Quant_{selected_ml_model_bt}", "type": "ml_quant",
                "use_ml": True, "ml_model_name": selected_ml_model_bt,
                "volatility_veto": vol_veto_bt, "alpha_threshold": alpha_threshold_bt,
                "use_llm": use_llm_backtest, "llm_weight": llm_weight if use_llm_backtest else 0.0
            }
        elif selected_backtest_type == "technical":
            st.markdown(f"**{translator.t('tech_strategy_configuration', fallback='技术指标策略配置')}**")
            strategy_options_map_bt = {
                "ma_crossover": translator.t('strat_ma_crossover'), "rsi": translator.t('strat_rsi'),
                "macd": translator.t('strat_macd'), "bollinger": translator.t('strat_bollinger')
            }
            selected_tech_strategy = st.selectbox(
                translator.t('tech_select_strategy'), options=list(strategy_options_map_bt.keys()),
                format_func=lambda x: strategy_options_map_bt.get(x, x), key="backtest_tech_strategy_select"
            )
            strategy_display_name = strategy_options_map_bt[selected_tech_strategy]
            strategy_config = {"name": strategy_display_name, "type": "technical",
                               "strategy_type": selected_tech_strategy}

        # --- 3. 创建一个容器用于显示动态内容 (按钮、进度、结果) ---
        result_container = st.container()

        # --- 4. 为后台任务定义 session_state 键 ---
        config_tuple = tuple(sorted(strategy_config.items()))
        backtest_state_key = f"backtest_state_{symbol_bt}_{days_for_backtest}_{initial_capital_bt}_{config_tuple}"

        # --- 5. 后台回测执行函数 ---
        def run_backtest_in_background():
            """[新版] 这个函数将在一个独立的线程中运行所有耗时操作，只在最后更新一次状态。"""
            try:
                # a. 获取数据
                LOOKBACK_BUFFER = 150
                stock_data_bt = system.data_manager.get_historical_data(symbol_bt,
                                                                        days=(days_for_backtest + LOOKBACK_BUFFER))
                if stock_data_bt is None or stock_data_bt.empty:
                    raise ValueError(f"无法获取 {symbol_bt} 的数据进行回测。")

                # b. 获取 LLM 特征 (如果需要)
                if strategy_config.get("use_llm", False):
                    if self.text_feature_extractor and self.text_feature_extractor.is_available:
                        text_features_df = self.text_feature_extractor.get_and_extract_features_for_backtest(symbol_bt,
                                                                                                             stock_data_bt.index)
                        if text_features_df is not None:
                            stock_data_bt = stock_data_bt.join(text_features_df).ffill().bfill()

                # c. 运行回测引擎
                result = None
                if strategy_config.get("type") == "ml_quant":
                    result = self.backtest_strategy(symbol=symbol_bt, data=stock_data_bt,
                                                    strategy_config=strategy_config, initial_capital=initial_capital_bt,
                                                    commission_rate=commission_bt)
                elif strategy_config.get("type") == "technical":
                    result = self._backtest_technical_strategy(symbol=symbol_bt, data=stock_data_bt,
                                                               strategy_type=strategy_config["strategy_type"],
                                                               initial_capital=initial_capital_bt,
                                                               commission_rate=commission_bt)

                if not result: raise ValueError("回测引擎未返回有效结果。")

                # d. 任务完成，一次性写入最终结果
                st.session_state[backtest_state_key] = {"status": "completed", "result": result}
            except Exception as e:
                logger.error(f"后台回测线程出错: {e}", exc_info=True)
                st.session_state[backtest_state_key] = {"status": "failed", "error": str(e)}

        # --- 5. UI 交互：按钮 ---
        if st.button(translator.t('run_backtest_button'), key=f"run_btn_{backtest_state_key}",
                     use_container_width=True):
            if not symbol_bt: st.error(translator.t('error_stock_symbol_required')); return

            # 点击按钮后，只做两件事：设置初始状态，然后启动线程
            st.session_state[backtest_state_key] = {"status": "running"}
            thread = threading.Thread(target=run_backtest_in_background, daemon=True)
            thread.start()
            # **不再调用 st.rerun()**

        # --- 6. 状态监控和结果渲染 (在容器内进行) ---
        with result_container:
            current_state = st.session_state.get(backtest_state_key)

            if current_state:
                status = current_state.get("status")

                if status == "running":
                    # 只显示一个 spinner，不进行任何 sleep 或 rerun
                    with st.spinner(
                            translator.t('info_backtest_in_background_spinner', fallback="回测正在后台运行，请稍候...")):
                        # 创建一个循环来检查状态，直到它不再是 'running'
                        while st.session_state.get(backtest_state_key, {}).get("status") == "running":
                            time.sleep(0.5)  # 短暂休眠，避免 CPU 空转
                    # 当 spinner 结束时（即状态已改变），Streamlit 会自动 rerun
                    st.rerun()

                elif status == "completed":
                    backtest_run_result = current_state.get("result")
                    if backtest_run_result and backtest_run_result.get("success"):
                        st.success(backtest_run_result.get("message"))
                        stats = backtest_run_result.get("stats", {})
                        history_df = backtest_run_result.get("history_df")

                        st.markdown(f"#### {translator.t('backtest_performance_metrics_header')}")
                        res_col1, res_col2, res_col3 = st.columns(3)
                        with res_col1:
                            st.write(f"**{translator.t('return_metrics_header', fallback='回报指标')}**")
                            st.dataframe(pd.DataFrame({
                                "Metric": [translator.t('total_return_metric'),
                                           translator.t('annual_return_metric'),
                                           translator.t('buy_and_hold_metric')],
                                "Value": [f"{stats.get('total_return', 0):.2%}",
                                          f"{stats.get('annual_return', 0):.2%}",
                                          f"{stats.get('buy_hold_return', 0):.2%}"]
                            }), use_container_width=True)
                        with res_col2:
                            st.write(f"**{translator.t('risk_metrics_header', fallback='风险指标')}**")
                            st.dataframe(pd.DataFrame({
                                "Metric": [translator.t('sharpe_ratio_metric'), translator.t('max_drawdown_metric'),
                                           translator.t('volatility_metric'), translator.t('calmar_ratio_metric')],
                                "Value": [f"{stats.get('sharpe_ratio', 0):.2f}",
                                          f"{stats.get('max_drawdown', 0):.2%}",
                                          f"{stats.get('annual_volatility', 0):.2%}",
                                          f"{stats.get('calmar_ratio', 0):.2f}"]
                            }), use_container_width=True)
                        with res_col3:
                            st.write(f"**{translator.t('trade_metrics_header', fallback='交易指标')}**")
                            st.dataframe(pd.DataFrame({
                                "Metric": [translator.t('total_trades_metric'), translator.t('win_rate_metric'),
                                           translator.t('pl_ratio_metric')],
                                "Value": [f"{stats.get('trades', 0)}", f"{stats.get('win_rate', 0):.2%}",
                                          f"{stats.get('profit_loss_ratio', 0):.2f}"]
                            }), use_container_width=True)

                        alpha_scores = backtest_run_result.get("alpha_scores")
                        if alpha_scores:
                            st.markdown(f"#### {translator.t('alpha_distribution_header')}")
                            fig_alpha = go.Figure(data=[go.Histogram(x=alpha_scores, nbinsx=50)])
                            fig_alpha.update_layout(title=translator.t('alpha_distribution_subheader'),
                                                    xaxis_title="Alpha Score", yaxis_title="频率")
                            alpha_threshold = strategy_config.get('alpha_threshold', 0.15)
                            fig_alpha.add_vline(x=alpha_threshold, line_dash="dash", line_color="green",
                                                annotation_text=translator.t('buy_threshold_label'))
                            fig_alpha.add_vline(x=-alpha_threshold, line_dash="dash", line_color="red",
                                                annotation_text=translator.t('sell_threshold_label'))
                            st.plotly_chart(fig_alpha, use_container_width=True)
                            st.info(translator.t('alpha_distribution_help'))

                        self._plot_backtest_chart_with_trades(history_df, stats,
                                                              f"{symbol_bt} - {strategy_display_name}")


                    else:

                        st.error(backtest_run_result.get("message", "回测失败，未提供原因。"))

                    if st.button(translator.t('clear_results_button', fallback="清除回测结果"),
                                 key=f"clear_btn_{backtest_state_key}"):

                        del st.session_state[backtest_state_key]

                        st.rerun()


                    elif status == "failed":

                        st.error(translator.t('error_during_backtest',
                                              fallback="回测过程中发生错误:") + f" {current_state.get('error')}")

                        if st.button(translator.t('clear_error_button', fallback="清除错误"),
                                     key=f"clear_err_{backtest_state_key}"):
                            del st.session_state[backtest_state_key]

                            st.rerun()


    def _plot_backtest_chart_unified(self, history_df, initial_capital, title):
        """统一的回测图表绘制函数"""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3],
                            subplot_titles=(translator.t('equity_curve_vs_buy_hold'), translator.t('drawdown')))

        # 绘制策略权益曲线
        fig.add_trace(go.Scatter(x=history_df.index, y=history_df['total_value'], mode='lines', name='Strategy Equity'),
                      row=1, col=1)
        # 计算并绘制买入持有曲线
        buy_hold_equity = initial_capital * (history_df['price'] / history_df['price'].iloc[0])
        fig.add_trace(go.Scatter(x=history_df.index, y=buy_hold_equity, mode='lines', name='Buy & Hold Equity',
                                 line=dict(dash='dot')), row=1, col=1)

        # 绘制回撤曲线
        roll_max = history_df['total_value'].cummax()
        drawdown = (history_df['total_value'] / roll_max - 1.0) * 100  # In percent
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, mode='lines', name='Drawdown', fill='tozeroy',
                                 line=dict(color='red')), row=2, col=1)

        fig.update_layout(title=title, xaxis_title="Date", height=600)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

    def _plot_technical_chart(self, df: pd.DataFrame, strategy_type: str, symbol: str, current_signal: str):
        """绘制技术分析图表，包括当前信号标注"""
        if df is None or df.empty:
            st.warning(f"没有数据可用于为 {symbol} 绘制图表。")
            return

        fig_title = f"{symbol} - {strategy_type} 分析 (当前信号: {current_signal})"

        rows = 2
        row_heights = [0.7, 0.3]
        subplot_titles = ("价格与均线/布林带", "成交量/RSI/MACD")

        if strategy_type in ["rsi", "macd"]:  # RSI和MACD通常在下方独立面板显示
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                row_heights=row_heights, subplot_titles=subplot_titles)
        else:  # MA Crossover 和 Bollinger 可以直接画在价格图上
            fig = make_subplots(rows=1, cols=1, shared_xaxes=True, subplot_titles=(subplot_titles[0],))  # 只用一个子图
            rows = 1  # 更新行数

        # K线图 (如果数据完整) 或 收盘价线图
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                                         name='K线'), row=1, col=1)
        elif 'close' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='收盘价', line=dict(color='black')),
                          row=1, col=1)
        else:
            st.warning("K线图或收盘价数据不完整。")
            return  # 没有价格数据无法继续

        # 根据策略类型添加指标
        if strategy_type == "ma_crossover":
            if 'MA5' in df.columns: fig.add_trace(
                go.Scatter(x=df.index, y=df['MA5'], mode='lines', name='MA5', line=dict(color='blue')), row=1, col=1)
            if 'MA20' in df.columns: fig.add_trace(
                go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20', line=dict(color='orange')), row=1,
                col=1)
            if rows == 2 and 'volume' in df.columns:  # 如果有第二行，显示成交量
                fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='成交量', marker_color='rgba(100,100,100,0.3)'),
                              row=2, col=1)
                fig.update_yaxes(title_text="成交量", row=2, col=1)


        elif strategy_type == "rsi":
            if 'RSI' in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple')),
                              row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超买 (70)", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超卖 (30)", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1)


        elif strategy_type == "macd":
            if 'MACD' in df.columns: fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue')), row=2, col=1)
            if 'Signal_Line' in df.columns: fig.add_trace(
                go.Scatter(x=df.index, y=df['Signal_Line'], mode='lines', name='信号线', line=dict(color='orange')),
                row=2, col=1)
            # MACD柱状图 (MACD - Signal_Line)
            if 'MACD' in df.columns and 'Signal_Line' in df.columns:
                macd_hist = df['MACD'] - df['Signal_Line']
                colors = ['green' if val >= 0 else 'red' for val in macd_hist]
                fig.add_trace(go.Bar(x=df.index, y=macd_hist, name='MACD柱', marker_color=colors), row=2, col=1)
            fig.update_yaxes(title_text="MACD", row=2, col=1)

        elif strategy_type == "bollinger":
            if 'BB_Mid' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['BB_Mid'], mode='lines', name='BB中轨',
                                                                line=dict(color='blue', dash='dash')), row=1, col=1)
            if 'BB_Upper' in df.columns: fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', name='BB上轨', line=dict(color='red')), row=1,
                col=1)
            if 'BB_Lower' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', name='BB下轨', line=dict(color='green'),
                               fill='tonexty', fillcolor='rgba(255,0,0,0.1)'), row=1, col=1)  # 填充上轨和下轨之间
                fig.data[-1].update(fillcolor='rgba(230,230,250,0.2)')  # 调整填充色

            if rows == 2 and 'volume' in df.columns:
                fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='成交量', marker_color='rgba(100,100,100,0.3)'),
                              row=2, col=1)
                fig.update_yaxes(title_text="成交量", row=2, col=1)

        fig.update_layout(title_text=fig_title, height=500 if rows == 1 else 700, showlegend=True,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                          xaxis_rangeslider_visible=False)
        fig.update_yaxes(title_text="价格", row=1, col=1)
        st.plotly_chart(fig, use_container_width=True)

    def _plot_backtest_chart(self, backtest_key: str, symbol: str, strategy_name_display: str):
        """绘制回测结果图表"""
        if backtest_key not in self.backtest_results:
            st.warning("回测结果未找到，无法绘制图表。")
            return

        bt_result_data = self.backtest_results[backtest_key]["data"]
        if bt_result_data.empty:
            st.warning("回测数据为空，无法绘制图表。")
            return

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            row_heights=[0.5, 0.25, 0.25],
                            subplot_titles=(
                            f"{symbol} - 价格与交易点 ({strategy_name_display})", "投资组合净值 vs 买入持有", "回撤"))

        # 图1: 价格与交易点
        if all(col in bt_result_data.columns for col in ['open', 'high', 'low', 'close']):
            fig.add_trace(go.Candlestick(x=bt_result_data.index,
                                         open=bt_result_data['open'], high=bt_result_data['high'],
                                         low=bt_result_data['low'], close=bt_result_data['close'],
                                         name='K线'), row=1, col=1)
        elif 'close' in bt_result_data.columns:
            fig.add_trace(go.Scatter(x=bt_result_data.index, y=bt_result_data['close'], mode='lines', name='收盘价',
                                     line=dict(color='black')), row=1, col=1)

        # 添加买卖信号点
        if 'Buy_Plot_Signal' in bt_result_data.columns:
            fig.add_trace(go.Scatter(x=bt_result_data.index, y=bt_result_data['Buy_Plot_Signal'], mode='markers',
                                     name='买入', marker=dict(symbol='triangle-up', size=10, color='green')), row=1,
                          col=1)
        if 'Sell_Plot_Signal' in bt_result_data.columns:
            fig.add_trace(go.Scatter(x=bt_result_data.index, y=bt_result_data['Sell_Plot_Signal'], mode='markers',
                                     name='卖出', marker=dict(symbol='triangle-down', size=10, color='red')), row=1,
                          col=1)

        # 图2: 投资组合净值 vs 买入持有
        if 'Portfolio_Total' in bt_result_data.columns:
            initial_capital = self.backtest_results[backtest_key]["stats"]["initial_capital"]
            portfolio_normalized = bt_result_data['Portfolio_Total'] / initial_capital
            fig.add_trace(
                go.Scatter(x=bt_result_data.index, y=portfolio_normalized, mode='lines', name='策略净值 (归一化)',
                           line=dict(color='blue')), row=2, col=1)

        if 'close' in bt_result_data.columns and not bt_result_data['close'].empty:
            buy_hold_normalized = bt_result_data['close'] / bt_result_data['close'].iloc[0]
            fig.add_trace(
                go.Scatter(x=bt_result_data.index, y=buy_hold_normalized, mode='lines', name='买入持有 (归一化)',
                           line=dict(color='grey')), row=2, col=1)

        fig.add_hline(y=1.0, line_dash="dash", line_color="black", row=2, col=1)  # 基准线

        # 图3: 回撤
        if 'Drawdown' in bt_result_data.columns:
            fig.add_trace(
                go.Scatter(x=bt_result_data.index, y=bt_result_data['Drawdown'] * 100, mode='lines', name='回撤 (%)',
                           line=dict(color='red'), fill='tozeroy'), row=3, col=1)

        fig.update_layout(title_text=f"{symbol} - {strategy_name_display}策略回测结果", height=800, showlegend=True,
                          legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
                          xaxis_rangeslider_visible=False)
        fig.update_yaxes(title_text="价格", row=1, col=1)
        fig.update_yaxes(title_text="归一化净值", row=2, col=1)
        fig.update_yaxes(title_text="回撤 (%)", row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)