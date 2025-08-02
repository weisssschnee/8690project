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

from .llm_trader_adapters import GeminiTraderAdapter, DeepSeekTraderAdapter, BaseLLMTraderAdapter

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.system import TradingSystem

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

        # --- 初始化 MLStrategy 实例 (保持原有逻辑) ---
        self.ml_strategy_instance: Optional[MLStrategy] = None
        can_init_ml = (SKLEARN_AVAILABLE or TENSORFLOW_AVAILABLE) and \
                      MLStrategy is not None and \
                      hasattr(self.system, 'config') and self.system.config is not None and \
                      hasattr(self.system, 'data_manager') and self.system.data_manager is not None

        if can_init_ml:
            try:
                self.ml_strategy_instance = MLStrategy(
                    config=self.system.config,
                    data_manager_ref=self.system.data_manager
                )
                logger.info("MLStrategy instance (with its TextFeatureExtractor) was successfully created.")
            except Exception as e:
                logger.error(f"Failed to create MLStrategy instance: {e}", exc_info=True)
                self.ml_strategy_instance = None
        else:
            logger.warning("ML features are disabled because one or more prerequisites are not met:")
            if not (SKLEARN_AVAILABLE or TENSORFLOW_AVAILABLE):
                logger.warning("- ML libraries (Sklearn/TensorFlow) are not available.")
            if not hasattr(self.system, 'config') or self.system.config is None:
                logger.warning("- System.config is not available.")
            if not hasattr(self.system, 'data_manager') or self.system.data_manager is None:
                logger.warning("- System.data_manager is not available.")

        try:
            from .llm_trader_adapters import GeminiTraderAdapter, DeepSeekTraderAdapter, BaseLLMTraderAdapter
            from .llm_trader_adapters import GEMINI_AVAILABLE, DEEPSEEK_AVAILABLE
            LLM_ADAPTERS_AVAILABLE = True
            print("✅ LLM适配器模块和可用性标志导入成功")
        except ImportError as e:
            GeminiTraderAdapter, DeepSeekTraderAdapter, BaseLLMTraderAdapter = None, None, None
            GEMINI_AVAILABLE, DEEPSEEK_AVAILABLE = False, False
            LLM_ADAPTERS_AVAILABLE = False
            logging.warning(f"LLM适配器无法导入 ({e})，LLM功能将不可用。")

        # --- 初始化 LLM Traders (增强调试) ---
        print("=" * 80)
        print("🚀 开始初始化 LLM Trader Adapters...")
        logger.info("🚀 开始初始化 LLM Trader Adapters...")

        self.llm_traders = {}

        if not LLM_ADAPTERS_AVAILABLE:
            print("❌ LLM适配器模块不可用，跳过LLM初始化")
            logger.error("❌ LLM adapter modules not available, skipping LLM initialization")
            print("=" * 80)
            return

        try:
            # 获取配置
            available_llms = getattr(self.system.config, 'AVAILABLE_LLM_TRADERS', {})
            print(f"📋 配置中的AVAILABLE_LLM_TRADERS: {available_llms}")
            logger.info(f"📋 配置中的AVAILABLE_LLM_TRADERS: {available_llms}")

            if not available_llms:
                print("❌ AVAILABLE_LLM_TRADERS 配置为空")
                logger.error("❌ AVAILABLE_LLM_TRADERS 配置为空")
                print("=" * 80)
                return

            for display_name, provider in available_llms.items():
                print(f"\n🔍 处理 LLM: {display_name} (provider: {provider})")
                provider = provider.upper()

                # 检查API密钥
                api_key_attr = f'{provider}_API_KEY'
                api_key = getattr(self.system.config, api_key_attr, None)
                print(f"   检查API密钥 {api_key_attr}: {'✅ 已配置' if api_key else '❌ 未配置'}")

                # 检查模型名称
                model_attr = f'{provider}_DEFAULT_MODEL'
                model_name = getattr(self.system.config, model_attr, None)
                print(f"   检查模型配置 {model_attr}: {model_name if model_name else '❌ 未配置'}")

                if not api_key:
                    print(f"   ⚠️ 跳过 {display_name}: API密钥未配置")
                    logger.warning(f"⚠️ SKIPPING {display_name}: {provider}_API_KEY not found in config.")
                    continue

                if not model_name:
                    print(f"   ⚠️ 跳过 {display_name}: 模型未配置")
                    logger.warning(f"⚠️ SKIPPING {display_name}: {provider}_DEFAULT_MODEL not found in config.")
                    continue

                try:
                    print(f"   🔧 正在创建 {provider} 适配器...")
                    if provider == 'GEMINI':
                        print(f"   检查Gemini可用性: {GEMINI_AVAILABLE}")
                        if not GEMINI_AVAILABLE:
                            print(f"   ❌ Gemini库不可用")
                            continue

                        self.llm_traders[display_name] = GeminiTraderAdapter(api_key=api_key, model_name=model_name)
                        print(f"   ✅ 成功创建 '{display_name}' (模型: '{model_name}')")
                        logger.info(f"✅ SUCCESSFULLY initialized '{display_name}' with model '{model_name}'.")

                    elif provider == 'DEEPSEEK':
                        print(f"   检查DeepSeek可用性: {DEEPSEEK_AVAILABLE}")
                        if not DEEPSEEK_AVAILABLE:
                            print(f"   ❌ DeepSeek库不可用")
                            continue

                        self.llm_traders[display_name] = DeepSeekTraderAdapter(api_key=api_key, model_name=model_name)
                        print(f"   ✅ 成功创建 '{display_name}' (模型: '{model_name}')")
                        logger.info(f"✅ SUCCESSFULLY initialized '{display_name}' with model '{model_name}'.")
                    else:
                        print(f"   ❌ 未知的LLM提供商: '{provider}'")
                        logger.error(f"Unknown LLM provider '{provider}' in config.")

                except Exception as e:
                    print(f"   ❌ 创建 '{display_name}' 失败: {e}")
                    logger.error(f"❌ FAILED to initialize '{display_name}': {e}", exc_info=True)

            # 总结
            final_count = len(self.llm_traders)
            final_names = list(self.llm_traders.keys())
            print(f"\n🏁 LLM初始化完成!")
            print(f"   成功创建: {final_count} 个")
            print(f"   可用列表: {final_names}")
            logger.info(f"🏁 LLM initialization completed: {final_count} traders available: {final_names}")

            if not self.llm_traders:
                print("❌ 没有成功初始化任何LLM Trader Adapters！")
                logger.error(
                    "No LLM Trader Adapters were successfully initialized. Check API keys and SDK installations.")

        except Exception as e:
            print(f"❌ LLM初始化过程中发生严重错误: {e}")
            logger.error(f"Critical error during LLM initialization: {e}", exc_info=True)
            import traceback
            print(traceback.format_exc())

        print("=" * 80)

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
            signal_state_key = f"tech_signal_result_{symbol_tech}_{selected_strategy_key}"
            st.session_state[signal_state_key] = "LOADING"
            try:
                REQUIRED_DAYS_FOR_ANALYSIS = 100

                data = system.data_manager.get_historical_data(symbol_tech, days=REQUIRED_DAYS_FOR_ANALYSIS)

                # Now, check if we received *enough* data points (e.g., at least 60)
                if data is not None and not data.empty and len(data) >= 60:
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
        signal_state_key = f"tech_signal_result_{symbol_tech}_{selected_strategy_key}"
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
                        portfolio_state = st.session_state.get('portfolio', {})
                        qty_to_sell = portfolio_state.get('positions', {}).get(symbol_tech, {}).get('quantity', 0)

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
                    df=signal_data["data"],
                    strategy_type=selected_strategy_key,
                    symbol=symbol_tech,
                    current_signal=signal_data.get("signal", "HOLD")
                )

    def _render_machine_learning_tab(self, system: Any):
        """
        [最终修复版] 渲染机器学习标签页，修复了模型切换时的状态同步问题。
        """
        st.subheader(translator.t('ml_predict_tab', fallback="🤖 机器学习预测与训练"))
        if not self.ml_strategy_instance:
            st.warning(translator.t('warning_ml_module_unavailable'));
            return

        # --- 1. 模型和股票选择 ---
        col1, col2 = st.columns(2)
        with col1:
            # (这部分 UI 定义保持不变)
            available_models = getattr(system.config, 'AVAILABLE_ML_MODELS', {})
            available_models_display_names = list(available_models.keys())
            new_model_option = translator.t('enter_new_model_option')
            options = available_models_display_names + [new_model_option]

            # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
            # 直接使用 session_state 来控制 selectbox，使其成为唯一的数据源
            if 'ml_selected_model_name' not in st.session_state:
                st.session_state.ml_selected_model_name = getattr(system.config, 'DEFAULT_ML_MODEL_NAME', options[0])

            def on_model_change():
                # 定义一个回调函数，当 selectbox 变化时，清空旧的预测状态
                # 这是一个好的实践，避免显示旧模型对新股票的预测
                # pass # 暂时不做任何事
                st.session_state.pop(
                    f"ml_predict_state_{st.session_state.get('ml_symbol_v11', 'AAPL')}_{st.session_state.ml_selected_model_name}_True",
                    None)
                st.session_state.pop(
                    f"ml_predict_state_{st.session_state.get('ml_symbol_v11', 'AAPL')}_{st.session_state.ml_selected_model_name}_False",
                    None)

            st.selectbox(
                translator.t('select_ml_model'), options,
                key='ml_selected_model_name',  # 直接绑定 session_state 的键
                on_change=on_model_change  # 添加回调
            )

            actual_model_name_to_use = st.session_state.ml_selected_model_name
            is_new_model_scenario = (actual_model_name_to_use == new_model_option)
            if is_new_model_scenario:
                actual_model_name_to_use = st.text_input(translator.t('ml_new_model_name_prompt'),
                                                         key="ml_new_model_input")

        with col2:
            symbol_ml = st.text_input(translator.t('stock_symbol'), "AAPL", key="ml_symbol_v11").upper()
            company_name_ml = st.text_input(translator.t('company_name_for_llm'), symbol_ml, key="ml_company_name_v11")

        # --- 2. 在所有 UI 渲染之前，立即处理模型加载 ---
        if actual_model_name_to_use and not is_new_model_scenario:
            # 只有在选择的模型与 MLStrategy 内部状态不一致时才加载
            if self.ml_strategy_instance.current_model_name != actual_model_name_to_use:
                with st.spinner(translator.t('loading_model_spinner', model_name=actual_model_name_to_use)):
                    self.ml_strategy_instance.set_active_model(actual_model_name_to_use)
                # set_active_model 会更新 current_model_name，无需 rerun

        # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^

        st.markdown("---")

        # --- 3. 动作选择和渲染 (现在可以安全地读取 current_model_name) ---
        predict_label, train_label = translator.t('predict'), translator.t('train_model')
        action_ml = st.radio(translator.t('select_action'), [predict_label, train_label], key="ml_action_radio",
                             horizontal=True)

        if not symbol_ml: return
        if is_new_model_scenario and not actual_model_name_to_use and action_ml == train_label:
            st.warning(translator.t('info_enter_new_model_name'));
            return

        if action_ml == predict_label:
            self._render_ml_predict_ui(system, symbol_ml, company_name_ml, is_new_model_scenario=is_new_model_scenario)
        elif action_ml == train_label:
            self._render_ml_train_ui(system, symbol_ml, actual_model_name_to_use, is_new_model_scenario)

    def _plot_technical_chart(self, df: pd.DataFrame, strategy_type: str, symbol: str, current_signal: str):
        """[CRASH FIX] Renders the technical analysis chart, correcting the row_heights mismatch."""
        if df is None or df.empty:
            st.warning(f"No data available to plot chart for {symbol}.");
            return

        # --- 1. Translate all text first ---
        strategy_options_map = {
            "ma_crossover": translator.t('strat_ma_crossover'), "rsi": translator.t('strat_rsi'),
            "macd": translator.t('strat_macd'), "bollinger": translator.t('strat_bollinger')
        }
        strategy_display = strategy_options_map.get(strategy_type, strategy_type)
        translated_signal = translator.t(current_signal.lower(), fallback=current_signal)
        fig_title = translator.t('chart_title_tech_analysis',
                                 fallback="{symbol} - {strategy} Analysis (Current Signal: {signal})").format(
            symbol=symbol, strategy=strategy_display, signal=translated_signal)

        # --- 2. Determine the correct layout based on strategy type (THE FIX IS HERE) ---
        if strategy_type in ["ma_crossover", "bollinger"]:
            # These strategies only need one row.
            rows = 1
            row_heights = None  # No height argument needed for a single row
            subplot_titles = (translator.t('chart_subplot_price'),)
        else:  # "rsi" and "macd" need two rows.
            rows = 2
            row_heights = [0.7, 0.3]  # Height argument is valid for two rows
            subplot_titles = (translator.t('chart_subplot_price'), translator.t('chart_subplot_indicators'))

        # --- 3. Create the subplot figure with now-consistent arguments ---
        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=row_heights,  # This will be None or [0.7, 0.3], now always correct
            subplot_titles=subplot_titles
        )

        # --- 4. Add traces (this logic was already correct) ---
        # Main Price Chart (Candlestick or Line)
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                                         name=translator.t('candlestick_label')), row=1, col=1)
        elif 'close' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name=translator.t('close_price_label'),
                                     line=dict(color='black')), row=1, col=1)
        else:
            st.warning("Price data for chart is incomplete.");
            return

        # Add indicators based on strategy type
        if strategy_type == "ma_crossover":
            if 'MA5' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], mode='lines', name='MA5'), row=1,
                                                  col=1)
            if 'MA20' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20'),
                                                   row=1, col=1)

        elif strategy_type == "rsi":
            if 'RSI' in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red",
                              annotation_text=translator.t('rsi_overbought_label'), row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green",
                              annotation_text=translator.t('rsi_oversold_label'), row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1)

        elif strategy_type == "macd":
            if 'MACD' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'),
                                                   row=2, col=1)
            if 'Signal_Line' in df.columns: fig.add_trace(
                go.Scatter(x=df.index, y=df['Signal_Line'], mode='lines', name=translator.t('macd_signal_line_label')),
                row=2, col=1)
            if 'MACD' in df.columns and 'Signal_Line' in df.columns:
                macd_hist = df['MACD'] - df['Signal_Line']
                colors = ['green' if val >= 0 else 'red' for val in macd_hist]
                fig.add_trace(
                    go.Bar(x=df.index, y=macd_hist, name=translator.t('macd_hist_label'), marker_color=colors), row=2,
                    col=1)
            fig.update_yaxes(title_text="MACD", row=2, col=1)

        elif strategy_type == "bollinger":
            if 'BB_Mid' in df.columns: fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_Mid'], mode='lines', name=translator.t('bollinger_mid_label'),
                           line=dict(color='blue', dash='dash')), row=1, col=1)
            if 'BB_Upper' in df.columns: fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', name=translator.t('bollinger_upper_label'),
                           line=dict(color='red')), row=1, col=1)
            if 'BB_Lower' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', name=translator.t('bollinger_lower_label'),
                               line=dict(color='green'), fill='tonexty', fillcolor='rgba(230,230,250,0.2)'), row=1,
                    col=1)

        # Layout and Axis Titles
        fig.update_layout(title_text=fig_title, height=500 if rows == 1 else 700, showlegend=True,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                          xaxis_rangeslider_visible=False)
        fig.update_yaxes(title_text=translator.t('price_axis_label', fallback="Price"), row=1, col=1)
        st.plotly_chart(fig, use_container_width=True)

    def get_signal_for_autotrader(self, config: dict) -> dict:
        """
        [实现] 为后台自动交易服务获取最终的交易信号。
        这是这个方法正确的位置。
        """
        symbol = config.get("symbol")
        model_name = config.get("ml_model_name")
        # use_llm = config.get("llm_enabled", False) # 未来可以启用

        logger.info(f"[UnifiedStrategy] Getting signal for {symbol} using model '{model_name}'")

        try:
            # 1. 检查 MLStrategy 实例是否存在
            if not self.ml_strategy_instance:
                raise RuntimeError("MLStrategy instance is not available within UnifiedStrategy.")

            # 2. 设置要使用的模型
            if not self.ml_strategy_instance.set_active_model(model_name):
                raise RuntimeError(f"Failed to load or set active model '{model_name}'.")

            # 3. 获取预测所需的数据
            # 需要足够的回溯数据，具体数值取决于模型
            lookback_buffer = 150
            latest_data = self.system.data_manager.get_historical_data(symbol, days=lookback_buffer)
            if latest_data is None or latest_data.empty:
                raise ValueError(f"Could not fetch sufficient historical data for {symbol}.")

            # 4. 调用 MLStrategy 的核心预测方法
            quant_prediction = self.ml_strategy_instance.predict(latest_data, symbol=symbol)

            if not quant_prediction or 'message' in quant_prediction:
                raise ValueError(
                    f"Prediction failed: {quant_prediction.get('message', 'Unknown error from MLStrategy.predict')}")

            # 5. [重要] 为后台服务补充 'confidence' 字段
            #    后台服务需要这个字段来与阈值比较
            if 'probability_up' in quant_prediction:
                prob_up = quant_prediction['probability_up']
                # 信度 = 概率离0.5（不确定性）的距离，乘以2进行归一化
                confidence = abs(prob_up - 0.5) * 2
                quant_prediction['confidence'] = confidence
            elif 'predicted_alpha' in quant_prediction:
                # 对于 Alpha 模型，我们可以基于 alpha 值的大小来估算一个信度
                alpha = quant_prediction['predicted_alpha']
                confidence = min(1.0, abs(alpha) / 0.05)  # 假设 alpha 达到 0.05 就有100%信度
                quant_prediction['confidence'] = confidence
            else:
                # 如果模型不输出概率或alpha，信度为0
                quant_prediction['confidence'] = 0.0

            # (未来可以在这里融合 LLM 信号)

            logger.info(f"[UnifiedStrategy] Successfully generated signal for {symbol}: {quant_prediction}")
            return quant_prediction

        except Exception as e:
            logger.error(f"Error in get_signal_for_autotrader for {symbol}: {e}", exc_info=True)
            return {"message": str(e)}

    def _render_ml_predict_ui(self, system: Any, symbol_ml: str, company_name: str, is_new_model_scenario: bool):
        """
        [新增] 渲染机器学习的“预测”部分UI。
        """
        # --- 预测配置 ---
        use_llm_analysis = st.checkbox(
            translator.t('enable_llm_analysis_predict', fallback="结合 Gemini 新闻分析进行预测"),
            key="ml_use_llm_checkbox"
        )

        # --- 预测执行按钮 ---
        if st.button(translator.t('ml_predict_button', fallback="执行预测"), key="ml_predict_btn",
                     use_container_width=True):
            if not self.ml_strategy_instance or not self.ml_strategy_instance.active_model_handler or getattr(
                    self.ml_strategy_instance.active_model_handler, 'model', None) is None:
                st.error(translator.t('error_model_not_loaded_for_predict', fallback="模型未加载，无法预测。"));
                return

            cached_result = system.persistence_manager.get_prediction_result(
                symbol=symbol_ml,
                model_name=self.ml_strategy_instance.current_model_name,
                use_llm=use_llm_analysis
            )

            if cached_result:
                # 缓存命中！直接将结果放入 session_state 并刷新
                st.session_state[f'ml_predict_result_{symbol_ml}'] = cached_result
                st.toast("从本地缓存加载了预测结果。")  # 提示用户
                st.rerun()

            else:
                with st.spinner(
                        translator.t('ml_predicting_spinner_combined', fallback="正在执行量化模型预测与新闻分析...")):
                    LOOKBACK_BUFFER = 150
                stock_data = system.data_manager.get_historical_data(symbol_ml, days=LOOKBACK_BUFFER)
                # 1. 获取基础数据
                if stock_data is None or len(stock_data) < 65:
                    st.error(translator.t('error_insufficient_data_for_predict').format(symbol=symbol_ml));
                    return

                # 2. 获取量化预测
                quant_prediction = self.ml_strategy_instance.predict(stock_data, symbol=symbol_ml)

                # 3. 获取文本分析 (如果启用)
                text_analysis = None
                if use_llm_analysis:
                    if self.text_feature_extractor and self.text_feature_extractor.is_available:
                        model_name = getattr(system.config, 'GEMINI_DEFAULT_MODEL', 'gemini-2.5-flash')
                        # get_and_extract_features 返回 (features, full_analysis)，我们需要第二个
                        _, text_analysis = self.text_feature_extractor.get_and_extract_features(symbol_ml, company_name,
                                                                                                model_name)
                    else:
                        st.warning(translator.t('llm_module_unavailable_warning'))

                final_result = {"quant": quant_prediction, "text": text_analysis}

                # --- 2. 将新结果存入持久化缓存 ---
                system.persistence_manager.set_prediction_result(
                    symbol=symbol_ml,
                    model_name=self.ml_strategy_instance.current_model_name,
                    use_llm=use_llm_analysis,
                    result=final_result
                )

                # 4. 将结果存入 session_state
                st.session_state[f'ml_predict_result_{symbol_ml}'] = {"quant": quant_prediction, "text": text_analysis}
            st.rerun()

        # --- 渲染预测结果 ---
        result_key = f'ml_predict_result_{symbol_ml}'
        if result_key in st.session_state:
            results = st.session_state[result_key]
            quant_pred = results.get('quant')
            text_pred = results.get('text')

            res_col1, res_col2 = st.columns(2)

            # 在左侧显示量化模型结果
            with res_col1:
                st.subheader(translator.t('quant_model_prediction', fallback="量化模型预测"))
                if quant_pred and 'message' not in quant_pred:
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
                elif quant_pred:
                    st.error(f"{translator.t('predict_failed', fallback='预测失败')}: {quant_pred.get('message')}")
                else:
                    st.info("量化模型未返回预测结果。")

            # 在右侧显示 Gemini 分析结果
            with res_col2:
                st.subheader(translator.t('gemini_news_analysis', fallback="Gemini 新闻分析"))
                if text_pred and 'error' not in text_pred:
                    st.metric(translator.t('gemini_aggregated_sentiment'),
                              f"{text_pred.get('aggregated_sentiment_score', 0.0):.2f}")
                    st.info(
                        f"**{translator.t('gemini_key_summary', fallback='核心摘要')}:** {text_pred.get('key_summary', 'N/A')}")
                    with st.expander(translator.t('gemini_analyzed_articles', fallback="查看分析的新闻源")):
                        for article in text_pred.get('analyzed_articles', []):
                            st.markdown(f"**[{article.get('title', 'No Title')}]({article.get('url', '#')})**")
                            st.caption(
                                f"**摘要:** {article.get('summary', 'N/A')} | **情绪分:** {article.get('sentiment_score', 0.0):.2f}")
                elif text_pred and 'error' in text_pred:
                    st.error(f"Gemini Error: {text_pred['error']}")
                else:
                    st.info("未启用或未进行 Gemini 分析。")

            # --- 最终决策融合与自动化交易 ---
            st.markdown("---")
            st.subheader(translator.t('final_decision_and_auto_trade', fallback="最终决策与自动化"))

            final_col1, final_col2 = st.columns([1, 2])
            with final_col1:
                quant_dir = quant_pred.get('direction', -1) if quant_pred else -1
                text_senti = text_pred.get('aggregated_sentiment_score',
                                           0.0) if text_pred and 'error' not in text_pred else 0.0

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

            # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
            with st.container(border=True):
                current_model_name = self.ml_strategy_instance.current_model_name
                if is_new_model_scenario or not current_model_name:
                    st.info(translator.t('info_select_model_for_auto_trade'))
                else:
                    st.write(f"**{translator.t('ml_auto_trade_settings')}**")
                    auto_col1, auto_col2 = st.columns(2)
                    with auto_col1:
                        qty_ml_trade = st.number_input(translator.t('trade_quantity_ml'), min_value=1, value=10, step=1, key="qty_ml_trade_v9")
                    with auto_col2:
                        ml_confidence_trade_threshold = st.slider(translator.t('ml_confidence_threshold_trade'), 0.50, 0.99, 0.65, 0.01, key="ml_confidence_slider_v9")

                    user_id = st.session_state.get('username', 'Guest')
                    auto_trade_id = f"auto_ml_{user_id}_{symbol_ml}_{current_model_name}"

                    existing_config = system.persistence_manager.load_strategy_config(auto_trade_id, user_id)
                    current_enabled = existing_config.get('enabled', False) if existing_config else False

                    new_enabled = st.toggle(
                        translator.t('enable_auto_trade_for_ml_strategy'),
                        value=current_enabled,
                        key=f"auto_toggle_{auto_trade_id}"
                    )

                    if current_enabled and existing_config:
                        last_exec_time = existing_config.get('last_executed')
                        if last_exec_time:
                            # 将'YYYY-MM-DD HH:MM:SS.ffffff'格式的字符串转换为datetime对象
                            try:
                                dt_obj = datetime.fromisoformat(last_exec_time.split('.')[0])
                                time_ago = datetime.now() - dt_obj
                                if time_ago.total_seconds() < 120:
                                    status_text = f"约 {int(time_ago.total_seconds())} 秒前"
                                else:
                                    status_text = f"约 {int(time_ago.total_seconds() / 60)} 分钟前"
                                st.caption(
                                    f"📈 **状态:** {translator.t('running', fallback='运行中')} (上次心跳: {status_text})")
                            except (ValueError, TypeError):
                                st.caption(
                                    f"📈 **状态:** {translator.t('running', fallback='运行中')} (上次心跳: {last_exec_time})")
                        else:
                            st.caption(
                                f"📈 **状态:** {translator.t('pending_execution', fallback='已启用，等待服务执行')}")
                    elif current_enabled:
                        st.caption(f"📈 **状态:** {translator.t('pending_execution', fallback='已启用，等待服务执行')}")

                    if new_enabled != current_enabled:
                        if new_enabled:
                            strategy_config_to_save = {
                                "strategy_id": auto_trade_id,
                                "user_id": user_id,
                                "core_type": "ml_model",
                                "type": "ml_quant",
                                "enabled": True,
                                "symbol": symbol_ml,
                                "ml_model_name": current_model_name,
                                "trade_quantity": qty_ml_trade,
                                "ml_confidence_threshold": ml_confidence_trade_threshold,
                                "llm_enabled": use_llm_analysis,
                                "llm_weight": 0.3, # Example weight, can be made configurable
                            }
                            system.persistence_manager.save_strategy_config(strategy_config_to_save)
                            st.success(translator.t('auto_trade_enabled_for_strategy'))
                        else:
                            system.persistence_manager.delete_strategy_config(strategy_id=auto_trade_id, user_id=user_id)
                            st.info(translator.t('auto_trade_disabled_for_strategy'))
                        st.rerun()

    def run_training_flow(self, symbol: str, days: int, model_name: str, is_new: bool) -> Dict:
        """
        [新增] 完整的、独立的模型训练业务逻辑流程。
        这个方法包含了所有耗时操作，专门用于被后台线程调用。
        """
        logger.info(f"Starting training flow for {symbol} with model {model_name}...")

        # 1. 获取数据 (包含 buffer)
        # a. 动态计算 buffer
        available_models = getattr(self.ml_strategy_instance.config, 'AVAILABLE_ML_MODELS', {})
        model_filename = available_models.get(model_name)
        if is_new:  # 为新模型猜测文件名以获取类型
            if "transformer" in model_name.lower():
                model_filename = "new.h5"
            elif "lstm" in model_name.lower():
                model_filename = "new.h5"
            else:
                model_filename = "new.joblib"
        if not model_filename: raise ValueError(f"无法确定模型 '{model_name}' 的文件名。")

        model_type = self.ml_strategy_instance._get_model_type(model_filename)
        hyperparams = self.ml_strategy_instance.config.ML_HYPERPARAMETERS.get(model_type, {})
        lookback_buffer = hyperparams.get('lookback', 60) + 65

        # b. 获取数据
        training_data = self.system.data_manager.get_historical_data(symbol, days=(days + lookback_buffer))
        if training_data is None or len(training_data) < lookback_buffer:
            raise ValueError(translator.t('error_insufficient_training_data',
                                          count=len(training_data) if training_data is not None else 0))

        logger.info(f"Data fetched for training. Shape: {training_data.shape}")

        # 2. 调用 MLStrategy 的核心训练方法
        train_result = self.ml_strategy_instance.train(
            data=training_data,
            symbol=symbol,
            model_display_name_to_save=model_name
        )

        logger.info(f"Training flow finished. Result: {train_result.get('success')}")
        return train_result

    def _render_ml_train_ui(self, system: Any, symbol_ml: str, model_name_to_use: str, is_new_model_scenario: bool):
        """
        [最终优化版] 渲染机器学习“训练”UI，只负责调用后台任务和渲染状态。
        """
        st.write(translator.t('ml_train_settings_label', fallback="**训练参数**"))
        days_for_training = st.number_input(
            translator.t('ml_training_days'), min_value=252, max_value=5000, value=1260, key="ml_training_days_v9"
        )

        # --- 准备按钮文本 (修复了变量名) ---
        label_key = 'ml_train_button_new' if is_new_model_scenario else 'ml_train_button_update'
        fallback_text = f"训练新模型 '{model_name_to_use}'" if is_new_model_scenario else f"训练并更新 '{model_name_to_use}'"
        button_text = translator.t(label_key, fallback=fallback_text).format(model_name=model_name_to_use)

        # --- 为后台任务定义 session_state 键 ---
        training_state_key = f"training_state_{symbol_ml}_{model_name_to_use}_{days_for_training}"

        # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
        # --- UI 交互逻辑 (现在调用独立的业务流程) ---
        if st.button(button_text, key=f"train_btn_{training_state_key}", use_container_width=True):
            if not model_name_to_use:
                st.error(translator.t('error_model_name_empty'));
                return

            st.session_state[training_state_key] = {"status": "running"}
            # 后台线程现在调用干净的业务逻辑方法 run_training_flow (修复了 is_new 参数)
            thread = threading.Thread(
                target=lambda: st.session_state.update({
                    training_state_key: self.run_training_flow(
                        symbol=symbol_ml, days=days_for_training,
                        model_name=model_name_to_use, is_new=is_new_model_scenario
                    )
                }),
                daemon=True
            )
            thread.start()
            st.rerun()
        # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^

        # --- 状态监控和结果显示 ---
        current_state = st.session_state.get(training_state_key)
        if current_state:
            status = current_state.get("status")

            # 这里需要一个转换：run_training_flow 会直接返回最终结果字典，
            # 所以我们需要检查 success 键来判断状态
            if status == "running":
                with st.spinner(translator.t('ml_training_in_progress_spinner_message', model_name=model_name_to_use)):
                    # 使用 while 循环等待后台任务完成
                    while st.session_state.get(training_state_key, {}).get("status") == "running":
                        time.sleep(0.5)
                st.rerun()  # 任务完成，刷新以显示结果

            # 检查 train_result 是否存在 (即 status 不再是 running)
            elif 'success' in current_state:
                train_result = current_state
                if train_result.get('success'):
                    st.success(train_result.get('message'))
                    # (渲染成功的指标和图表，与您现有代码一致)
                    res_cols = st.columns(3)
                    # ...
                else:
                    st.error(translator.t('ml_training_failed_ui', error=train_result.get('message')))

                if st.button(translator.t('clear_results_button'), key=f"clear_btn_{training_state_key}"):
                    del st.session_state[training_state_key];
                    st.rerun()

            # 旧的状态 'failed' (以防万一)
            elif status == "failed":
                st.error(translator.t('ml_training_failed_ui', error=current_state.get('error')))
                if st.button(translator.t('clear_error_button'), key=f"clear_err_{training_state_key}"):
                    del st.session_state[training_state_key];
                    st.rerun()

    def _render_backtest_tab(self, system: Any):
        """
        [FINAL TRANSLATED VERSION] Renders the backtest tab, supporting technical and ML+LLM strategies.
        All UI strings are now covered by translation keys.
        """
        st.subheader(translator.t('backtest_tab', fallback="🔍 Strategy Backtesting"))

        # --- 1. General backtest parameters ---
        col_bt1, col_bt2 = st.columns([2, 1])
        with col_bt1:
            symbol_bt = st.text_input(translator.t('stock_symbol'), "AAPL", key="backtest_symbol_v4").upper()
            backtest_type_options = {
                "ml_quant": translator.t('backtest_type_ml', fallback="ML Quant Strategy (Alpha Driven)"),
                "technical": translator.t('backtest_type_tech', fallback="Traditional Technical Indicator Strategy")
            }
            selected_backtest_type = st.radio(
                translator.t('backtest_type_label', fallback="Select Backtest Type:"),
                options=list(backtest_type_options.keys()),
                format_func=lambda x: backtest_type_options[x],
                key="backtest_type_radio_v4", horizontal=True
            )
        with col_bt2:
            initial_capital_bt = st.number_input(translator.t('backtest_initial_capital', fallback="Initial Capital"),
                                                 1000.0, 1000000.0, 10000.0, key="backtest_capital_v4")
            days_for_backtest = st.number_input(translator.t('backtest_days', fallback="Backtest Days"), 100, 5000, 730,
                                                key="backtest_days_v4")
            commission_bt = st.number_input(translator.t('backtest_commission', fallback="Commission Rate"), 0.0, 0.01,
                                            0.0003, 0.0001, format="%.4f", key="backtest_commission_v4")

        st.markdown("---")

        # --- 2. Strategy-specific configurations ---
        strategy_config = {}
        strategy_display_name = ""

        if selected_backtest_type == "ml_quant":
            st.markdown(f"**{translator.t('ml_strategy_configuration', fallback='ML Quant Strategy Configuration')}**")

            if not self.ml_strategy_instance:
                st.error("ML module not initialized, cannot perform ML backtest.");
                return

            available_models = list(getattr(system.config, 'AVAILABLE_ML_MODELS', {}).keys())
            if not available_models: st.warning("No available ML models in config."); return

            selected_ml_model_bt = st.selectbox(
                translator.t('select_ml_model_for_backtest', fallback="Select ML Model for Backtest:"),
                options=available_models, key="backtest_ml_model_select_v4"
            )
            strategy_display_name = f"{backtest_type_options['ml_quant']} ({selected_ml_model_bt})"

            ml_param_col1, ml_param_col2 = st.columns(2)
            with ml_param_col1:
                st.write(f"**{translator.t('signal_generation', fallback='Signal Generation')}**")
                use_llm_backtest = st.checkbox(
                    translator.t('enable_llm_analysis_backtest', fallback="Include Gemini News Features"),
                    key="backtest_use_llm_checkbox"
                )
                llm_weight = st.slider(
                    translator.t('llm_alpha_weight', fallback="LLM Signal Weight"),
                    0.0, 1.0, 0.3, 0.05,
                    key="backtest_llm_weight",
                    disabled=not use_llm_backtest,
                    help=translator.t('llm_alpha_weight_help',
                                      fallback="Weight of the LLM sentiment in the final Alpha decision.")
                )
            with ml_param_col2:
                st.write(f"**{translator.t('trade_thresholds_risk', fallback='Trade Thresholds & Risk')}**")
                alpha_threshold_bt = st.slider(
                    translator.t('alpha_threshold', fallback="Alpha Signal Threshold"),
                    0.01, 0.5, 0.1, 0.01, key="bt_alpha_threshold_v4"
                )
                vol_veto_bt = st.checkbox(
                    translator.t('use_vol_veto', fallback="Use Volatility Veto"),
                    value=True, key="bt_vol_veto_v4"
                )

            strategy_config = {
                "name": f"ML_Quant_{selected_ml_model_bt}", "type": "ml_quant",
                "use_ml": True, "ml_model_name": selected_ml_model_bt,
                "volatility_veto": vol_veto_bt, "alpha_threshold": alpha_threshold_bt,
                "use_llm": use_llm_backtest, "llm_weight": llm_weight if use_llm_backtest else 0.0
            }

        elif selected_backtest_type == "technical":
            st.markdown(
                f"**{translator.t('tech_strategy_configuration', fallback='Technical Indicator Strategy Configuration')}**")
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

        # --- 3. Execution button and logic ---
        if st.button(translator.t('backtest_run_button', fallback="🚀 Run Backtest"), key="run_backtest_btn_final",
                     use_container_width=True):
            if not symbol_bt: st.error(translator.t('error_stock_symbol_required')); return

            with st.spinner(translator.t('backtesting_spinner_v2',
                                         fallback="Running backtest for {symbol} with {strategy_name} strategy...").format(
                    symbol=symbol_bt, strategy_name=strategy_display_name)):
                try:
                    LOOKBACK_BUFFER = 150
                    stock_data_bt = system.data_manager.get_historical_data(symbol_bt,
                                                                            days=(days_for_backtest + LOOKBACK_BUFFER))

                    if stock_data_bt is None or stock_data_bt.empty:
                        st.error(f"Could not fetch data for {symbol_bt} to run backtest.");
                        return

                    backtest_run_result = self.backtest_strategy(
                        symbol=symbol_bt, data=stock_data_bt, strategy_config=strategy_config,
                        initial_capital=initial_capital_bt, commission_rate=commission_bt
                    )

                    if backtest_run_result and backtest_run_result.get("success"):
                        st.success(backtest_run_result.get("message"))
                        stats = backtest_run_result.get("stats", {})
                        history_df = backtest_run_result.get("history_df")

                        st.markdown(
                            f"#### {translator.t('backtest_results_header', fallback='Backtest Performance Metrics')}")

                        performance_metrics = {
                            translator.t("total_return",
                                         fallback="Total Return"): f"{stats.get('total_return', 0):.2%}",
                            translator.t("annual_return",
                                         fallback="Annual Return"): f"{stats.get('annual_return', 0):.2%}",
                            translator.t("buy_hold_return",
                                         fallback="Buy & Hold Return"): f"{stats.get('buy_hold_return', 0):.2%}"
                        }
                        risk_metrics = {
                            translator.t("sharpe_ratio",
                                         fallback="Sharpe Ratio"): f"{stats.get('sharpe_ratio', 0):.2f}",
                            translator.t("max_drawdown",
                                         fallback="Max Drawdown"): f"{stats.get('max_drawdown', 0):.2%}",
                            translator.t("annual_volatility",
                                         fallback="Annual Volatility"): f"{stats.get('annual_volatility', 0):.2%}",
                            translator.t("calmar_ratio", fallback="Calmar Ratio"): f"{stats.get('calmar_ratio', 0):.2f}"
                        }
                        trade_metrics = {
                            translator.t("total_trades", fallback="Total Trades"): f"{stats.get('trades', 0)}",
                            translator.t("win_rate", fallback="Win Rate"): f"{stats.get('win_rate', 0):.2%}",
                            translator.t("profit_loss_ratio",
                                         fallback="P/L Ratio"): f"{stats.get('profit_loss_ratio', 0):.2f}"
                        }

                        res_col1, res_col2, res_col3 = st.columns(3)
                        with res_col1:
                            st.write(f"**{translator.t('returns_metrics', fallback='Return Metrics')}**")
                            st.dataframe(pd.DataFrame(performance_metrics.items(), columns=['Metric', 'Value']),
                                         use_container_width=True)
                        with res_col2:
                            st.write(f"**{translator.t('risk_metrics', fallback='Risk Metrics')}**")
                            st.dataframe(pd.DataFrame(risk_metrics.items(), columns=['Metric', 'Value']),
                                         use_container_width=True)
                        with res_col3:
                            st.write(f"**{translator.t('trade_metrics', fallback='Trade Metrics')}**")
                            st.dataframe(pd.DataFrame(trade_metrics.items(), columns=['Metric', 'Value']),
                                         use_container_width=True)

                        alpha_scores = backtest_run_result.get("alpha_scores")
                        if alpha_scores:
                            st.markdown(
                                f"#### {translator.t('alpha_distribution_header', fallback='Alpha Signal Distribution')}")
                            fig_alpha = go.Figure(data=[go.Histogram(x=alpha_scores, nbinsx=50)])
                            fig_alpha.update_layout(title=translator.t('alpha_distribution_title',
                                                                       fallback="Model Alpha Signal Distribution"),
                                                    xaxis_title=translator.t('alpha_score_axis',
                                                                             fallback="Alpha Score"),
                                                    yaxis_title=translator.t('frequency_axis', fallback="Frequency"))
                            alpha_threshold = strategy_config.get('alpha_threshold', 0.15)
                            fig_alpha.add_vline(x=alpha_threshold, line_width=2, line_dash="dash", line_color="green",
                                                annotation_text=translator.t('buy_threshold_label',
                                                                             fallback="Buy Threshold"))
                            fig_alpha.add_vline(x=-alpha_threshold, line_width=2, line_dash="dash", line_color="red",
                                                annotation_text=translator.t('sell_threshold_label',
                                                                             fallback="Sell Threshold"))
                            st.plotly_chart(fig_alpha, use_container_width=True)
                            st.info(translator.t('alpha_distribution_help',
                                                 fallback="Observing the Alpha distribution can help you determine if the threshold is reasonable."))

                        self._plot_backtest_chart_with_trades(history_df, stats,
                                                              f"{symbol_bt} - {strategy_display_name}")

                    elif backtest_run_result:
                        st.error(backtest_run_result.get("message", "Backtest failed with no specific message."))
                    else:
                        st.error("Backtest process did not return a valid result.")
                except Exception as e_backtest:
                    logger.error(f"Error during backtest execution: {e_backtest}", exc_info=True)
                    st.error(translator.t('error_during_backtest', fallback="An error occurred during the backtest."));
                    st.code(traceback.format_exc())

    def get_llm_trader_signal(self, config: dict, contextual_data: dict) -> dict:
            """
            [新增] 获取 LLM 交易员的决策信号。
            """
            symbol = config.get("symbol")
            llm_name = config.get("llm_name")
            user_id = config.get("user_id")

            logger.info(f"[UnifiedStrategy] Getting signal from LLM Trader '{llm_name}' for {symbol}")

            try:
                if llm_name not in self.llm_traders:
                    raise ValueError(f"LLM Trader '{llm_name}' is not available or configured.")

                # --- 1. 构建 Prompt Context ---
                prompt_context = self._build_llm_prompt_context(symbol, user_id, contextual_data)

                # --- 2. 调用对应的 LLM 适配器 ---
                adapter = self.llm_traders[llm_name]
                decision = adapter.get_decision(prompt_context)

                # --- 3. 保存决策到数据库以供UI展示 ---
                if "error" not in decision:
                    self.system.persistence_manager.save_strategy_last_decision(config['strategy_id'], decision)

                return decision

            except Exception as e:
                logger.error(f"Error getting LLM trader signal for {symbol}: {e}", exc_info=True)
                return {"error": str(e)}

    # In core/strategy/unified_strategy.py -> class UnifiedStrategy

    def _build_llm_prompt_context(self, symbol: str, user_id: str, contextual_data: dict) -> str:
        """
        [DEFINITIVE HARDENED VERSION] Builds a context-aware prompt for the LLM.
        This version is structurally guaranteed to always return a valid, non-empty prompt string.
        """

        # --- 1. Initialize with a default, failsafe prompt. This is our guarantee. ---
        # This prompt will be used if any subsequent step fails.
        prompt = f"""
        CRITICAL FALLBACK: System failed to generate a valid prompt.
        Your task is to analyze the stock {symbol}.
        Provide a trading decision (BUY, SELL, or HOLD) in the required JSON format based on your general knowledge.
        {{
            "decision": "HOLD",
            "confidence": 0.50,
            "reasoning": "Failsafe activation: The primary analysis system failed to provide data. Holding position due to lack of information.",
            "risk_parameters": {{ "stop_loss": null, "take_profit": null }}
        }}
        """

        try:
            # --- 2. Determine if system data is truly available and usable ---
            historical_data = contextual_data.get('historical_data')
            is_data_available = (
                    historical_data is not None and
                    isinstance(historical_data, pd.DataFrame) and
                    not historical_data.empty and
                    'close' in historical_data.columns and
                    len(historical_data) > 20  # Require a minimum number of rows for indicators
            )

            # --- 3. Prepare common information (always available) ---
            portfolio = self.system.persistence_manager.load_portfolio(user_id)
            position = portfolio.get('positions', {}).get(symbol) if portfolio else None
            position_status = f"Currently HOLDING {position['quantity']} shares." if position else "Currently NOT HOLDING any shares."

            news_list = contextual_data.get('news', [])
            news_headlines = "\n".join([f"- {n['title']}" for n in
                                        news_list[:5]]) if news_list else "No recent news available from system feed."

            # --- 4. Attempt to build the appropriate prompt ---
            if is_data_available:
                # --- HAPPY PATH: Data is available, build the detailed analysis prompt ---
                logger.info(f"Building data-rich prompt for {symbol} as data IS available.")
                tech_signals = self.get_technical_signals(historical_data)
                indicators = tech_signals.get('indicators', {})

                rsi_val = indicators.get('RSI')
                rsi_str = f"{rsi_val:.2f}" if isinstance(rsi_val, (int, float)) else "N/A"
                macd_hist = indicators.get('hist')
                macd_hist_str = f"{macd_hist:.4f}" if isinstance(macd_hist, (int, float)) else "N/A"
                close_price = indicators.get('close')
                close_price_str = f"${close_price:.2f}" if isinstance(close_price, (int, float)) else "N/A"

                # Overwrite the default prompt with the data-rich version
                prompt = f"""
                As "Alpha Agent", a quantitative trader, analyze the provided data for {symbol} and make a trading decision for the next 1-day horizon.
                **//-- INSTRUCTIONS & RULES --//**
                1.  **Role**: You are analytical, unemotional, and concise.
                2.  **Decision**: Your final `decision` MUST be one of: "BUY", "SELL", or "HOLD".
                3.  **Output**: You MUST respond ONLY with a single, valid JSON object.
                **//-- PROVIDED SYSTEM DATA --//**
                **1. Current Portfolio Status:** {position_status}
                **2. Technical Analysis Snapshot:**
                - Last Close Price: {close_price_str}
                - RSI(14): {rsi_str}
                - MACD Histogram: {macd_hist_str}
                **3. Recent News Headlines (from system):**
                {news_headlines}
                **//-- REQUIRED OUTPUT FORMAT (JSON ONLY) --//**
                {{
                  "decision": "BUY", "confidence": 0.75, "reasoning": "1. Technical Analysis: [Your reasoning]. 2. News/Sentiment Analysis: [Your analysis]. 3. Conclusion: [Your final summary].",
                  "risk_parameters": {{ "stop_loss": 150.50, "take_profit": 175.00 }}
                }}
                """
            else:
                # --- FALLBACK PATH: Data is NOT available, build the autonomous researcher prompt ---
                logger.warning(f"Building autonomous research prompt for {symbol} as system data is UNAVAILABLE.")
                # Overwrite the default prompt with the researcher version
                prompt = f"""
                As "Alpha Agent", a quantitative trader, make a trading decision for {symbol} for the next 1-day horizon.
                **//-- CRITICAL ALERT: SYSTEM DATA UNAVAILABLE --//**
                The primary data feed has failed. You must operate autonomously using your own internal knowledge and available tools to gather the necessary information.
                **//-- INSTRUCTIONS & RULES --//**
                1.  **Mission**: First, determine the most recent closing price for {symbol}. Second, analyze its immediate trend. Third, find any highly significant recent news. Finally, synthesize all your findings to form a trading decision.
                2.  **Decision**: Your final `decision` MUST be one of: "BUY", "SELL", or "HOLD".
                3.  **Output**: You MUST respond ONLY with a single, valid JSON object.
                **//-- CONTEXTUAL INFORMATION --//**
                **1. Current Portfolio Status:** {position_status}
                **2. Recent News Headlines (from system, may be stale):**
                {news_headlines}
                **//-- REQUIRED OUTPUT FORMAT (JSON ONLY) --//**
                {{
                  "decision": "HOLD", "confidence": 0.60, "reasoning": "1. Autonomous Data Gathering: [State the closing price and trend you found]. 2. News Analysis: [Summarize critical news you found, or state that none was found]. 3. Conclusion: [Combine findings into a final decision].",
                  "risk_parameters": {{ "stop_loss": 150.50, "take_profit": 175.00 }}
                }}
                """

        except Exception as e:
            logger.error(f"CRITICAL ERROR during prompt generation for {symbol}: {e}", exc_info=True)
            # If any error occurs, the pre-initialized failsafe prompt will be used by default.

        logger.debug(f"Final Generated Prompt for {symbol}:\n{prompt}")
        return prompt


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


