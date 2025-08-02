# core/risk/manager.py
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from core.translate import translator
from typing import Dict, Optional, List, Any
logger = logging.getLogger(__name__)
import streamlit as st


@st.cache_data(ttl=3600, show_spinner="正在获取市场状态数据...")  # 缓存1小时
def get_market_regime_data(_system_ref: Any, market_benchmark: str = "SPY") -> Optional[pd.DataFrame]:
    """获取用于判断市场状态的宏观数据 (VIX 和市场基准)。"""
    if not (_system_ref and _system_ref.data_manager):
        logger.error("无法获取市场状态数据：DataManager 不可用。")
        return None
    try:
        # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
        # --- 调用新的、专门的指数获取方法 ---
        vix_data = _system_ref.data_manager.get_index_data("^VIX", days=300)
        benchmark_data = _system_ref.data_manager.get_index_data(market_benchmark, days=300)
        # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^

        if vix_data is None or benchmark_data is None or vix_data.empty or benchmark_data.empty:
            logger.warning("获取 VIX 或市场基准数据失败，无法判断市场状态。")
            return None

        # 合并数据
        combined_df = pd.DataFrame(index=benchmark_data.index)
        combined_df['benchmark_close'] = benchmark_data['close']
        combined_df['vix_close'] = vix_data['close']
        combined_df.ffill(inplace=True)  # 向前填充以对齐日期
        return combined_df

    except Exception as e:
        logger.error(f"获取市场状态数据时出错: {e}", exc_info=True)
        return None


class RiskManager:
    def __init__(self, config):
        self.config = config
        # 直接从 Config 实例获取 RISK_LIMITS 字典
        self.risk_limits = getattr(config, 'RISK_LIMITS', { # 使用 getattr 安全获取
            "max_position_size": 0.3, # Fallback defaults if RISK_LIMITS not in config
            "max_drawdown": 0.15,
            "var_confidence": 0.95,
            "max_sector_exposure": 0.3
        })
        logger.info(f"RiskManager initialized with limits: {self.risk_limits}")

    def validate_order(self, order: Dict, portfolio: Dict, system_ref: Optional[Any] = None) -> Dict:  # 添加 system_ref
        """验证订单是否符合风险控制标准"""
        logger.debug(f"Validating order: {order} against portfolio: {portfolio.get('cash')}")
        symbol = order.get('symbol')
        quantity = order.get('quantity', 0)
        price_from_order = order.get('price')  # Price from order, could be None
        direction = order.get('direction', '')
        order_type = order.get('order_type', 'Market Order')

        # vvvvvvvvvvvvvvvvvvvv START OF MODIFIED SECTION vvvvvvvvvvvvvvvvvvvv
        validation_price = price_from_order

        if order_type == 'Market Order' and validation_price is None:
            # --- 关键修改：检查配置开关 ---
            should_fetch_price = getattr(self.config, 'RISK_FETCH_MARKET_PRICE', False)  # 安全地获取开关状态，默认为True

            if should_fetch_price and system_ref and hasattr(system_ref,
                                                             'data_manager') and system_ref.data_manager is not None:
                
                logger.debug(f"RiskValidate: Market order for {symbol}, fetching realtime price for validation.")
                rt_price_data = system_ref.data_manager.get_realtime_price(symbol)
                if rt_price_data and rt_price_data.get('price') is not None:
                    validation_price = rt_price_data['price']
                    logger.info(
                        f"RiskValidate: Using fetched realtime price {validation_price} for market order {symbol}.")
                else:
                    logger.warning(
                        f"RiskValidate: Market order for {symbol}, FAILED to fetch realtime price. Cannot validate order value accurately.")
                    # 决定如何处理：可以返回验证失败，或者跳过基于价值的检查
                    # 为安全起见，如果无法估价，可以认为验证失败或高风险
                    return {'valid': False, 'reason': translator.t('risk_market_order_no_price_validate',
                                                                   fallback="市价单无法获取估价进行风险校验")}
            else:
                logger.warning(
                    f"RiskValidate: Market order for {symbol} has no price, and no system_ref to fetch it. Cannot validate order value.")
                return {'valid': False, 'reason': translator.t('risk_market_order_no_price_source_validate',
                                                               fallback="市价单无估价来源进行风险校验")}

        elif validation_price is None:  # Limit order or other type that still has None price
            logger.error(f"RiskValidate: Order for {symbol} (type: {order_type}) has None price. Invalid order.")
            return {'valid': False,
                    'reason': translator.t('risk_order_price_is_none', fallback="订单价格为None，无法校验")}

        # 确保 quantity 和 validation_price 是数值
        try:
            quantity = float(quantity)
            validation_price = float(validation_price)
        except (ValueError, TypeError):
            logger.error(
                f"RiskValidate: Invalid quantity or price type for {symbol}. Qty: {quantity}, Price: {validation_price}")
            return {'valid': False,
                    'reason': translator.t('risk_invalid_qty_price_type', fallback="订单数量或价格类型无效")}

        order_value = abs(quantity * validation_price)
        # ^^^^^^^^^^^^^^^^^^^^ END OF MODIFIED SECTION ^^^^^^^^^^^^^^^^^^^^

        portfolio_value = portfolio.get('total_value', 0)
        max_pos_size_limit = self.risk_limits.get("max_position_size", 0.1)

        if portfolio_value > 0 and order_value > portfolio_value * max_pos_size_limit:
            reason_msg = translator.t('error_order_exceeds_pos_limit',
                                      fallback="订单价值 {order_val:.2f} 超过了单一持仓限制 ({limit_pct:.0f}%)").format(
                order_val=order_value, limit_pct=max_pos_size_limit * 100)
            logger.warning(f"Order validation failed for {symbol}: {reason_msg}")
            return {'valid': False, 'reason': reason_msg}

        if direction == 'Buy' and order_value > portfolio.get('cash', 0):
            reason_msg = translator.t('error_insufficient_funds',
                                      fallback="资金不足! 需要 ${needed:.2f}, 可用 ${available:.2f}").format(
                needed=order_value, available=portfolio.get('cash', 0))
            logger.warning(f"Order validation failed for {symbol} (Buy): {reason_msg}")
            return {'valid': False, 'reason': reason_msg}

        if direction == 'Sell':
            position = portfolio.get('positions', {}).get(symbol, {})
            current_quantity = position.get('quantity', 0)
            if quantity > current_quantity:
                reason_msg = translator.t('error_insufficient_position',
                                          fallback="持仓不足! 需要 {needed} 股, 持有 {available} 股").format(
                    needed=quantity, available=current_quantity)
                logger.warning(f"Order validation failed for {symbol} (Sell): {reason_msg}")
                return {'valid': False, 'reason': reason_msg}

        logger.info(f"Order for {symbol} passed risk validation.")
        return {'valid': True}

    def get_current_market_regime(self, system_ref: Any, market_benchmark: str = "SPY") -> Dict:
        """
        [最终健壮版] 使用 VIX 和市场基准判断市场状态。
        如果无法获取真实数据，则返回一个安全的中性默认状态。
        """
        # --- 默认返回值 (安全的、中性的) ---
        default_regime = {
            "status": "NEUTRAL",
            "volatility": "NORMAL",
            "trend": "SIDEWAYS",
            "vix_value": 22.1,  # 使用 VIX 的长期平均值作为默认
            "reason": translator.t('regime_reason_default', fallback="无法获取实时市场数据，使用默认中性状态。")
        }

        try:
            # --- 1. 尝试获取真实数据 ---
            logger.debug("Regime analysis: Fetching VIX and benchmark data...")
            # DataManager 的 get_index_data 内部已经有缓存
            vix_data = system_ref.data_manager.get_index_data("^VIX", days=5)
            benchmark_data = system_ref.data_manager.get_index_data(market_benchmark, days=300)

            # --- 2. 严格的数据验证 ---
            if vix_data is None or vix_data.empty or 'close' not in vix_data.columns or vix_data['close'].iloc[
                -1] is None or pd.isna(vix_data['close'].iloc[-1]):
                logger.warning("VIX data is invalid or missing. Falling back to default regime.")
                return default_regime  # VIX 无效，直接返回默认值

            latest_vix = vix_data['close'].iloc[-1]

            # --- 3. 判断波动率状态 (总可以进行) ---
            volatility_status = "NORMAL"
            if latest_vix > self.risk_limits.get("vix_high_threshold", 25):
                volatility_status = "HIGH"
            elif latest_vix < self.risk_limits.get("vix_low_threshold", 15):
                volatility_status = "LOW"

            # --- 4. 判断趋势状态 (如果数据足够) ---
            trend_status = "SIDEWAYS"  # 默认趋势为横盘
            if benchmark_data is not None and not benchmark_data.empty and 'close' in benchmark_data.columns and len(
                    benchmark_data) >= 200:
                latest_benchmark_price = benchmark_data['close'].iloc[-1]
                ma50 = benchmark_data['close'].rolling(window=50).mean().iloc[-1]
                ma200 = benchmark_data['close'].rolling(window=200).mean().iloc[-1]

                if not any(pd.isna([latest_benchmark_price, ma50, ma200])):
                    if latest_benchmark_price > ma50 and ma50 > ma200:
                        trend_status = "BULL"
                    elif latest_benchmark_price < ma50 and ma50 < ma200:
                        trend_status = "BEAR"
            else:
                logger.warning(
                    f"Benchmark data for trend analysis is insufficient or invalid. Defaulting trend to SIDEWAYS.")

            # --- 5. 组合最终的真实状态 ---
            return {
                "status": f"{volatility_status}_{trend_status}",
                "volatility": volatility_status,
                "trend": trend_status,
                "vix_value": round(latest_vix, 2),
                "reason": f"VIX: {latest_vix:.2f}. Trend analysis based on {len(benchmark_data) if benchmark_data is not None else 0} days of data."
            }

        except Exception as e:
            logger.error(f"判断市场状态时发生未知错误: {e}", exc_info=True)
            # 发生任何未知错误，都安全地返回默认值
            return default_regime

    def get_dynamic_risk_suggestions(self, current_regime_status: str) -> Dict[str, Any]:
        """
        [新版] 根据当前市场状态，生成动态的风险参数调整建议。
        """
        suggestions = {
            "trade_quantity_multiplier": 1.0,
            "alpha_threshold_multiplier": 1.0,
            "suggestion_reason": translator.t('regime_suggestion_neutral', fallback="市场状态中性，建议维持标准参数。")
        }
        if not current_regime_status: return suggestions

        if "HIGH" in current_regime_status and "BEAR" in current_regime_status:
            suggestions.update({
                "trade_quantity_multiplier": 0.5, "alpha_threshold_multiplier": 1.5,
                "suggestion_reason": translator.t('regime_suggestion_high_vol_bear',
                                                  fallback="高波动熊市：建议降低仓位，提高信号阈值。")
            })
        elif "HIGH" in current_regime_status:
            suggestions.update({
                "trade_quantity_multiplier": 0.7, "alpha_threshold_multiplier": 1.2,
                "suggestion_reason": translator.t('regime_suggestion_high_vol',
                                                  fallback="高波动市场：建议减小仓位，提高信号阈值。")
            })
        elif "LOW" in current_regime_status and "BULL" in current_regime_status:
            suggestions.update({
                "trade_quantity_multiplier": 1.2, "alpha_threshold_multiplier": 0.8,
                "suggestion_reason": translator.t('regime_suggestion_low_vol_bull',
                                                  fallback="低波动牛市：可适度放大仓位，放宽信号阈值。")
            })

        return suggestions

    # ==========================================================================
    # SECTION: 投资组合级风险分析 (Portfolio Oversight)
    # ==========================================================================
    def analyze_portfolio_risk(self, portfolio: Dict, historical_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        [COMPREHENSIVE FIX] Analyzes portfolio risk, including VaR and CVaR.
        Requires the complete portfolio state and historical data for all positions.
        """
        # --- 0. Initialize a robust results dictionary ---
        results = {
            "suggestions": ["无法进行风险分析：数据不足。"],
            "diversification": {"max_single_exposure": 0.0, "concentration_score": 0},
            "risk_metrics": {"cash_ratio": 1.0, "diversification_score": 0},
            "var_cvar": {
                "var_1day_pct": "N/A",
                "cvar_1day_pct": "N/A",
                "confidence_level": f"{self.risk_limits.get('var_confidence', 0.95):.0%}"
            }
        }

        positions = portfolio.get('positions', {})
        total_value = portfolio.get('total_value', 0)
        cash_value = portfolio.get('cash', 0)

        # --- 1. Basic validation ---
        if total_value == 0:
            return results  # Return default values if portfolio is empty
        if not positions or not historical_data:
            results["suggestions"] = ["投资组合为空或缺少历史数据，无法进行深度风险分析。"]
            results["risk_metrics"]["cash_ratio"] = 1.0
            return results

        # --- 2. Calculate Portfolio Returns DataFrame ---
        # This is the foundational data for VaR/CVaR
        returns_df = pd.DataFrame()
        weights = []

        # Use only symbols that have both position data AND historical data
        valid_symbols = [s for s in positions.keys() if s in historical_data and not historical_data[s].empty]
        if not valid_symbols:
            results["suggestions"] = ["缺少有效持仓的历史数据，无法计算VaR/CVaR。"]
            # We can still calculate some metrics
            # (Code to calculate diversification even without returns will be here)
            return results

        # Calculate the total value of only the valid positions for weighting
        valid_position_value = sum(
            positions[s].get('quantity', 0) * positions[s].get('current_price', 0) for s in valid_symbols)

        if valid_position_value == 0:
            results["suggestions"] = ["有效持仓的总市值为零，无法计算VaR/CVaR。"]
            return results

        for symbol in valid_symbols:
            position_value = positions[symbol].get('quantity', 0) * positions[symbol].get('current_price', 0)
            weights.append(position_value / valid_position_value)  # Weight relative to other stocks
            returns_df[symbol] = historical_data[symbol]['close'].pct_change()

        returns_df.dropna(inplace=True)
        if returns_df.empty:
            results["suggestions"] = ["计算每日收益率后数据为空，无法计算VaR/CVaR。"]
            # (Diversification metrics can still be calculated below)

        # --- 3. Calculate VaR and CVaR (if we have returns data) ---
        if not returns_df.empty:
            portfolio_returns = returns_df.dot(np.array(weights))
            confidence_level = self.risk_limits.get('var_confidence', 0.95)

            # VaR: The "worst expected loss" at a given confidence level.
            var_1day = portfolio_returns.quantile(1 - confidence_level)

            # CVaR: The "expected loss given that the loss is greater than the VaR". A better measure of tail risk.
            cvar_1day = portfolio_returns[portfolio_returns <= var_1day].mean()

            results["var_cvar"] = {
                "var_1day_pct": f"{abs(var_1day):.2%}",
                "cvar_1day_pct": f"{abs(cvar_1day):.2%}",  # CVaR is now calculated
                "confidence_level": f"{confidence_level:.0%}"
            }
            # Add suggestion based on CVaR
            max_cvar_limit = self.risk_limits.get("max_cvar_pct", 0.05)
            if abs(cvar_1day) > max_cvar_limit:
                results["suggestions"] = [
                    f"警告：投资组合的条件风险价值(CVaR)为 {abs(cvar_1day):.2%}，超过了 {max_cvar_limit:.2%} 的限制。这意味着在极端亏损情况下，平均亏损可能过高。"]
            else:
                results["suggestions"] = [f"投资组合的尾部风险 (CVaR: {abs(cvar_1day):.2%}) 在可接受范围内。"]

        # --- 4. Calculate Diversification and General Metrics (always possible if there are positions) ---
        all_position_values = {s: p.get('quantity', 0) * p.get('current_price', 0) for s, p in positions.items()}
        all_weights = [v / total_value for v in all_position_values.values()]

        max_single_exposure = max(all_weights) if all_weights else 0.0

        sector_exposure = {}
        for symbol, pos_value in all_position_values.items():
            sector = self._get_symbol_sector(symbol)
            sector_exposure.setdefault(sector, 0)
            sector_exposure[sector] += pos_value / total_value

        results["diversification"] = {
            "max_single_exposure": max_single_exposure,
            "concentration_score": self._calculate_concentration(sector_exposure)
        }

        results["risk_metrics"] = {
            "cash_ratio": cash_value / total_value,
            "diversification_score": self._calculate_diversification_score(all_weights)
        }

        # --- 5. Generate final suggestions by combining them ---
        # The CVaR suggestion is generated first, other suggestions can be appended.
        other_suggestions = self._generate_optimization_suggestions(results)
        # Ensure we don't have duplicate default messages
        if results["suggestions"] and "无法" in results["suggestions"][0]:
            results["suggestions"] = other_suggestions
        else:
            results["suggestions"].extend(other_suggestions)

        return results

    def _get_symbol_sector(self, symbol):
        """获取股票行业分类"""
        sectors = {
            "AAPL": "科技",
            "MSFT": "科技",
            "GOOGL": "科技",
            "AMZN": "消费",
            "META": "通信",
            "TSLA": "汽车",
            "JPM": "金融",
            "JNJ": "医疗",
            "PG": "日用品",
            "XOM": "能源",
            "600519.SH": "白酒",
            "000001.SZ": "金融",
            "600036.SH": "金融",
            "601318.SH": "保险"
        }
        return sectors.get(symbol, "其他")

    def _calculate_diversification_score(self, weights):
        """计算投资组合多样化得分 (0-100)"""
        if not weights:
            return 0

        # 计算有效持仓数量 (考虑权重)
        n = len(weights)
        if n <= 1:
            return 0

        # 理想情况下每个资产权重相等
        ideal_weight = 1.0 / n

        # 计算权重偏离度
        weight_deviation = sum([abs(w - ideal_weight) for w in weights]) / (2 * (1 - ideal_weight))

        # 转换为0-100分，偏离度越低分数越高
        return int((1 - weight_deviation) * 100)

    def _calculate_concentration(self, sector_exposure):
        """计算行业集中度 (0-100)"""
        if not sector_exposure:
            return 0

        # 赫芬达尔-赫希曼指数 (HHI)
        hhi = sum([weight ** 2 for weight in sector_exposure.values()])

        # 归一化到0-100，值越高表示集中度越高
        return min(100, int(hhi * 100))

    def _calculate_var(self, symbols, weights, historical_data, days=10, confidence=0.95):
        """计算风险价值 (VaR)"""
        try:
            # 准备收益率数据
            returns_data = {}
            for symbol in symbols:
                if symbol in historical_data:
                    # 计算每日收益率
                    prices = historical_data[symbol]['close']
                    returns = prices.pct_change().dropna()
                    returns_data[symbol] = returns.values

            if not returns_data:
                return {"daily_var": 0, "var_10day": 0}

            # 创建投资组合的历史收益率
            portfolio_returns = np.zeros(len(next(iter(returns_data.values()))))

            for i, symbol in enumerate(symbols):
                if symbol in returns_data:
                    portfolio_returns += weights[i] * returns_data[symbol]

            # 计算日VaR (历史模拟法)
            sorted_returns = np.sort(portfolio_returns)
            var_index = int((1 - confidence) * len(sorted_returns))
            daily_var = abs(sorted_returns[var_index])

            # 扩展到多日VaR
            var_nday = daily_var * np.sqrt(days)

            return {
                "daily_var": round(daily_var * 100, 2),  # 百分比形式
                f"var_{days}day": round(var_nday * 100, 2)
            }

        except Exception as e:
            logger.error(f"计算VaR失败: {e}")
            return {"daily_var": 0, "var_10day": 0}

    def _generate_optimization_suggestions(self, risk_analysis):
        """生成投资组合优化建议"""
        suggestions = []

        # 现金比例建议
        cash_ratio = risk_analysis.get("risk_metrics", {}).get("cash_ratio", 0)
        if cash_ratio > 0.4:
            suggestions.append("现金比例过高，可能错过投资机会，建议适度增加仓位")
        elif cash_ratio < 0.1:
            suggestions.append("现金比例过低，流动性风险较高，建议保留部分现金以应对突发需求")

        # 多样化建议
        diversification = risk_analysis.get("diversification", {})
        max_exposure = diversification.get("max_single_exposure", 0)
        concentration = diversification.get("concentration_score", 0)

        if max_exposure > self.risk_limits.get("max_position_size", 0.1):
            suggestions.append(f"单一持仓比例过高 ({max_exposure:.1%})，建议降低单一资产暴露风险")

        if concentration > 70:
            suggestions.append("行业集中度过高，建议分散投资不同行业以降低系统性风险")

        # 基于VaR的建议
        var = risk_analysis.get("var", {})
        var_10day = var.get("var_10day", 0)

        if var_10day > 15:  # 如果10天VaR超过15%
            suggestions.append(f"投资组合波动性较高 (10日VaR: {var_10day}%)，建议调整持仓降低风险")

        # 如果没有任何建议，添加一个积极的反馈
        if not suggestions:
            suggestions.append("投资组合风险分布合理，保持当前配置")

        return suggestions

    def optimize_portfolio(self, symbols, historical_data, risk_preference="balanced"):
        """投资组合优化"""
        try:
            # 准备收益率数据
            returns_df = pd.DataFrame()

            for symbol in symbols:
                if symbol in historical_data:
                    # 计算每日收益率
                    prices = historical_data[symbol]['close']
                    returns = prices.pct_change().dropna()
                    returns_df[symbol] = returns

            if returns_df.empty:
                return {"error": "无足够历史数据进行优化"}

            # 计算平均收益和协方差
            mean_returns = returns_df.mean()
            cov_matrix = returns_df.cov()

            # 根据风险偏好调整
            risk_weights = {
                "conservative": {"return": 0.2, "risk": 0.8},
                "balanced": {"return": 0.5, "risk": 0.5},
                "aggressive": {"return": 0.8, "risk": 0.2}
            }

            weights = self._calculate_optimal_weights(
                mean_returns,
                cov_matrix,
                risk_preference=risk_weights.get(risk_preference, risk_weights["balanced"])
            )

            # 计算优化后的预期回报和风险
            portfolio_return = np.sum(mean_returns * weights) * 252  # 年化
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

            # 整理结果
            allocation = {}
            for i, symbol in enumerate(returns_df.columns):
                if weights[i] > 0.01:  # 只显示权重大于1%的
                    allocation[symbol] = round(weights[i] * 100, 2)

            return {
                "allocation": allocation,
                "expected_annual_return": round(portfolio_return * 100, 2),
                "expected_volatility": round(portfolio_std * 100, 2),
                "sharpe_ratio": round(portfolio_return / portfolio_std, 2) if portfolio_std > 0 else 0
            }

        except Exception as e:
            logger.error(f"投资组合优化失败: {e}")
            return {"error": f"优化失败: {str(e)}"}

    def _calculate_optimal_weights(self, mean_returns, cov_matrix, risk_preference):
        """计算最优权重 (使用简化的优化方法)"""
        n = len(mean_returns)

        # 生成多组随机权重
        num_portfolios = 10000
        results = np.zeros((3, num_portfolios))
        weights_record = np.zeros((num_portfolios, n))

        for i in range(num_portfolios):
            weights = np.random.random(n)
            weights /= np.sum(weights)
            weights_record[i] = weights

            # 计算回报和风险
            portfolio_return = np.sum(mean_returns * weights) * 252
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

            # 效用函数 = 回报权重*回报 - 风险权重*风险
            utility = (risk_preference["return"] * portfolio_return) - (risk_preference["risk"] * portfolio_std)

            results[0, i] = portfolio_return
            results[1, i] = portfolio_std
            results[2, i] = utility

        # 找到效用最大的组合
        max_utility_idx = np.argmax(results[2])

        return weights_record[max_utility_idx]