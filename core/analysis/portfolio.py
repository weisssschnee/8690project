# core/analysis/portfolio.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import logging
from typing import Dict, List, Any, Tuple
from core.translate import translator

logger = logging.getLogger(__name__)


class PortfolioAnalyzer:
    """
    负责计算所有投资组合相关的性能、风险指标，并渲染分析UI。
    """

    def __init__(self, system: Any):
        self.system = system
        # self.config 可以在需要时通过 self.system.config 获取

    # --------------------------------------------------------------------------
    # 核心分析方法 (数据计算层)
    # --------------------------------------------------------------------------

    def analyze_portfolio_performance(self, portfolio: Dict, trades: List[Dict], history: List[Dict]) -> Dict:
        """
        [核心] 分析投资组合表现的主函数，计算所有指标并返回一个包含所有结果的字典。
        """
        # --- 1. 概览指标 ---
        total_value = portfolio.get('total_value', 0)
        cash = portfolio.get('cash', 0)
        invested_value = total_value - cash
        positions = portfolio.get('positions', {})
        position_count = len(positions)

        # --- 2. 交易摘要 ---
        trading_summary = self._calculate_trading_summary(trades)

        # --- 3. 个股表现 ---
        stock_perf_df = self._calculate_individual_stock_performance(portfolio, trades)

        # --- 4. 收益与风险指标 (依赖历史数据) ---
        returns_analysis = self._calculate_returns_and_risk(history)

        # --- 5. 资产配置 ---
        allocation_analysis = self._calculate_asset_allocation(portfolio)

        # --- 汇总所有结果 ---
        return {
            "overview": {
                "total_value": total_value, "cash": cash,
                "invested_value": invested_value, "position_count": position_count,
            },
            "stock_performance": stock_perf_df,
            "trading_summary": trading_summary,
            "returns_analysis": returns_analysis,
            "asset_allocation": allocation_analysis,
        }

    def _calculate_trading_summary(self, trades: List[Dict]) -> Dict:
        """[修复版] 计算交易摘要，修复统计口径和胜率计算。"""
        if not trades:
            return {'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'win_rate': 0.0}

        total_trades = len(trades)
        profits, losses = [], []

        buys = sorted([t for t in trades if t.get('direction', '').lower() == 'buy'], key=lambda x: x.get('timestamp'))
        sells = sorted([t for t in trades if t.get('direction', '').lower() == 'sell'],
                       key=lambda x: x.get('timestamp'))

        # 为每个股票创建一个买入队列 (FIFO)
        buy_queues = {t['symbol']: [] for t in buys}
        for t in buys:
            buy_queues[t['symbol']].append(t)

        for sell_trade in sells:
            symbol = sell_trade['symbol']
            sell_qty_rem = sell_trade.get('quantity', 0)
            sell_price = sell_trade.get('price', 0)

            while sell_qty_rem > 0 and buy_queues.get(symbol):
                buy_trade = buy_queues[symbol][0]
                buy_price = buy_trade.get('price', 0)
                buy_qty_avail = buy_trade.get('quantity', 0)

                matched_qty = min(sell_qty_rem, buy_qty_avail)
                pnl = (sell_price - buy_price) * matched_qty

                if pnl > 0:
                    profits.append(pnl)
                else:
                    losses.append(pnl)

                sell_qty_rem -= matched_qty
                buy_trade['quantity'] -= matched_qty  # 修改字典中的值来标记已匹配

                if buy_trade['quantity'] <= 1e-6:
                    buy_queues[symbol].pop(0)

        win_trades_count = len(profits)
        loss_trades_count = len(losses)
        total_closed_trades = win_trades_count + loss_trades_count
        win_rate = (win_trades_count / total_closed_trades * 100) if total_closed_trades > 0 else 0.0

        return {
            'total_trades': total_trades,
            'winning_trades': win_trades_count,
            'losing_trades': loss_trades_count,
            'win_rate': win_rate,
        }

    def _calculate_individual_stock_performance(self, portfolio: Dict, trades: List[Dict]) -> pd.DataFrame:
        """[修复版] 计算个股表现，确保统计口径一致。"""
        positions = portfolio.get('positions', {})
        if not positions: return pd.DataFrame()

        perf_data = []
        total_value = portfolio.get('total_value', 1)

        for symbol, pos in positions.items():
            symbol_trades = [t for t in trades if t.get('symbol') == symbol]
            summary = self._calculate_trading_summary(symbol_trades)
            market_value = pos.get('quantity', 0) * pos.get('current_price', 0)
            cost_value = pos.get('quantity', 0) * pos.get('cost_basis', 0)

            perf_data.append({
                'Stock': symbol,
                'Quantity': pos.get('quantity', 0),
                'Current Price': pos.get('current_price', 0),
                'Cost Basis': pos.get('cost_basis', 0),
                'Market Value': market_value,
                'Weight': (market_value / total_value * 100) if total_value > 0 else 0,
                'P/L': market_value - cost_value,
                'P/L %': ((market_value / cost_value) - 1) * 100 if cost_value > 0 else 0,
                'Trades': summary['total_trades'],
                'Win Rate': summary['win_rate'],
            })

        return pd.DataFrame(perf_data)

    def _calculate_returns_and_risk(self, history: List[Dict]) -> Dict:
        """计算收益和风险指标。"""
        if not history or len(history) < 2:
            return {"dates": [], "returns": [], "cumulative": []}  # 返回空结构

        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()

        # 确保 total_value 列存在
        if 'total_value' not in df.columns: return {"dates": [], "returns": [], "cumulative": []}

        # 计算日收益率
        daily_returns = df['total_value'].resample('D').last().pct_change().dropna()
        if daily_returns.empty:
            return {"dates": [], "returns": [], "cumulative": []}

        cumulative_returns = (1 + daily_returns).cumprod() - 1

        # 风险指标
        annualized_return = (1 + cumulative_returns.iloc[-1]) ** (252 / len(daily_returns)) - 1 if len(
            daily_returns) > 0 else 0
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        risk_free_rate = getattr(self.system.config, 'RISK_FREE_RATE', 0.03)
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0

        # 最大回撤
        net_values = (1 + cumulative_returns).fillna(1)
        running_max = net_values.cummax()
        drawdowns = (net_values - running_max) / running_max
        max_drawdown = drawdowns.min()

        return {
            "dates": daily_returns.index.tolist(),
            "returns": daily_returns.tolist(),
            "cumulative": cumulative_returns.tolist(),
            "annualized_return": annualized_return * 100,
            "annualized_volatility": annualized_volatility * 100,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown * 100,
            "drawdowns": drawdowns.tolist()
        }

    def _calculate_asset_allocation(self, portfolio: Dict) -> Dict:
        """计算资产配置。"""
        positions = portfolio.get('positions', {})
        total_value = portfolio.get('total_value', 1)
        cash = portfolio.get('cash', 0)

        by_value = {'Cash': cash / total_value} if total_value > 0 else {'Cash': 1.0}
        for symbol, pos in positions.items():
            market_value = pos.get('quantity', 0) * pos.get('current_price', 0)
            by_value[symbol] = market_value / total_value if total_value > 0 else 0

        return {"by_value": by_value}

    # --------------------------------------------------------------------------
    # UI 渲染方法 (UI 渲染层)
    # --------------------------------------------------------------------------

    def render_portfolio_analysis_ui(self, portfolio: Dict, trades: List[Dict], history: List[Dict]):
        """渲染组合分析UI的主入口。"""
        analysis_results = self.analyze_portfolio_performance(portfolio, trades, history)

        tab_keys = ['overview', 'returns_performance', 'risk_analysis', 'trade_records', 'asset_allocation']
        tab_defaults = ["Overview", "Returns Performance", "Risk Analysis", "Trade Records", "Asset Allocation"]
        tab_labels = [translator.t(key, fallback=default) for key, default in zip(tab_keys, tab_defaults)]

        try:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_labels)
            with tab1:
                self._render_overview_tab(analysis_results)
            with tab2:
                self._render_returns_tab(analysis_results)
            with tab3:
                self._render_risk_tab(analysis_results)
            with tab4:
                self._render_trades_tab(trades)  # 交易记录直接使用原始trades
            with tab5:
                self._render_allocation_tab(analysis_results)
        except Exception as e:
            logger.error(f"Error rendering analysis tabs: {e}", exc_info=True)
            st.error("Failed to render analysis tabs.")

    def _render_overview_tab(self, analysis: Dict):
        """渲染概览标签页。"""
        st.subheader(translator.t('portfolio_overview_subheader', fallback="Portfolio Overview"))

        overview = analysis.get("overview", {})
        stock_perf_df = analysis.get("stock_performance", pd.DataFrame())
        trading_summary = analysis.get("trading_summary", {})

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Asset Value", f"${overview.get('total_value', 0):,.2f}")
        col2.metric("Cash", f"${overview.get('cash', 0):,.2f}",
                    f"{overview.get('cash', 0) / overview.get('total_value', 1):.2%}")
        col3.metric("Invested Value", f"${overview.get('invested_value', 0):,.2f}")
        col4.metric("Position Count", f"{overview.get('position_count', 0)}")

        st.subheader(translator.t('individual_stock_performance_subheader', fallback="Individual Stock Performance"))
        if not stock_perf_df.empty:
            display_df = stock_perf_df.copy()
            for col in ['Current Price', 'Cost Basis', 'Market Value', 'P/L']:
                display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")
            for col in ['Weight', 'P/L %', 'Win Rate']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info(translator.t('no_positions_for_analysis', fallback="No positions to analyze."))

        st.subheader(translator.t('trading_summary_subheader', fallback="Trading Summary"))
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        s_col1.metric("Total Trades", f"{trading_summary.get('total_trades', 0)}")
        s_col2.metric("Winning Trades", f"{trading_summary.get('winning_trades', 0)}")
        s_col3.metric("Losing Trades", f"{trading_summary.get('losing_trades', 0)}")
        s_col4.metric("Win Rate", f"{trading_summary.get('win_rate', 0):.2f}%")

    def _render_returns_tab(self, analysis: Dict):
        """渲染收益表现标签页。"""
        st.subheader(translator.t('returns_analysis_subheader', fallback="Returns Performance"))
        returns_analysis = analysis.get("returns_analysis", {})

        if not returns_analysis or not returns_analysis.get("dates"):
            st.info(translator.t('insufficient_data_for_returns', fallback="Insufficient data for returns analysis."));
            return

        r_col1, r_col2, r_col3 = st.columns(3)
        r_col1.metric("Annualized Return", f"{returns_analysis.get('annualized_return', 0):.2f}%")
        r_col2.metric("Annualized Volatility", f"{returns_analysis.get('annualized_volatility', 0):.2f}%")
        r_col3.metric("Sharpe Ratio", f"{returns_analysis.get('sharpe_ratio', 0):.2f}")

        st.subheader(translator.t('portfolio_equity_curve_subheader', fallback="Portfolio Equity Curve"))
        fig_equity = go.Figure(go.Scatter(
            x=returns_analysis['dates'],
            y=(1 + pd.Series(returns_analysis['cumulative'])).fillna(1),
            mode='lines', name='Equity'
        ))
        fig_equity.update_layout(title="Portfolio Value Over Time", height=400)
        st.plotly_chart(fig_equity, use_container_width=True)

    def _render_risk_tab(self, analysis: Dict):
        """渲染风险分析标签页。"""
        st.subheader(translator.t('risk_analysis_subheader', fallback="Risk Analysis"))
        returns_analysis = analysis.get("returns_analysis", {})

        if not returns_analysis or not returns_analysis.get("dates"):
            st.info(translator.t('insufficient_data_for_risk', fallback="Insufficient data for risk analysis."));
            return

        st.metric("Max Drawdown", f"{returns_analysis.get('max_drawdown', 0):.2f}%")

        st.subheader(translator.t('drawdown_analysis_subheader', fallback="Drawdown Analysis"))
        fig_dd = go.Figure(go.Scatter(
            x=returns_analysis['dates'],
            y=pd.Series(returns_analysis['drawdowns']) * 100,
            fill='tozeroy', mode='lines', name='Drawdown', line_color='red'
        ))
        fig_dd.update_layout(title="Portfolio Drawdown Over Time", yaxis_title="Drawdown (%)", height=400)
        st.plotly_chart(fig_dd, use_container_width=True)

    def _render_trades_tab(self, trades: List[Dict]):
        """渲染交易记录标签页。"""
        st.subheader(translator.t('trade_records_subheader', fallback="Trade Records"))
        if not trades:
            st.info(translator.t('no_trades_records_available', fallback="No trade records available."))
            return

        trades_df = pd.DataFrame(trades)
        display_df = trades_df[['timestamp', 'symbol', 'direction', 'quantity', 'price', 'total']].copy()
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    def _render_allocation_tab(self, analysis: Dict):
        """渲染资产配置标签页。"""
        st.subheader(translator.t('asset_allocation_analysis_subheader', fallback="Asset Allocation"))
        allocation = analysis.get("asset_allocation", {}).get("by_value", {})

        if not allocation or len(allocation) <= 1 and 'Cash' in allocation:  # Only cash
            st.info("No invested assets to analyze allocation.")
            return

        labels = list(allocation.keys())
        values = list(allocation.values())

        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
        fig_pie.update_layout(title="Portfolio Asset Allocation", height=400)
        st.plotly_chart(fig_pie, use_container_width=True)