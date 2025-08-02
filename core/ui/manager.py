# core/ui/manager.py
import streamlit as st
import time
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from core.translate import translator  # 确保 translator 实例已正确初始化
import traceback  # 确保 traceback 已导入
from typing import Dict, Any, Optional  # 添加类型提示
import numpy as np
from core.ui.autotrader_tab import AutoTraderTab

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.system import TradingSystem

logger = logging.getLogger(__name__)


class UIManager:
    """UI管理器类"""

    def __init__(self):
        """初始化UI管理器，但不在这里实例化依赖system的组件。"""
        # 可以在这里初始化不依赖 system 的属性
        self.autotrader_tab_renderer: AutoTraderTab | None = None
    def _render_refresh_controls_in_sidebar(self):
        """[重构] 专门在侧边栏内部渲染刷新控件。"""
        with st.expander(translator.t('refresh_settings_expander', fallback="刷新设置"), expanded=False):
            auto_refresh_on = st.session_state.get('auto_refresh', False)

            st.session_state.auto_refresh = st.toggle(
                translator.t('enable_auto_refresh'),
                value=auto_refresh_on,
                key="sidebar_toggle_auto_refresh"
            )

            if st.session_state.auto_refresh:
                refresh_interval = st.session_state.get('refresh_interval', 60)
                st.session_state.refresh_interval = st.slider(
                    translator.t('refresh_interval_seconds'), min_value=10, max_value=300,
                    value=refresh_interval, step=10, key="sidebar_slider_refresh"
                )
                last_refresh = st.session_state.get('last_refresh_time', time.time())
                next_refresh_in = max(0, (last_refresh + st.session_state.refresh_interval) - time.time())
                st.caption(f"{translator.t('next_refresh_in')} {int(next_refresh_in)} {translator.t('seconds_suffix')}")

            if st.button(translator.t('refresh_now'), use_container_width=True, key="sidebar_btn_refresh_now"):
                st.session_state.last_refresh_time = time.time()
                st.rerun()

    def render_sidebar(self, system: Any):
        """
        [最终修复版 v2] 侧边栏渲染，修复了重复调用和渲染错误。
        """
        logger.debug("Rendering sidebar...")
        try:
            with st.sidebar:
                st.title("智能量化交易系统")
                st.caption(f"{translator.t('current_time')}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # --- 界面设置 ---
                st.header(translator.t('interface_settings_header', fallback="界面设置"))
                translator.add_switcher(location=st.sidebar)  # 正确使用 st.sidebar
                self._render_refresh_controls_in_sidebar()  # 只在这里调用一次

                # --- 用户账户 ---
                st.header(translator.t('user_account_header', fallback="用户账户"))
                if st.session_state.get('logged_in'):
                    st.info(
                        f"👤 {translator.t('welcome_user', fallback='欢迎')}, **{st.session_state.get('username', 'User')}**!")
                    if st.button(translator.t('logout_button', fallback="登出"), use_container_width=True):
                        system.logout_user()
                        st.toast(translator.t('logout_success'))
                        st.rerun()
                else:
                    with st.expander(translator.t('login_expander', fallback="登录以加载您的账户")):
                        with st.form("sidebar_login_form"):
                            username = st.text_input(translator.t('username_label'))
                            password = st.text_input(translator.t('password_label'), type="password")
                            if st.form_submit_button(translator.t('login_button')):
                                if system.login_user(username, password):
                                    st.toast(translator.t('login_success'))
                                    st.rerun()
                                else:
                                    st.error(translator.t('login_failed'))

                # --- 账户概览 ---
                st.header(translator.t('account_overview', fallback="账户概览"))
                portfolio = st.session_state.get('portfolio', {})
                cash = portfolio.get('cash', 0)
                positions_value = sum(pos.get('quantity', 0) * pos.get('current_price', 0) for pos in
                                      portfolio.get('positions', {}).values())
                total_value = cash + positions_value
                st.metric(translator.t('total_assets'), f"${total_value:,.2f}")
                st.metric(translator.t('investment_amount'), f"${positions_value:,.2f}")
                st.metric(translator.t('available_funds'), f"${cash:,.2f}")

                # --- 自动交易引擎 ---
                if hasattr(system, 'render_autotrader_controls_in_sidebar'):
                    system.render_autotrader_controls_in_sidebar()

                # --- 系统菜单 ---
                st.header(translator.t('system_menu', fallback="系统菜单"))
                with st.container(border=True):
                    if st.button(translator.t('reset_account', fallback="重置模拟账户"), use_container_width=True):
                        if st.session_state.get('logged_in'):
                            system.reset_user_account()
                            st.toast(translator.t('user_account_reset_success'))
                        else:
                            system.reset_to_default_portfolio()
                            st.toast(translator.t('guest_account_reset_success'))
                        st.rerun()

                    if st.button(translator.t('export_trades_btn_label'), use_container_width=True):
                        self._export_trade_data()

        except Exception as e:
            logger.error(f"Error rendering sidebar: {e}", exc_info=True)
            # 外部的 try...except 仍然会捕获并显示这个错误
            st.sidebar.error(translator.t('error_rendering_sidebar', fallback="侧边栏渲染出错"))

    def _export_trade_data(self):
        """导出交易数据为CSV"""
        trades = st.session_state.get('trades', [])
        if not trades:
            st.warning(translator.t('no_trades_to_export', fallback="没有交易数据可导出"))
            return
        try:
            trades_data = []
            for trade in trades:
                direction_key = str(trade.get('direction', 'N/A')).lower()
                trades_data.append({
                    'timestamp': trade.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
                    'symbol': trade.get('symbol', 'N/A'),
                    'direction': translator.t(direction_key, fallback=trade.get('direction', 'N/A')),
                    'quantity': trade.get('quantity', 0),
                    'price': f"{trade.get('price', 0):.2f}",
                    'total': f"{trade.get('total', 0):.2f}",
                    'commission': f"{trade.get('commission', 0):.4f}",  # Include commission if available
                    'is_mock': trade.get('is_mock', False)
                })
            df = pd.DataFrame(trades_data)
            csv = df.to_csv(index=False, encoding='utf-8-sig')
            download_label = translator.t('download_csv_label', fallback="下载CSV文件")
            file_prefix = translator.t('trade_data_filename_prefix', fallback="trade_data")
            file_name = f"{file_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            st.download_button(label=download_label, data=csv, file_name=file_name, mime="text/csv",
                               key="download_trades_csv_btn")
            # st.info(translator.t('csv_ready_for_download', fallback="CSV文件已准备好下载。")) # Maybe remove this info as button is there
        except Exception as e:
            logger.error(f"导出交易数据时出错: {e}", exc_info=True)
            st.error(translator.t('export_error', fallback="导出数据时出错:") + f" {e}")

    def render_main_tabs(self, system):
        """[修改版] 渲染主界面选项卡，并加入新的“自动化”标签页。"""
        logger.debug("Rendering main tabs...")
        if self.autotrader_tab_renderer is None:
            logger.info("Initializing AutoTraderTab renderer for the first time.")
            self.autotrader_tab_renderer = AutoTraderTab(system)
        # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
        tab_keys = ["dashboard", "market", "trade", "strategy", "autotrader", "portfolio", "analysis", "settings",
                    "alerts"]
        tab_defaults = ["仪表盘", "市场", "交易", "策略分析", "自动化", "投资组合", "业绩分析", "设置", "报警系统"]
        tab_labels = [translator.t(key, fallback=default) for key, default in zip(tab_keys, tab_defaults)]
        try:
            tabs = st.tabs(tab_labels)
        except Exception as e:
            st.error(f"创建主选项卡时出错: {e}")
            st.code(traceback.format_exc())
            logger.error("创建主选项卡时出错", exc_info=True)
            return

        tab_render_functions = [
            self.render_dashboard_tab,
            self.render_market_tab,
            self.render_trading_tab,
            self.render_strategy_tab,
            self.autotrader_tab_renderer.render,
            self.render_portfolio_tab,
            self.render_analysis_tab,
            self._render_settings_tab,
            self.render_alert_tab
        ]

        for i, (tab, func) in enumerate(zip(tabs, tab_render_functions)):
            with tab:
                try:
                    # 获取函数需要的参数数量
                    import inspect
                    sig = inspect.signature(func)

                    if len(sig.parameters) > 0:  # 如果函数需要参数 (比如 render_dashboard_tab(self, system))
                        func(system)
                    else:  # 如果函数不需要参数 (比如 autotrader_tab_renderer.render(self))
                        func()
                except Exception as e:
                    st.error(f"渲染 {tab_labels[i]} 标签页时出错: {e}")
                    logger.error(f"渲染标签页 '{tab_labels[i]}' 时出错", exc_info=True)

    def _render_settings_tab(self, system):
        """内部辅助方法，用于调用 Config 的设置UI渲染 (使用翻译)"""
        logger.debug("Rendering settings tab...")
        # vvvvvvvvvvvvvvvvvvvv START OF MODIFIED SECTION vvvvvvvvvvvvvvvvvvvv
        try:
            # Settings tab header is usually handled by Config.render_settings_ui itself,
            # but if not, you can add it here:
            # st.header(translator.t('settings', fallback="设置"))

            if hasattr(system, 'config') and system.config is not None and \
                    hasattr(system.config, 'render_settings_ui') and \
                    callable(system.config.render_settings_ui):

                system.config.render_settings_ui()  # Assume this method handles its own internal translations

            else:
                # Fallback header if the main settings UI cannot be rendered
                st.header(
                    translator.t('settings', fallback="系统设置"))  # Use a generic key if 'settings' is already for tab
                st.warning(translator.t('settings_ui_unavailable', fallback="设置UI渲染功能未找到。"))
                logger.warning("UIManager: system.config 或其 render_settings_ui 方法不可用。")
        except Exception as e:
            logger.error(f"Error calling config.render_settings_ui or rendering settings tab: {e}", exc_info=True)
            st.error(translator.t('error_rendering_settings', fallback="渲染设置界面时出错。"))
            st.code(traceback.format_exc())

    def _get_standardized_price_data(self, data: Optional[pd.DataFrame]) -> (Optional[pd.DataFrame], Dict[str, str]):
        """标准化股票数据的列名，返回数据和列名映射"""
        logger.debug("Standardizing price data columns...")
        if data is None or data.empty:
            logger.debug("Input data is None or empty, returning None.")
            return None, {}

        columns = data.columns
        col_map = {}
        price_patterns = {
            'close': ['close', 'Close', '收盘', '收盘价', 'adj close', 'Adj Close', '4. close', 'AdjClose'],
            'open': ['open', 'Open', '开盘', '开盘价', '1. open'],
            'high': ['high', 'High', '最高', '最高价', '2. high'],
            'low': ['low', 'Low', '最低', '最低价', '3. low'],
            'volume': ['volume', 'Volume', '成交量', 'vol', '5. volume']
        }

        for std_name, patterns in price_patterns.items():
            found = False
            for pattern in patterns:
                if pattern in columns:
                    col_map[std_name] = pattern
                    found = True;
                    break
            if not found: logger.debug(f"Standard column '{std_name}' not found in data columns: {list(columns)}")

        if 'close' not in col_map: logger.error(
            f"Standardization failed: Could not find 'close' column. Available: {list(columns)}"); return data, {}
        for col in ['open', 'high', 'low']:
            if col not in col_map: logger.warning(f"Standardization warning: Optional column '{col}' missing.")

        logger.debug(f"Standardization map: {col_map}")
        return data, col_map

    def render_dashboard_tab(self, system):
        """渲染仪表盘标签页 (优化热图, 使用翻译)"""
        st.header(translator.t('dashboard', fallback="交易系统仪表盘"))
        logger.debug("Rendering dashboard tab...")
        try:
            # --- Portfolio Overview ---
            col1, col2, col3 = st.columns(3)
            portfolio = st.session_state.get('portfolio', {'cash': 0, 'positions': {}})
            cash = portfolio.get('cash', 0)
            positions_value = 0
            for symbol_pos, position_data in portfolio.get('positions', {}).items():
                quantity = position_data.get('quantity', 0)
                current_price = position_data.get('current_price', 0)
                if isinstance(quantity, (int, float)) and isinstance(current_price, (int, float)):
                    positions_value += quantity * current_price
                else:
                    logger.warning(f"Dashboard: Invalid data for position {symbol_pos} in portfolio overview.")

            calculated_total_value = cash + positions_value

            with col1:
                st.metric(translator.t('total_assets', fallback="账户总值"), f"${calculated_total_value:,.2f}")

            cash_pct = (cash / calculated_total_value) * 100 if calculated_total_value > 0 else 0
            with col2:
                st.metric(translator.t('available_funds', fallback="可用资金"), f"${cash:,.2f}",
                          f"{cash_pct:.1f}%" if calculated_total_value > 0 else None)

            invested_pct = (positions_value / calculated_total_value) * 100 if calculated_total_value > 0 else 0
            with col3:
                st.metric(translator.t('investment_amount', fallback="持仓市值"), f"${positions_value:,.2f}",
                          f"{invested_pct:.1f}%" if calculated_total_value > 0 else None)

            # --- Current Holdings ---
            st.subheader(translator.t('current_holdings', fallback="当前持仓"))
            positions = portfolio.get('positions', {})
            if positions:
                positions_data_list = []
                pos_cols_keys = ['stock_symbol', 'quantity', 'current_price', 'cost_basis', 'market_value',
                                 'profit_loss', 'profit_loss_pct']
                pos_cols_defaults = ["股票", "数量", "现价", "成本价", "市值", "盈亏", "盈亏(%)"]
                pos_cols_display = [translator.t(k, fallback=d) for k, d in zip(pos_cols_keys, pos_cols_defaults)]

                for symbol, position_item in positions.items():
                    quantity = position_item.get('quantity', 0)
                    current_price = position_item.get('current_price', 0)
                    cost_basis = position_item.get('cost_basis', 0)
                    if not all(isinstance(v, (int, float)) for v in [quantity, current_price, cost_basis]):
                        logger.warning(f"Dashboard: Skipping invalid position data for {symbol} in holdings table.")
                        continue
                    current_value = quantity * current_price
                    cost_value = quantity * cost_basis
                    profit_loss = current_value - cost_value
                    profit_loss_pct = (profit_loss / cost_value) * 100 if cost_value != 0 else 0
                    positions_data_list.append({
                        pos_cols_display[0]: symbol, pos_cols_display[1]: quantity,
                        pos_cols_display[2]: f"${current_price:.2f}", pos_cols_display[3]: f"${cost_basis:.2f}",
                        pos_cols_display[4]: f"${current_value:.2f}", pos_cols_display[5]: f"${profit_loss:.2f}",
                        pos_cols_display[6]: f"{profit_loss_pct:.2f}%"
                    })
                if positions_data_list:
                    st.dataframe(pd.DataFrame(positions_data_list), use_container_width=True)
                else:
                    st.info(translator.t('no_valid_holdings_to_display', fallback="没有有效的持仓可供显示。"))
            else:
                st.info(translator.t('no_holdings', fallback="当前没有持仓。"))

                # vvvvvvvvvvvvvvvvvvvv START OF NEW SECTION vvvvvvvvvvvvvvvvvvvv
                # --- 2. 市场状态仪表盘 ---
                st.subheader(translator.t('market_regime_dashboard_header'))
                st.info(translator.t('market_regime_dashboard_help'))

                if hasattr(system, 'risk_manager') and system.risk_manager:
                    # 调用 RiskManager 获取当前状态 (这个调用内部是缓存的)
                    regime_info = system.risk_manager.get_current_market_regime(system_ref=system)

                    if regime_info and regime_info.get("status") != "UNKNOWN":
                        volatility = regime_info.get("volatility", "N/A")
                        trend = regime_info.get("trend", "N/A")
                        vix_value = regime_info.get("vix_value", "N/A")

                        # 定义状态对应的颜色和图标
                        vol_color_map = {"LOW": "green", "NORMAL": "blue", "HIGH": "red"}
                        trend_icon_map = {"BULL": "🔼", "BEAR": "🔽", "SIDEWAYS": "↔️"}

                        db_col1, db_col2, db_col3 = st.columns(3)
                        with db_col1:
                            st.metric(
                                label=translator.t('volatility_status_label'),
                                value=volatility,
                                help=f"Color indicates risk level: green (low), blue (normal), red (high)."
                            )
                            # 使用 HTML/CSS 来给 value 添加颜色
                            st.markdown(
                                f"<h3 style='color: {vol_color_map.get(volatility, 'black')};'>{volatility}</h3>",
                                unsafe_allow_html=True)

                        with db_col2:
                            st.metric(
                                label=translator.t('trend_status_label'),
                                value=f"{trend} {trend_icon_map.get(trend, '')}"
                            )

                        with db_col3:
                            st.metric(label=translator.t('vix_value_label'), value=f"{vix_value}")
                    else:
                        st.warning("无法获取当前市场状态信息。")
                else:
                    st.warning("风险管理模块不可用。")

                st.markdown("---")  # 添加分隔线
                # ^^^^^^^^^^^^^^^^^^^^ END OF NEW SECTION ^^^^^^^^^^^^^^^^^^^^

            # --- Recent Trades ---
            st.subheader(translator.t('recent_trades', fallback="近期交易记录"))
            trades = st.session_state.get('trades', [])
            if trades:
                recent_trades_list = trades[-5:]
                trades_data_list = []
                trade_cols_keys = ['time', 'stock_symbol', 'direction', 'quantity', 'price', 'amount']
                trade_cols_defaults = ["时间", "股票", "方向", "数量", "价格", "金额"]
                trade_cols_display = [translator.t(k, fallback=d) for k, d in zip(trade_cols_keys, trade_cols_defaults)]
                for trade_item in reversed(recent_trades_list):
                    direction_key = str(trade_item.get('direction', 'N/A')).lower()
                    trades_data_list.append({
                        trade_cols_display[0]: trade_item.get('timestamp', datetime.now()).strftime("%Y-%m-%d %H:%M"),
                        trade_cols_display[1]: trade_item.get('symbol', 'N/A'),
                        trade_cols_display[2]: translator.t(direction_key, fallback=trade_item.get('direction', 'N/A')),
                        trade_cols_display[3]: trade_item.get('quantity', 0),
                        trade_cols_display[4]: f"${trade_item.get('price', 0):.2f}",
                        trade_cols_display[5]: f"${trade_item.get('total', 0):.2f}"
                    })
                st.dataframe(pd.DataFrame(trades_data_list), use_container_width=True)
            else:
                st.info(translator.t('no_trades', fallback="暂无交易记录。"))

            # --- 市场热图 (优化版 - 按钮触发) ---
            st.subheader(translator.t('market_heatmap', fallback="市场热图"))

            # 确保 system 和 system.config 及相关属性存在
            default_heatmap_symbols = ["AAPL", "MSFT", "GOOGL"]  # 硬编码默认值
            full_heatmap_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", "JNJ"]  # 硬编码默认值
            if hasattr(system, 'config'):
                default_heatmap_symbols = getattr(system.config, 'DEFAULT_HEATMAP_SYMBOLS', default_heatmap_symbols)
                full_heatmap_symbols = getattr(system.config, 'DASHBOARD_HEATMAP_SYMBOLS', full_heatmap_symbols)
            else:
                logger.warning("system.config not available for heatmap symbol configuration.")

            if 'dashboard_heatmap_data' not in st.session_state:
                st.session_state.dashboard_heatmap_data = None
            if 'dashboard_heatmap_title_suffix' not in st.session_state:
                st.session_state.dashboard_heatmap_title_suffix = ""

            def _display_heatmap_data(heatmap_points_to_display, title_suffix_str):
                """Helper to display heatmap from pre-fetched data points."""
                if not heatmap_points_to_display:
                    st.info(translator.t('no_heatmap_data_to_display', fallback="无热图数据可显示。"))
                    return
                fig_ht = go.Figure()
                for sym_ht_disp, data_item_ht in heatmap_points_to_display.items():
                    chg_pct_val = data_item_ht["change_pct"]
                    bar_color = "green" if chg_pct_val >= 0 else "red"
                    fig_ht.add_trace(go.Bar(x=[sym_ht_disp], y=[chg_pct_val], name=sym_ht_disp,
                                            marker_color=bar_color, text=f"{chg_pct_val:.2f}%", textposition="auto"))
                title_text = translator.t('market_heatmap', fallback="市场热图") + title_suffix_str
                fig_ht.update_layout(title=title_text, xaxis_title=translator.t('stock_symbol'),
                                     yaxis_title=translator.t('change_pct_label'),
                                     yaxis_tickformat=".2f", height=300, showlegend=False)
                st.plotly_chart(fig_ht, use_container_width=True)

            # Buttons for loading heatmap
            col_ht1, col_ht2 = st.columns(2)
            if col_ht1.button(translator.t('load_default_heatmap', fallback="加载默认热图"),
                              key="btn_load_default_heatmap"):
                with st.spinner(translator.t('loading_default_heatmap', fallback="加载默认热图...")):
                    current_heatmap_data = {}
                    successful_fetches = 0
                    for symbol_ht in default_heatmap_symbols:
                        logger.info(f"Dashboard Heatmap: Fetching data for {symbol_ht} (default set)")
                        data = system.data_manager.get_historical_data(symbol_ht, days=5, interval="1d")
                        if data is not None and not data.empty and 'close' in data.columns and len(data) >= 2:
                            try:
                                last_p = float(data['close'].iloc[-1]);
                                prev_p = float(data['close'].iloc[-2])
                                chg_pct = ((last_p - prev_p) / prev_p) * 100 if prev_p != 0 else 0
                                current_heatmap_data[symbol_ht] = {"price": last_p, "change_pct": chg_pct}
                                successful_fetches += 1
                            except Exception as e_calc:
                                logger.warning(f"计算热图数据点时出错 ({symbol_ht}): {e_calc}")
                        else:
                            logger.warning(f"无法为热图获取有效的 {symbol_ht} 数据。")
                        # time.sleep(0.1) # Consider removing if yfinance helper has sleep

                    if successful_fetches > 0:
                        st.session_state.dashboard_heatmap_data = current_heatmap_data
                    else:
                        st.session_state.dashboard_heatmap_data = None
                    st.session_state.dashboard_heatmap_title_suffix = translator.t('default_view_suffix',
                                                                                   fallback=" (默认视图)")
                    st.rerun()  # Rerun to display the newly fetched data

            if col_ht2.button(translator.t('load_full_heatmap', fallback="加载完整热图"), key="btn_load_full_heatmap"):
                with st.spinner(translator.t('loading_full_heatmap', fallback="加载完整热图...")):
                    current_heatmap_data = {}
                    successful_fetches = 0
                    for symbol_ht in full_heatmap_symbols:
                        logger.info(f"Dashboard Heatmap: Fetching data for {symbol_ht} (full set)")
                        data = system.data_manager.get_historical_data(symbol_ht, days=5, interval="1d")
                        if data is not None and not data.empty and 'close' in data.columns and len(data) >= 2:
                            try:
                                last_p = float(data['close'].iloc[-1]);
                                prev_p = float(data['close'].iloc[-2])
                                chg_pct = ((last_p - prev_p) / prev_p) * 100 if prev_p != 0 else 0
                                current_heatmap_data[symbol_ht] = {"price": last_p, "change_pct": chg_pct}
                                successful_fetches += 1
                            except Exception as e_calc:
                                logger.warning(f"计算热图数据点时出错 ({symbol_ht}): {e_calc}")
                        else:
                            logger.warning(f"无法为热图获取有效的 {symbol_ht} 数据。")
                        # time.sleep(0.1) # Consider removing if yfinance helper has sleep

                    if successful_fetches > 0:
                        st.session_state.dashboard_heatmap_data = current_heatmap_data
                    else:
                        st.session_state.dashboard_heatmap_data = None
                    st.session_state.dashboard_heatmap_title_suffix = translator.t('full_view_suffix',
                                                                                   fallback=" (完整视图)")
                    st.rerun()

            # Display existing heatmap data from session state
            if st.session_state.dashboard_heatmap_data:
                _display_heatmap_data(st.session_state.dashboard_heatmap_data,
                                      st.session_state.dashboard_heatmap_title_suffix)
            else:
                st.info(translator.t('click_to_load_heatmap', fallback="点击按钮加载市场热图。"))

        except Exception as e:
            logger.error(f"渲染仪表盘时出错: {e}", exc_info=True)
            st.error(translator.t('error_rendering_dashboard', fallback="渲染仪表盘时出错。") + f": {e}")
            st.code(traceback.format_exc())

    def render_market_tab(self, system: Any):
        """
        [最终修复版] 渲染市场标签页，采用正确的左右分栏布局。
        """
        st.header(translator.t('market_data', fallback="市场数据"))
        logger.debug("Rendering market tab...")
        try:
            # --- 1. 股票代码输入 (页面顶部) ---
            symbol = st.text_input(
                translator.t('enter_stock_symbol'),
                st.session_state.get('current_market_symbol', 'AAPL'),
                key="market_tab_symbol_input_v3"
            ).upper()
            st.session_state['current_market_symbol'] = symbol

            if not symbol: return

            # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
            # --- 2. 创建左右分栏 ---
            left_col, right_col = st.columns([2, 1])  # 左2右1比例

            # --- 3. 在左栏中渲染所有与“数据”相关的内容 ---
            with left_col:
                st.subheader(
                    f"{translator.t('data_and_analysis_for', fallback='{symbol} Data & Analysis').format(symbol=symbol)}")

                session_key_prefix = f"market_view_{symbol}"
                session_key_data = f"{session_key_prefix}_data"
                session_key_col_map = f"{session_key_prefix}_col_map"

                # a. 数据加载按钮
                if st.button(translator.t('load_data_button').format(symbol=symbol), key=f"load_btn_{symbol}"):
                    with st.spinner(translator.t('loading_stock_data').format(symbol=symbol)):
                        data = system.get_stock_data(symbol, days=365)
                        if data is not None and not data.empty:
                            data_std, col_map = self._get_standardized_price_data(data)
                            if col_map:
                                st.session_state[session_key_data] = data_std
                                st.session_state[session_key_col_map] = col_map
                            else:
                                st.session_state[session_key_data] = {"error": "data_format_invalid"}
                        else:
                            st.session_state[session_key_data] = {"error": "data_fetch_failed"}
                    st.rerun()
            # vvvvvvvvvvvvvvvvvvvv START OF YOUR PASTED CODE (INTEGRATED) vvvvvvvvvvvvvvvvvvvv
            # --- 3. 渲染已加载的数据 ---
                loaded_data_state = st.session_state.get(f'{session_key_prefix}_data')

                if isinstance(loaded_data_state, str) and loaded_data_state == "LOADING":
                    st.info(translator.t('loading_stock_data', fallback="正在加载股票数据..."))

                elif isinstance(loaded_data_state, dict) and "error" in loaded_data_state:
                    if loaded_data_state["error"] == "data_fetch_failed":
                        st.error(translator.t('error_fetching_data_for', fallback="无法获取 {symbol} 的数据。").format(
                            symbol=symbol))
                    else:  # data_format_invalid
                        st.error(translator.t('error_invalid_data_format',
                                              fallback="获取到的 {symbol} 数据格式不正确或不完整。").format(
                            symbol=symbol))

                elif isinstance(loaded_data_state, pd.DataFrame):
                    # 只有当确认 loaded_data_state 是 DataFrame 时，才执行这里的逻辑
                    if loaded_data_state.empty:
                        st.warning(translator.t('warning_no_data_available', fallback="该股票代码没有可用的历史数据。"))
                    else:
                        data = loaded_data_state
                        col_map = st.session_state.get(f'{session_key_prefix}_col_map', {})

                        # --- Display Price Info ---
                        st.subheader(translator.t('price_info_for', fallback="{symbol} 价格信息").format(symbol=symbol))
                        try:
                            close_col = col_map['close']
                            last_price = float(data[close_col].iloc[-1]) if len(data) > 0 else 0
                            prev_price = float(data[close_col].iloc[-2]) if len(data) > 1 else last_price
                            change_pct = ((last_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0
                            open_price = float(data[col_map['open']].iloc[-1]) if 'open' in col_map and len(
                                data) > 0 else 'N/A'
                            high_price = float(data[col_map['high']].iloc[-1]) if 'high' in col_map and len(
                                data) > 0 else 'N/A'
                            low_price = float(data[col_map['low']].iloc[-1]) if 'low' in col_map and len(
                                data) > 0 else 'N/A'

                            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                            with m_col1:
                                st.metric(translator.t('latest_price'), f"${last_price:.2f}", f"{change_pct:.2f}%")
                            with m_col2:
                                st.metric(translator.t('open_price'),
                                          f"${open_price:.2f}" if isinstance(open_price, float) else open_price)
                            with m_col3:
                                st.metric(translator.t('high_price'),
                                          f"${high_price:.2f}" if isinstance(high_price, float) else high_price)
                            with m_col4:
                                st.metric(translator.t('low_price'),
                                          f"${low_price:.2f}" if isinstance(low_price, float) else low_price)
                        except Exception as e_price:
                            logger.error(f"显示价格信息时出错 ({symbol}): {e_price}", exc_info=True)
                            st.warning(translator.t('warning_display_price'))

                            # --- K线图 ---
                        st.subheader(translator.t('candlestick_chart', fallback="K线图"))
                        if all(c in col_map for c in ['open', 'high', 'low', 'close']):
                            try:
                                fig_k = go.Figure(data=[
                                    go.Candlestick(x=data.index, open=data[col_map['open']],
                                                   high=data[col_map['high']],
                                                   low=data[col_map['low']], close=data[col_map['close']],
                                                   increasing_line_color='red', decreasing_line_color='green',
                                                   name=symbol)])
                                fig_k.update_layout(title=f"{symbol} {translator.t('candlestick_chart')}",
                                                    xaxis_title=translator.t('date', fallback="日期"),
                                                    yaxis_title=translator.t('price', fallback="价格"), height=500,
                                                    xaxis_rangeslider_visible=False)
                                st.plotly_chart(fig_k, use_container_width=True)
                            except Exception as e_kline:
                                logger.error(f"绘制K线图失败 ({symbol}): {e_kline}", exc_info=True);
                                st.error(
                                    translator.t('error_rendering_kline', fallback="无法绘制K线图。"))
                        else:
                            st.warning(
                                translator.t('warning_kline_data_incomplete',
                                             fallback="无法绘制K线图：缺少必要的OHLC数据。"))

                            # --- 成交量图 ---
                        st.subheader(translator.t('volume', fallback="成交量"))
                        if 'volume' in col_map:
                            try:
                                fig_v = go.Figure(
                                    data=[go.Bar(x=data.index, y=data[col_map['volume']], marker_color='blue',
                                                 name=translator.t('volume'))])
                                fig_v.update_layout(title=f"{symbol} {translator.t('volume')}",
                                                    xaxis_title=translator.t('date'),
                                                    yaxis_title=translator.t('volume'),
                                                    height=300)
                                st.plotly_chart(fig_v, use_container_width=True)
                            except Exception as e_volume:
                                logger.error(f"绘制成交量图失败 ({symbol}): {e_volume}", exc_info=True);
                                st.error(
                                    translator.t('error_rendering_volume', fallback="无法绘制成交量图。"))
                        else:
                            st.warning(
                                translator.t('warning_volume_data_missing',
                                             fallback="无法绘制成交量图：缺少成交量数据。"))

                            # --- 市场情绪分析 ---
                        st.subheader(translator.t('market_sentiment', fallback="市场情绪分析"))
                        try:
                            # Assume system.analyze_sentiment is synchronous or handled elsewhere
                            sentiment = system.analyze_sentiment(symbol)
                            if sentiment:
                                s_col1, s_col2, s_col3, s_col4 = st.columns(4)
                                score = sentiment.get('composite_score', 0);
                                color = "green" if score > 0.1 else "red" if score < -0.1 else "gray"
                                with s_col1:
                                    st.markdown(f"<h1 style='text-align: center; color: {color};'>{score:.2f}</h1>",
                                                unsafe_allow_html=True);
                                    st.markdown(
                                        f"<p style='text-align: center;'>{translator.t('composite_score', fallback='综合得分')}</p>",
                                        unsafe_allow_html=True)
                                with s_col2:
                                    st.metric(translator.t('news_sentiment'),
                                              f"{sentiment.get('news_score', 0):.2f}")
                                with s_col3:
                                    st.metric(translator.t('social_sentiment'),
                                              f"{sentiment.get('social_score', 0):.2f}")
                                with s_col4:
                                    st.metric(translator.t('technical_sentiment'),
                                              f"{sentiment.get('technical_score', 0):.2f}")
                                status = sentiment.get('sentiment_status',
                                                       translator.t('unknown', fallback='未知'));
                                timestamp = sentiment.get('timestamp', 'N/A')
                                st.info(
                                    f"{translator.t('sentiment_status')}: {status} ({translator.t('last_updated')}: {timestamp})")
                            else:
                                st.warning(translator.t('warning_sentiment_module_missing',
                                                        fallback="情绪分析模块不可用或未返回数据。"))
                        except Exception as e_sentiment:
                            logger.error(f"分析市场情绪时出错 ({symbol}): {e_sentiment}", exc_info=True);
                            st.error(
                                translator.t('error_analyzing_sentiment', fallback="分析市场情绪时出错。"))
                        # --- 3. LLM 新闻分析板块
                    self._render_llm_analysis_ui(system, symbol)

            with right_col:
                st.subheader(translator.t('dynamic_risk_suggestions_header'))
                st.info(translator.t('dynamic_risk_suggestions_help'))

                if hasattr(system, 'risk_manager') and system.risk_manager:
                    # a. 获取市场状态
                    regime_info = system.risk_manager.get_current_market_regime(system_ref=system)

                    if regime_info and regime_info.get("status") != "UNKNOWN":
                        # b. 获取基于该状态的建议
                        suggestions = system.risk_manager.get_dynamic_risk_suggestions(
                            regime_info.get("status"))

                        st.caption(translator.t('suggestion_reason_label'))
                        st.success(f"*{suggestions.get('suggestion_reason')}*")  # 用斜体和成功框突出显示

                        st.metric(
                            label=translator.t('trade_qty_multiplier_label'),
                            value=f"{suggestions.get('trade_quantity_multiplier', 1.0):.2f}x"
                        )
                        st.metric(
                            label=translator.t('alpha_threshold_multiplier_label'),
                            value=f"{suggestions.get('alpha_threshold_multiplier', 1.0):.2f}x"
                        )
                    else:
                        st.warning("无法获取市场状态，因此无动态建议。")
                else:
                    st.warning("风险管理模块不可用。")

                # --- 5. 市场扫描器 (放在分栏之后，页面底部) ---
            st.markdown("---")
            self._render_market_scanner_ui(system)




        except Exception as e_market_main:
            st.error(f"渲染市场主标签页时发生严重错误: {e_market_main}")
            logger.error("渲染市场主标签页时发生严重错误", exc_info=True)
            st.code(traceback.format_exc())

    def _render_llm_analysis_ui(self, system, symbol):
        """
        [最终修复版] 渲染 LLM 新闻分析板块。
        分析过程仅在用户明确点击按钮时触发，且结果直接显示，不依赖 session_state 触发。
        """
        st.subheader(translator.t('llm_news_analysis_header', fallback="📰 LLM 新闻分析 (Gemini)"))

        extractor = None
        if hasattr(system, 'strategy_manager') and system.strategy_manager:
            extractor = system.strategy_manager.text_feature_extractor

        if extractor and extractor.is_available:
            col1, col2 = st.columns([2, 1])
            with col1:
                # 使用 session_state 来记住上次为特定符号输入的公司名
                company_name_key = f"llm_company_name_{symbol}"
                default_company_name = st.session_state.get(company_name_key, symbol)
                company_name = st.text_input(
                    translator.t('company_name_for_llm', fallback="公司全名 (用于精确搜索)"),
                    value=default_company_name,
                    key=f"llm_company_name_input_{symbol}"
                )
                # 更新 session_state 以便下次使用
                st.session_state[company_name_key] = company_name

            with col2:
                gemini_models = getattr(system.config, 'GEMINI_MODELS', [''])
                selected_model = st.selectbox(
                    translator.t('select_gemini_model_label', fallback="选择 Gemini 模型:"),
                    gemini_models, key=f"llm_model_select_{symbol}"
                )

            # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
            button_label = translator.t('run_llm_analysis_button', fallback="运行 Gemini 新闻分析")

            # 2. 将准备好的文本作为 label 参数传递给 st.button
            if st.button(button_label, key=f"llm_analysis_btn_{symbol}"):



                # --- 所有耗时操作都在这个 if 块内执行 ---
                with st.spinner(translator.t('llm_analyzing_spinner', fallback="Gemini 正在搜索并分析新闻...")):
                    # 1. 调用 extractor
                    # get_and_extract_features 返回 (features, full_analysis) 或 (None, error_dict) 或 None
                    result = extractor.get_and_extract_features(
                        symbol,
                        company_name,
                        selected_model # <--- 传递
                    )

                    # 2. 直接处理和渲染结果，不再使用 st.rerun()
                    if result:
                        features, full_analysis = result

                        if full_analysis and 'error' not in full_analysis:
                            # 成功获取并分析
                            st.metric(
                                translator.t('gemini_aggregated_sentiment', fallback="综合情绪得分"),
                                f"{full_analysis.get('aggregated_sentiment_score', 0.0):.2f}"
                            )
                            st.info(
                                f"**{translator.t('gemini_key_summary', fallback='核心摘要')}:** {full_analysis.get('key_summary', 'N/A')}")

                            with st.expander(translator.t('gemini_analyzed_articles', fallback="查看分析的新闻源")):
                                for article in full_analysis.get('analyzed_articles', []):
                                    st.markdown(f"**[{article.get('title', 'No Title')}]({article.get('url', '#')})**")
                                    st.caption(
                                        f"**摘要:** {article.get('summary', 'N/A')} | **情绪分:** {article.get('sentiment_score', 0.0):.2f}")

                        elif full_analysis and 'error' in full_analysis:
                            # API 或内部逻辑返回了错误信息
                            st.error(f"Gemini 分析失败: {full_analysis['error']}")
                        else:
                            # 罕见情况：返回了 None 或其他无效格式
                            st.error(translator.t('llm_analysis_failed_error', fallback="Gemini 分析失败或返回空结果。"))
                    else:
                        # get_and_extract_features 本身返回了 None
                        st.error(translator.t('llm_analysis_failed_error', fallback="Gemini 分析失败或返回空结果。"))

            st.caption(translator.t('llm_analysis_time_warning',
                                    fallback="请注意：Gemini 分析可能需要较长时间，请耐心等待。"))
            # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^
        else:
            st.warning(translator.t('llm_module_unavailable_warning', fallback="文本分析模块未能初始化。"))

    def _render_market_scanner_ui(self, system):
        """
        [新增] 渲染市场扫描功能的 UI 组件。
        这是一个独立的辅助方法，以避免在 render_market_tab 中造成混乱。
        """
        st.header(translator.t('market_scan', fallback="市场扫描"))
        try:
            with st.form("market_scan_form"):
                col1, col2 = st.columns(2)
                with col1:
                    scan_symbols = st.text_input(
                        translator.t('scan_symbols_label', fallback="股票代码 (逗号分隔)"),
                        "AAPL,MSFT,GOOGL,AMZN",
                        key="scan_symbols_input"
                    )
                    scan_market = st.selectbox(
                        translator.t('market', fallback="市场"),
                        ["US", "CN"], 0,
                        key="scan_market_select"
                    )
                with col2:
                    vol_threshold = st.slider(
                        translator.t('scan_vol_threshold', fallback="成交量变化阈值 (%)"),
                        0, 200, 50, key="scan_vol_slider"
                    )
                    price_threshold = st.slider(
                        translator.t('scan_price_threshold', fallback="价格变化阈值 (%)"),
                        0, 50, 5, key="scan_price_slider"
                    )

                scan_days = st.slider(
                    translator.t('scan_lookback_days', fallback="回溯天数"),
                    1, 30, 5, key="scan_days_slider"
                )

                submitted = st.form_submit_button(translator.t('scan', fallback="扫描"))

                if submitted:
                    criteria = {
                        "symbols": scan_symbols, "market": scan_market,
                        "vol_threshold": vol_threshold, "price_threshold": price_threshold,
                        "days": scan_days
                    }
                    with st.spinner(translator.t('scanning_market', fallback="正在扫描市场...")):
                        scan_results = system.run_market_scan(criteria)

                    if scan_results:
                        st.success(translator.t('scan_found_matches', fallback="找到 {count} 个符合条件的股票").format(
                            count=len(scan_results)))

                        result_data = []
                        scan_cols_keys = ['stock_symbol', 'latest_price', 'price_change', 'volume_change']
                        scan_cols_defaults = ["股票", "最新价", "价格变化(%)", "成交量变化(%)"]
                        scan_cols_display = [translator.t(k, fallback=d) for k, d in
                                             zip(scan_cols_keys, scan_cols_defaults)]

                        for result in scan_results:
                            result_data.append({
                                scan_cols_display[0]: result.get('symbol', 'N/A'),
                                scan_cols_display[1]: f"${result.get('last_price', 0):,.2f}",
                                scan_cols_display[2]: f"{result.get('price_change', 0):.2f}%",
                                scan_cols_display[3]: f"{result.get('vol_change', 0):.2f}%"
                            })
                        st.dataframe(pd.DataFrame(result_data), use_container_width=True)

                        if scan_results and 'data' in scan_results[0]:
                            first_result = scan_results[0]
                            chart_data, chart_col_map = self._get_standardized_price_data(first_result['data'])
                            if chart_data is not None and all(
                                    c in chart_col_map for c in ['open', 'high', 'low', 'close']):
                                st.subheader(
                                    f"{first_result['symbol']} {translator.t('recent_trend', fallback='近期走势')}")
                                fig_scan = go.Figure(data=[
                                    go.Candlestick(x=chart_data.index, open=chart_data[chart_col_map['open']],
                                                   high=chart_data[chart_col_map['high']],
                                                   low=chart_data[chart_col_map['low']],
                                                   close=chart_data[chart_col_map['close']])])
                                fig_scan.update_layout(xaxis_title=translator.t('date'),
                                                       yaxis_title=translator.t('price'), height=400,
                                                       xaxis_rangeslider_visible=False)
                                st.plotly_chart(fig_scan, use_container_width=True)
                    else:
                        st.info(translator.t('scan_no_matches', fallback="没有找到符合条件的股票。"))
        except Exception as e_scan_form:
            logger.error(f"处理市场扫描表单时出错: {e_scan_form}", exc_info=True)
            st.error(translator.t('error_market_scan_form', fallback="处理市场扫描时出错。"))

    def render_trading_tab(self, system):
        """渲染交易标签页 (使用翻译, 确保使用 system.order_executor)"""
        st.header(translator.t('trade', fallback="交易"))
        logger.debug("Rendering trading tab...")
        try:
            trade_tab, batch_tab = st.tabs([
                translator.t('single_trade', fallback="单笔交易"),
                translator.t('batch_trade', fallback="批量交易")
            ])

            with trade_tab:
                with st.form("trade_form"):
                    t_col1, t_col2 = st.columns(2)
                    with t_col1:
                        symbol_trade = st.text_input(translator.t('stock_symbol'), "AAPL", key="trade_symbol").upper()
                    # Get translated options for radio/selectbox
                    buy_option = translator.t('buy', fallback="买入");
                    sell_option = translator.t('sell', fallback="卖出")
                    market_option = translator.t('market_order', fallback="市价单");
                    limit_option = translator.t('limit_order', fallback="限价单")
                    with t_col2:
                        direction_trade = st.radio(translator.t('direction'), [buy_option, sell_option],
                                                   key="trade_direction")


                    t_col3, t_col4 = st.columns(2)
                    with t_col3:
                        quantity_trade = st.number_input(translator.t('quantity'), min_value=1, value=10,
                                                         key="trade_quantity")
                    with t_col4:
                        order_type_trade = st.selectbox(translator.t('order_type', fallback="订单类型"),
                                                        [market_option, limit_option], key="trade_order_type")

                    if system.data_manager:
                        realtime_price_data = system.data_manager.get_realtime_price(symbol_trade)
                        current_price_trade = realtime_price_data['price'] if realtime_price_data else None
                    else:
                        st.error(translator.t('error_dm_not_available', fallback="数据管理器不可用，无法获取实时价格。"))
                        current_price_trade = None

                    price_trade = None
                    # Get realtime price *outside* the conditional for reuse
                    realtime_price_data = system.data_manager.get_realtime_price(
                        symbol_trade)  # Use data_manager directly
                    current_price_trade = realtime_price_data['price'] if realtime_price_data else None

                    if order_type_trade == limit_option:
                        default_price_trade = current_price_trade if current_price_trade else 100.0
                        price_trade = st.number_input(translator.t('price'), min_value=0.01, value=default_price_trade,
                                                      format="%.2f", key="trade_price")
                    else:  # Market Order
                        if current_price_trade:
                            st.info(
                                f"{translator.t('current_market_price', fallback='当前市价')}: ${current_price_trade:.2f}")
                            price_trade = current_price_trade  # Use fetched price for calculation and potential submission
                        else:
                            st.warning(translator.t('warning_cannot_get_market_price',
                                                    fallback="无法获取当前市价，市价单可能无法准确执行或使用默认值。"))
                            # Decide: Either disallow market order or allow submission without exact price display
                            # Option 1: Disallow (more complex form logic needed)
                            # Option 2: Allow submission, backend executor MUST handle fetching market price if price_trade is None
                            price_trade = None  # Signal to executor to fetch market price

                    if price_trade is not None and quantity_trade > 0:  # Calculate only if price is known
                        total_cost_trade = quantity_trade * price_trade
                        st.write(f"{translator.t('total_trade_amount', fallback='交易总额')}: ${total_cost_trade:.2f}")
                    else:
                        total_cost_trade = 0  # Cannot calculate without price
                        st.write(
                            f"{translator.t('total_trade_amount', fallback='交易总额')}: {translator.t('calculating', fallback='计算中...') if order_type_trade == market_option else translator.t('requires_limit_price', fallback='需指定限价')}")

                    # --- Validation ---
                    portfolio = st.session_state.get('portfolio', {})
                    is_buy = (direction_trade == buy_option)
                    is_sell = (direction_trade == sell_option)
                    can_submit = True

                    if is_buy and total_cost_trade > 0 and total_cost_trade > portfolio.get('cash', 0):
                        st.error(translator.t('error_insufficient_funds',
                                              fallback="资金不足! 需要 ${needed:.2f}, 可用 ${available:.2f}").format(
                            needed=total_cost_trade, available=portfolio.get('cash', 0)))
                        can_submit = False
                    if is_sell:
                        position = portfolio.get('positions', {}).get(symbol_trade, {})
                        current_quantity = position.get('quantity', 0)
                        if quantity_trade > current_quantity:
                            st.error(translator.t('error_insufficient_position',
                                                  fallback="持仓不足! 需要 {needed} 股, 持有 {available} 股").format(
                                needed=quantity_trade, available=current_quantity))
                            can_submit = False
                    if price_trade is None and order_type_trade == limit_option:
                        st.error(translator.t('error_limit_price_required', fallback="限价单需要指定价格。"))
                        can_submit = False

                    # --- Submission ---
                    submitted = st.form_submit_button(translator.t('execute_trade', fallback="执行交易"),
                                                      disabled=not can_submit)
                    if submitted and can_submit:
                        # Map UI text back to internal identifiers
                        internal_direction = 'Buy' if is_buy else 'Sell'
                        internal_order_type = 'Market Order' if order_type_trade == market_option else 'Limit Order'

                        # Prepare order data for the backend
                        order_data_internal = {
                            "symbol": symbol_trade,
                            "quantity": quantity_trade,
                            "price": price_trade,  # Pass None for market if needed by executor
                            "direction": internal_direction,
                            "order_type": internal_order_type
                        }
                        logger.info(f"Submitting trade via system.execute_trade: {order_data_internal}")
                        result = system.execute_trade(order_data_internal)

                        if result and result.get('success'):
                            # Use execution price from result if available
                            exec_price = result.get('price', price_trade)
                            st.success(translator.t('trade_success_msg',
                                                    fallback="交易成功! {direction} {quantity} 股 {symbol} @ ${price:.2f}").format(
                                direction=direction_trade, quantity=quantity_trade, symbol=symbol_trade,
                                price=exec_price
                            ))
                            st.rerun()  # Refresh UI after successful trade
                        elif result:
                            st.error(translator.t('trade_failed_msg', fallback="交易失败: {message}").format(
                                message=result.get('message', '未知错误')))
                        else:
                            st.error(translator.t('trade_failed_msg', fallback="交易失败: {message}").format(
                                message='执行器未返回有效结果。'))

                # --- Display Holdings ---
                st.subheader(translator.t('current_holdings'))
                positions = st.session_state.get('portfolio', {}).get('positions', {})
                if positions:
                    pos_data = []
                    pos_cols = [translator.t('stock_symbol'), translator.t('quantity'), translator.t('current_price'),
                                translator.t('cost_basis'), translator.t('market_value'),
                                translator.t('cost_value', fallback='成本'), translator.t('profit_loss'),
                                translator.t('profit_loss_pct')]
                    for symbol, position in positions.items():
                        q = position.get('quantity', 0);
                        p = position.get('current_price', 0);
                        c = position.get('cost_basis', 0)
                        if not all(isinstance(v, (int, float)) for v in [q, p, c]): continue
                        mv = q * p;
                        cv = q * c;
                        pl = mv - cv;
                        pl_pct = (pl / cv) * 100 if cv != 0 else 0
                        pos_data.append(
                            {pos_cols[0]: symbol, pos_cols[1]: q, pos_cols[2]: f"${p:.2f}", pos_cols[3]: f"${c:.2f}",
                             pos_cols[4]: f"${mv:.2f}", pos_cols[5]: f"${cv:.2f}", pos_cols[6]: f"${pl:.2f}",
                             pos_cols[7]: f"{pl_pct:.2f}%"})
                    if pos_data:
                        st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
                    else:
                        st.info(translator.t('no_holdings'))  # Should not happen if positions exist
                else:
                    st.info(translator.t('no_holdings'))

                # --- Display Recent Trades ---
                st.subheader(translator.t('recent_trades'))
                trades = st.session_state.get('trades', [])
                if trades:
                    last_n = min(10, len(trades));
                    recent_trades = trades[-last_n:]
                    trade_data = [];
                    trade_cols = [translator.t('time'), translator.t('stock_symbol'), translator.t('direction'),
                                  translator.t('quantity'), translator.t('price'), translator.t('amount')]
                    for trade in reversed(recent_trades):
                        dir_key = str(trade.get('direction', 'N/A')).lower()
                        trade_data.append(
                            {trade_cols[0]: trade.get('timestamp', datetime.now()).strftime("%Y-%m-%d %H:%M"),
                             trade_cols[1]: trade.get('symbol', 'N/A'),
                             trade_cols[2]: translator.t(dir_key, fallback=trade.get('direction', 'N/A')),
                             trade_cols[3]: trade.get('quantity', 0), trade_cols[4]: f"${trade.get('price', 0):.2f}",
                             trade_cols[5]: f"${trade.get('total', 0):.2f}"})
                    st.dataframe(pd.DataFrame(trade_data), use_container_width=True)
                else:
                    st.info(translator.t('no_trades'))

            with batch_tab:
                if hasattr(system, 'order_executor') and system.order_executor is not None and hasattr(
                        system.order_executor, 'render_batch_trading_ui'):
                    logger.debug("Rendering batch trading UI via OrderExecutor.")
                    system.order_executor.render_batch_trading_ui(system)
                else:
                    st.warning(translator.t('warning_batch_trade_unavailable',
                                            fallback="批量交易功能不可用。请检查交易组件是否正确初始化。"))
                    logger.warning("Batch trading UI unavailable: OrderExecutor or render method missing.")

        except Exception as e:
            st.error(f"渲染交易主标签页时发生严重错误: {e}")
            logger.error("渲染交易主标签页时发生严重错误", exc_info=True)
            st.code(traceback.format_exc())

    def render_strategy_tab(self, system):
        """渲染策略标签页 (调用 strategy_manager)"""
        logger.debug("Rendering strategy tab...")
        try:
            if hasattr(system, 'strategy_manager') and system.strategy_manager is not None and \
                    hasattr(system.strategy_manager, 'render_strategy_ui') and callable(
                system.strategy_manager.render_strategy_ui):
                logger.debug("Calling system.strategy_manager.render_strategy_ui...")
                system.strategy_manager.render_strategy_ui(system)
            else:
                st.warning(translator.t('warning_strategy_manager_load_failed',
                                        fallback="策略管理器未正确初始化或UI渲染方法不存在。"))
                logger.warning("UIManager: system.strategy_manager or render_strategy_ui unavailable.")
        except Exception as e:
            st.error(f"渲染策略标签页时发生严重错误: {e}"); st.code(traceback.format_exc()); logger.error(
                "渲染策略标签页时发生严重错误", exc_info=True)

    def safe_value(self, data_dict: Dict, key: str, default: Any = 0) -> Any:  # 修改参数以接受字典和键
        """安全地从字典获取值并尝试转换为float，如果适用。"""
        logger.debug(f"safe_value called with data_dict: {type(data_dict)}, key: {key}")
        try:
            if data_dict is None or not isinstance(data_dict, dict):
                logger.warning(f"safe_value: data_dict is None or not a dict for key '{key}'.")
                return default

            value = data_dict.get(key, default)

            # 如果期望的是数值类型，尝试转换
            if isinstance(default, (int, float)) or isinstance(value, (int, float, str)):
                try:
                    # 处理可能已经是数字的情况
                    if isinstance(value, (int, float)) and pd.notna(value):
                        return float(value)
                    # 处理字符串表示的数字
                    elif isinstance(value, str):
                        cleaned_value = value.replace('$', '').replace(',', '')  # 移除货币符号和逗号
                        if cleaned_value.replace('.', '', 1).isdigit() or \
                                (cleaned_value.startswith('-') and cleaned_value[1:].replace('.', '', 1).isdigit()):
                            return float(cleaned_value)
                except (ValueError, TypeError):
                    logger.warning(
                        f"safe_value: Could not convert value '{value}' for key '{key}' to float. Returning as is or default.")
                    # Return original value if conversion fails but it exists, or default if it doesn't exist
                    return data_dict.get(key, default)  # Return original if conversion fails

            return value  # Return value as is if not numeric context or failed conversion
        except Exception as e:  # General catch-all
            logger.error(f"Unexpected error in safe_value for key '{key}': {e}", exc_info=True)
            return default

    def render_portfolio_tab(self, system):
        """渲染投资组合标签页 (使用翻译)"""
        st.header(translator.t('portfolio_management', fallback="投资组合管理"))
        logger.debug("Rendering portfolio tab...")
        # --- Try block for the entire portfolio tab rendering ---
        try:
            # --- Portfolio Overview ---
            portfolio = st.session_state.get('portfolio', {'cash': 0, 'positions': {}})
            cash = portfolio.get('cash', 0);
            positions_value = 0
            for position in portfolio.get('positions', {}).values():
                q = position.get('quantity', 0);
                p = position.get('current_price', 0);
                if isinstance(q, (int, float)) and isinstance(p, (int, float)): positions_value += q * p
            calculated_total_value = cash + positions_value
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                st.metric(translator.t('total_assets', fallback="账户总值"), f"${calculated_total_value:,.2f}")
            cash_pct = (cash / calculated_total_value) * 100 if calculated_total_value > 0 else 0
            with p_col2:
                st.metric(translator.t('available_funds', fallback="可用资金"), f"${cash:,.2f}",
                          f"{cash_pct:.1f}%" if calculated_total_value > 0 else None)
            invested_pct = (positions_value / calculated_total_value) * 100 if calculated_total_value > 0 else 0
            with p_col3:
                st.metric(translator.t('investment_amount', fallback="持仓市值"), f"${positions_value:,.2f}",
                          f"{invested_pct:.1f}%" if calculated_total_value > 0 else None)

            # --- Holdings Details ---
            st.subheader(translator.t('holdings_details', fallback="持仓明细"))
            positions = portfolio.get('positions', {})
            if positions:
                pos_data = []
                pos_cols = [translator.t('stock_symbol'), translator.t('quantity'), translator.t('current_price'),
                            translator.t('cost_basis'), translator.t('market_value'),
                            translator.t('cost_value', fallback='成本'), translator.t('profit_loss'),
                            translator.t('profit_loss_pct')]
                for symbol, pos in positions.items():
                    q = pos.get('quantity', 0); p = pos.get('current_price', 0); c = pos.get('cost_basis', 0)
                    if not all(isinstance(v, (int, float)) for v in [q, p, c]): continue
                    mv = q * p; cv = q * c; pl = mv - cv; pl_pct = (pl / cv) * 100 if cv != 0 else 0
                    pos_data.append(
                        {pos_cols[0]: symbol, pos_cols[1]: q, pos_cols[2]: f"${p:.2f}", pos_cols[3]: f"${c:.2f}",
                         pos_cols[4]: f"${mv:.2f}", pos_cols[5]: f"${cv:.2f}", pos_cols[6]: f"${pl:.2f}",
                         pos_cols[7]: f"{pl_pct:.2f}%"})
                if pos_data:
                    st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
                else:
                    st.info(translator.t('no_holdings', fallback="当前没有持仓。")) # Should not happen if positions exist
            else:
                st.info(translator.t('no_holdings', fallback="当前没有持仓。"))

            # --- Asset Allocation ---
            if positions or cash > 0:
                st.subheader(translator.t('asset_allocation', fallback="资产配置"))
                try:
                    labels = [translator.t('cash', fallback="现金")] + list(positions.keys())
                    values = [cash] + [
                        self.safe_value(positions[s], 'quantity') * self.safe_value(positions[s], 'current_price') for s
                        in positions.keys()]
                    valid_labels = [l for l, v in zip(labels, values) if v > 1e-6]
                    valid_values = [v for v in values if v > 1e-6]
                    if valid_values:
                        fig_pie = go.Figure(
                            data=[go.Pie(labels=valid_labels, values=valid_values, hole=.3, textinfo='percent+label')])
                        fig_pie.update_layout(title=translator.t('portfolio_composition', fallback="投资组合配置"),
                                              height=400, margin=dict(t=50, b=0, l=0, r=0))
                        st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.info(translator.t('no_positive_value_assets', fallback="没有正价值的资产可供展示配置。"))
                except Exception as e_pie:
                    logger.error(f"绘制资产配置饼图时出错: {e_pie}", exc_info=True)
                    st.warning(translator.t('error_rendering_pie_chart', fallback="无法绘制资产配置图。"))

            # --- Risk Analysis ---
            st.subheader(translator.t('risk_analysis', fallback="风险分析"))
            if hasattr(system, 'risk_manager') and system.risk_manager is not None:
                try:
                    # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
                    # 1. 获取当前持仓的所有股票代码
                    portfolio_for_risk = st.session_state.get('portfolio', {})
                    position_symbols = list(portfolio_for_risk.get('positions', {}).keys())

                    # 2. Fetch historical data ONLY for the symbols currently in the portfolio
                    portfolio_historical_data = {}
                    if position_symbols:
                        with st.spinner("正在加载风险分析所需的历史数据..."):
                            for symbol in position_symbols:
                                # This call is cached by Streamlit, so it's efficient
                                hist_data = system.data_manager.get_historical_data(symbol, days=252)
                                if hist_data is not None and not hist_data.empty:
                                    portfolio_historical_data[symbol] = hist_data

                    # 3. Call analyze_portfolio_risk with BOTH the portfolio state AND the fetched historical data
                    risk_analysis = system.risk_manager.analyze_portfolio_risk(
                        portfolio=portfolio_for_risk,
                        historical_data=portfolio_historical_data
                    )

                    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
                    with r_col1: st.metric(translator.t('max_single_position_ratio', fallback="最大单一持仓 %"),
                                           f"{risk_analysis.get('diversification', {}).get('max_single_exposure', 0):.1%}")
                    with r_col2: st.metric(translator.t('cash_ratio', fallback="现金比例"),
                                           f"{risk_analysis.get('risk_metrics', {}).get('cash_ratio', 0):.1%}")
                    with r_col3:
                        st.metric(translator.t('concentration_score', fallback="行业集中度"),
                                  f"{risk_analysis.get('diversification', {}).get('concentration_score', 0):.0f}/100")
                    with r_col4:
                        st.metric(translator.t('diversification_score', fallback="多样化得分"),
                                  f"{risk_analysis.get('risk_metrics', {}).get('diversification_score', 0):.0f}/100")

                    st.subheader(translator.t('value_at_risk_var', fallback="风险价值 (VaR) / CVaR"))

                    # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
                    var_cvar_data = risk_analysis.get('var_cvar', {})  # Get the sub-dictionary
                    confidence_level_str = var_cvar_data.get('confidence_level', '95%')

                    var_col1, var_col2 = st.columns(2)
                    with var_col1:
                        st.metric(
                            label=translator.t('daily_var_label', fallback="日 VaR ({conf})").format(
                                conf=confidence_level_str),
                            value=f"{var_cvar_data.get('var_1day_pct', 'N/A')}"
                        )
                    with var_col2:
                        st.metric(
                            label=translator.t('daily_cvar_label', fallback="日 CVaR ({conf})").format(
                                conf=confidence_level_str),
                            value=f"{var_cvar_data.get('cvar_1day_pct', 'N/A')}",
                            help=translator.t('cvar_help_text',
                                              fallback="条件风险价值：衡量在发生极端亏损时，预期的平均亏损程度。")
                        )

                    st.subheader(translator.t('portfolio_optimization_suggestions', fallback="投资组合优化建议"))
                    suggestions = risk_analysis.get('suggestions', [])
                    if suggestions:
                        for suggestion in suggestions:
                            st.info(suggestion)
                    else:
                        st.success(translator.t('portfolio_risk_reasonable', fallback="投资组合风险分布当前看起来合理。"))
                except Exception as e_risk:
                    logger.error(f"执行风险分析时出错: {e_risk}", exc_info=True)
                    st.warning(translator.t('warning_risk_analysis_failed', fallback="风险分析失败。"))
            else: # risk_manager not available
                st.warning(translator.t('warning_risk_manager_unavailable', fallback="风险管理模块不可用。"))

            # --- Investment Portfolio History ---
            st.subheader(translator.t('portfolio_history', fallback="投资组合历史"))
            portfolio_history = st.session_state.get('portfolio_history', [])
            if portfolio_history:
                try:
                    history_df = pd.DataFrame(portfolio_history)
                    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                    history_df = history_df.set_index('timestamp')

                    if 'total_value' not in history_df.columns:
                        logger.debug("Portfolio history missing total_value, calculating on the fly...")
                        values_to_plot = []
                        for entry in portfolio_history:
                            value = entry.get('cash', 0)
                            for pos_data in entry.get('positions', {}).values():
                                q = pos_data.get('quantity', 0); p = pos_data.get('current_price', 0)
                                if isinstance(q, (int, float)) and isinstance(p, (int, float)): value += q * p
                            values_to_plot.append(value)
                        history_df['total_value'] = values_to_plot

                    if 'total_value' in history_df.columns and not history_df['total_value'].empty:
                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Scatter(x=history_df.index, y=history_df['total_value'], mode='lines', name=translator.t('portfolio_value_label', fallback='投资组合净值')))
                        fig_hist.update_layout(title=translator.t('portfolio_value_history_title', fallback="投资组合净值历史"), xaxis_title=translator.t('date', fallback="日期"), yaxis_title=translator.t('net_value_label', fallback="净值 ($)"), height=400)
                        st.plotly_chart(fig_hist, use_container_width=True)
                    else: st.info(translator.t('no_portfolio_history_data', fallback="暂无有效的投资组合历史数据可供绘图。"))
                except Exception as e_hist:
                    logger.error(f"绘制投资组合历史图表时出错: {e_hist}", exc_info=True)
                    st.warning(translator.t('error_rendering_history_chart', fallback="无法绘制投资组合历史图表。"))
            else: # No history data
                st.info(translator.t('no_portfolio_history_data', fallback="暂无投资组合历史数据。"))

        # --- This is the overall except block for render_portfolio_tab ---
        except Exception as e: # Catch any unexpected error during the rendering of this tab
            st.error(f"渲染投资组合主标签页时发生严重错误: {e}")
            logger.error("渲染投资组合主标签页时发生严重错误", exc_info=True)
            st.code(traceback.format_exc())

    def render_analysis_tab(self, system):
        """渲染分析标签页 (确保传递最新数据)"""
        try:
            if hasattr(system, 'portfolio_analyzer'):
                # 从 session_state 获取最新的 portfolio 和 trades 数据
                latest_portfolio = st.session_state.get('portfolio', {})
                latest_trades = st.session_state.get('trades', [])
                latest_history = st.session_state.get('portfolio_history', [])

                # 将最新数据传递给分析器
                system.portfolio_analyzer.render_portfolio_analysis_ui(
                    latest_portfolio,
                    latest_trades,
                    latest_history
                )
            else:
                st.warning("分析模块 (PortfolioAnalyzer) 未初始化或未加载。")
        except Exception as e:
            st.error(f"渲染分析标签页时出错: {e}")
            st.code(traceback.format_exc())


    def render_alert_tab(self, system):
        """渲染报警标签页 (依赖 AlertManager, 使用翻译)"""
        logger.debug("Rendering alert tab...")
        # vvvvvvvvvvvvvvvvvvvv START OF MODIFIED SECTION vvvvvvvvvvvvvvvvvvvv
        try:
            # Alerts tab header is usually handled by AlertManager.render_alerts_ui,
            # but if not, you can add it here:
            # st.header(translator.t('alerts', fallback="报警系统"))

            if hasattr(system, 'alert_manager') and \
               system.alert_manager is not None and \
               hasattr(system.alert_manager, 'render_alerts_ui') and \
               callable(system.alert_manager.render_alerts_ui):

                 logger.debug("Calling system.alert_manager.render_alerts_ui...")
                 # AlertManager's UI method should handle its internal translations
                 system.alert_manager.render_alerts_ui()
            else:
                 # Fallback header if the main alerts UI cannot be rendered
                 st.header(translator.t('price_alert_system', fallback="价格报警系统")) # Generic key
                 st.warning(translator.t('warning_alert_system_unavailable', fallback="报警系统 (AlertManager) 未初始化或未加载。"))
                 logger.warning("UIManager: AlertManager or its render_alerts_ui method unavailable.")
        except Exception as e:
            st.error(translator.t('error_rendering_alerts_tab', fallback="渲染报警标签页时出错:") + f" {e}") # New key
            st.code(traceback.format_exc())
            logger.error("渲染报警标签页时出错", exc_info=True)