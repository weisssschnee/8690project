import streamlit as st
import logging
import time
from datetime import datetime, timedelta
import importlib
import os
from dotenv import load_dotenv
import sys
from pathlib import Path
import asyncio
from typing import Dict, Optional, List, Any
import pandas as pd         # <--- 添加
import traceback            # <--- 添加
import newsapi
import requests


# 导入核心组件
from core.config import Config
from core.analysis.portfolio import PortfolioAnalyzer
from core.analysis.technical import TechnicalAnalyzer
from core.ui.manager import UIManager
from core.utils.data_mock import create_demo_data
from core.data.manager import DataManager
from core.alert.manager import AlertManager
from core.strategy.unified_strategy import UnifiedStrategy
from core.translate import translator
from core.risk.manager import RiskManager  # <--- 确保 RiskManager 导入
from core.trading.executor import OrderExecutor # <--- 正确导入 OrderExecutor
from core.market.scanner import MarketScanner
from core.utils.persistence_manager import PersistenceManager
from core.data.batch_fetcher import BatchFetcher
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def load_environment():
    env_file = ".env.txt"  # 注意文件名包含.txt后缀
    if os.path.exists(env_file):
        load_dotenv(env_file)
        print(f"已加载环境变量文件: {env_file}")
        # 验证变量是否正确加载
        print(f"TUSHARE_TOKEN: {os.environ.get('TUSHARE_TOKEN', '未设置')[:5]}...") # 示例性打印
    else:
        print(f"FINNHUB_KEY 已加载: {os.environ.get('FINNHUB_KEY', '未设置')[:5]}...")  # 打印前5位
        print(f"警告: 环境变量文件不存在: {env_file}")

if __name__ == "__main__":
    load_environment() # 在主程序执行时加载环境变量


class TradingSystem:
    """Trading System Core Class"""

    def __init__(self):
        """Initialize the entire trading system and its components."""
        init_start_time = time.time()
        logger.info("Initializing TradingSystem...")

        # 1. 加载配置
        try:
            self.config = Config()
            logger.info(f"Config loaded successfully. Log file: {getattr(self.config, 'LOG_FILE', 'N/A')}")
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to load Config: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize system configuration: {e}") from e



        # 2. 初始化 UI 管理器
        try:
            self.ui = UIManager()
            logger.info("UIManager initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize UIManager: {e}", exc_info=True)
            self.ui = None

        # 3. 初始化 FutuManager 作为核心服务 (如果启用)
        self.futu_manager = None
        if getattr(self.config, 'FUTU_ENABLED', False):
            try:
                from core.data.manager import FutuManager  # Assuming FutuManager is in data.manager
                self.futu_manager = FutuManager(self.config)
                # Start the connection process in a background thread here, managed by the system
                futu_connect_thread = threading.Thread(target=self.futu_manager.connect, daemon=True)
                futu_connect_thread.start()
                logger.info("FutuManager instance created and connection process started.")
            except ImportError:
                logger.warning("FutuManager class not found. Futu integration disabled.")
            except Exception as e:
                logger.error(f"Failed to create and start FutuManager instance: {e}", exc_info=True)
                self.futu_manager = None
        else:
            logger.info("Futu integration is disabled in the config.")

        db_path = self.config.DATA_PATH / "trading_system.db"
        self.persistence_manager = PersistenceManager(db_path=db_path)

        # 4. 初始化 DataManager (它会接收 FutuManager 实例, 无论是否为 None)
        try:
            self.technical_analyzer = TechnicalAnalyzer(self.config)
            logger.info("TechnicalAnalyzer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize TechnicalAnalyzer: {e}", exc_info=True)
            self.technical_analyzer = None

        try:
            self.data_manager = DataManager(config=self.config, futu_manager=self.futu_manager)
            logger.info("DataManager initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize DataManager: {e}", exc_info=True)
            self.data_manager = None
            st.warning("数据管理器初始化失败，数据相关功能可能受限。")

        # 5. 初始化其他核心组件
        self.components = {}
        if self.data_manager: self.components['data_manager'] = self.data_manager
        if self.futu_manager: self.components['futu_manager'] = self.futu_manager

        # 初始化风险管理器
        try:
            self.risk_manager = RiskManager(self.config)
            self.components['risk_manager'] = self.risk_manager
            logger.info("RiskManager initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize RiskManager: {e}", exc_info=True)
            self.risk_manager = None

        # 初始化订单执行器 (现在它可以接收 futu_manager)
        try:
            self.order_executor = OrderExecutor(config=self.config, risk_manager=self.risk_manager,
                                                futu_manager=self.futu_manager)
            self.components['order_executor'] = self.order_executor
            logger.info("OrderExecutor initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize OrderExecutor: {e}", exc_info=True)
            self.order_executor = None

        # 初始化市场扫描器
        try:
            scanner_config = self.config.get("SCREENER_CONFIG", {})
            self.market_scanner = MarketScanner(scanner_config)
            # 将 DataManager 实例传递给 Scanner
            if hasattr(self.market_scanner, 'set_data_manager'):
                self.market_scanner.set_data_manager(self.data_manager)
            self.components['market_scanner'] = self.market_scanner
            logger.info("MarketScanner initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize MarketScanner: {e}", exc_info=True)
            self.market_scanner = None

        # 初始化报警和分析模块
        try:
            self.alert_manager = AlertManager(self); logger.info("AlertManager initialized.")
        except Exception as e:
            logger.error(f"Failed to init AlertManager: {e}"); self.alert_manager = None

        try:
            self.portfolio_analyzer = PortfolioAnalyzer(self)
            logger.info("PortfolioAnalyzer initialized.")
        except Exception as e:
            logger.error(f"Failed to init PortfolioAnalyzer: {e}"); self.portfolio_analyzer = None

        # 初始化策略管理器
        try:
            self.technical_analyzer = TechnicalAnalyzer(self.config)
            logger.info("TechnicalAnalyzer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize TechnicalAnalyzer: {e}", exc_info=True)
            self.technical_analyzer = None

            # --- 初始化策略管理器 (它会需要 technical_analyzer) ---
        try:
            logging.info("开始初始化 UnifiedStrategy...")

            # 检查前置条件
            if not self.config:
                raise RuntimeError("Config is not initialized")
            if not self.data_manager:
                raise RuntimeError("DataManager is not initialized")
            if not self.technical_analyzer:
                raise RuntimeError("TechnicalAnalyzer is not initialized")

            # UnifiedStrategy 的 __init__ 会使用 self.technical_analyzer
            self.strategy_manager = UnifiedStrategy(self)

            # 检查 LLM traders 是否正确初始化
            if hasattr(self.strategy_manager, 'llm_traders'):
                llm_count = len(self.strategy_manager.llm_traders)
                llm_names = list(self.strategy_manager.llm_traders.keys())
                logging.info(f"✅ UnifiedStrategy 初始化成功！LLM Traders: {llm_count} 个可用: {llm_names}")
            else:
                logging.warning("⚠️ UnifiedStrategy 创建成功但没有 llm_traders 属性")

        except Exception as e:
            logging.error(f"❌ UnifiedStrategy 初始化失败: {e}", exc_info=True)

            # 使用增强的 fallback
            logging.info("🔄 创建增强的 SimpleStrategyManager 作为备用...")
            self.strategy_manager = SimpleStrategyManager(self, self.persistence_manager)

            # 检查 fallback 的 LLM traders
            if hasattr(self.strategy_manager, 'llm_traders'):
                fallback_llm_count = len(self.strategy_manager.llm_traders)
                fallback_llm_names = list(self.strategy_manager.llm_traders.keys())
                logging.info(
                    f"🔧 Fallback策略管理器创建完成。LLM Traders: {fallback_llm_count} 个可用: {fallback_llm_names}")

            # Fallback
            class SimpleStrategyManager:
                def __init__(self, system: Any, persistence_manager: Any):
                    self.system = system
                    self.persistence_manager = persistence_manager
                    # 添加空的 llm_traders 字典以避免 AttributeError
                    self.llm_traders = {}

                    # 尝试简单地初始化一些LLM适配器
                    try:
                        from core.strategy.llm_trader_adapters import GeminiTraderAdapter, DeepSeekTraderAdapter

                        # 检查Gemini配置
                        gemini_key = getattr(system.config, 'GEMINI_API_KEY', None)
                        gemini_model = getattr(system.config, 'GEMINI_DEFAULT_MODEL', 'gemini-2.5-flash')
                        if gemini_key:
                            self.llm_traders['Gemini'] = GeminiTraderAdapter(api_key=gemini_key,
                                                                             model_name=gemini_model)
                            logging.info(f"✅ Fallback: Successfully initialized Gemini trader")

                        # 检查DeepSeek配置
                        deepseek_key = getattr(system.config, 'DEEPSEEK_API_KEY', None)
                        deepseek_model = getattr(system.config, 'DEEPSEEK_DEFAULT_MODEL', 'deepseek-reasoner')
                        if deepseek_key:
                            self.llm_traders['DeepSeek'] = DeepSeekTraderAdapter(api_key=deepseek_key,
                                                                                 model_name=deepseek_model)
                            logging.info(f"✅ Fallback: Successfully initialized DeepSeek trader")

                    except Exception as e:
                        logging.error(f"Fallback strategy manager failed to initialize LLM traders: {e}")
                        self.llm_traders = {}

                def render_strategy_ui(self, system):
                    import streamlit as st
                    from core.translate import translator
                    st.header(translator.t('strategy_trading'))
                    st.warning(translator.t('warning_strategy_manager_load_failed',
                                            fallback="⚠️ 完整的策略管理器加载失败，正在使用简化版本。ML功能可能受限，但LLM交易员应该可用。"))

                def get_signal_for_autotrader(self, config):
                    logging.error("Attempted to get autotrader signal for ML model in fallback mode.")
                    return {'message': 'ML Strategy manager is not available in fallback mode.'}

                def get_llm_trader_signal(self, config, contextual_data):
                    """处理LLM交易员信号"""
                    llm_name = config.get('llm_name')
                    if not llm_name or llm_name not in self.llm_traders:
                        return {'error': f'LLM trader "{llm_name}" not available in fallback mode.'}

                    try:
                        # 构建简化的prompt context
                        symbol = config.get('symbol', 'UNKNOWN')
                        prompt_context = f"""
                        As an AI trading advisor, analyze {symbol} and provide a trading decision.

                        Recent data: {contextual_data.get('historical_data', 'No data available')}
                        News: {contextual_data.get('news', 'No news available')}

                        Respond with JSON format:
                        {{
                          "decision": "BUY/SELL/HOLD",
                          "confidence": 0.75,
                          "reasoning": "Your analysis here"
                        }}
                        """

                        adapter = self.llm_traders[llm_name]
                        return adapter.get_decision(prompt_context)

                    except Exception as e:
                        logging.error(f"Error in fallback LLM trader signal: {e}")
                        return {'error': f'Fallback LLM trader error: {e}'}

        # 6. 初始化 Session State
        self._init_session_state()

        # 7. 创建模拟数据
        try:
            self.demo_data = create_demo_data()
            logger.info(f"Demo data created for {len(self.demo_data)} symbols.")
        except Exception as e:
            logger.error(f"Failed to create demo data: {e}", exc_info=True);
            self.demo_data = {}

        try:
            if self.strategy_manager and self.strategy_manager.ml_strategy_instance:
                self.text_feature_extractor = self.strategy_manager.ml_strategy_instance.text_feature_extractor
                logger.info("Shortcut to TextFeatureExtractor created on TradingSystem instance.")
            else:
                self.text_feature_extractor = None
                logger.warning(
                    "Could not create shortcut to TextFeatureExtractor because StrategyManager or MLStrategy is not available.")
        except AttributeError:
            self.text_feature_extractor = None
            logger.warning("AttributeError while creating shortcut to TextFeatureExtractor.")

        self.batch_fetcher = BatchFetcher(data_manager=self.data_manager)

        init_duration = time.time() - init_start_time
        logger.info(f"Trading system initialization completed in {init_duration:.2f} seconds.")

    if 'central_data_cache' not in st.session_state:
        st.session_state.central_data_cache = {}

    def get_cached_data(self, key: str, fetch_func, *args, **kwargs):
        """
        [新增] 一个通用的、用于获取和缓存数据的辅助方法。
        """
        if key not in st.session_state.central_data_cache:
            # 如果缓存中没有，则调用实际的获取函数
            st.session_state.central_data_cache[key] = fetch_func(*args, **kwargs)

        return st.session_state.central_data_cache[key]

        # --- 修改现有的数据获取方法，让它们使用这个缓存 ---

    def get_stock_data(self, symbol, days=90, interval="1d"):
        """获取历史数据 (现在通过中央缓存)**"""
        cache_key = f"hist_{symbol}_{days}_{interval}"
        # 使用 get_cached_data 来获取，如果不存在，则调用 data_manager 的方法
        return self.get_cached_data(
            cache_key,
            self.data_manager.get_historical_data,
            symbol, days=days, interval=interval
        )

    def get_stock_details(self, symbol):
        """获取公司详情 (现在通过中央缓存)**"""
        cache_key = f"details_{symbol}"
        return self.get_cached_data(
            cache_key,
            self.data_manager.get_stock_details,
            symbol
        )

    def _init_components(self):
        """初始化系统组件"""
        components_config = {
            "data_manager": {"class": "core.data.manager.DataManager", "args": [self.config]},
            "risk_manager": {"class": "core.risk.manager.RiskManager", "args": [self.config]},
            "order_executor": {"class": "core.trading.executor.OrderExecutor", "args": [self.config]},
            "market_scanner": {"class": "core.market.scanner.MarketScanner",
                               "args": [self.config.get("scanner_settings", {})]},
            "custom_strategy": {"class": "core.strategy.custom.CustomStrategy", "args": []},
            "ml_strategy": {"class": "core.strategy.ml.MLStrategy", "args": [self.config]}
        }

    def _init_session_state(self):
        """[可选登录版] 初始化 session state，总是从默认模拟账户开始。"""
        if 'logged_in' not in st.session_state:
            st.session_state.logged_in = False
            st.session_state.username = "Guest" # 游客

        if 'portfolio' not in st.session_state:
            current_user = st.session_state.username
            portfolio_from_db = self.persistence_manager.load_portfolio(user_id=current_user)  # load 时传递 user_id

            if portfolio_from_db:
                st.session_state.portfolio = portfolio_from_db
            else:
                self.reset_to_default_portfolio()
                # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
                # 创建并保存初始投资组合时，必须提供 user_id
                self.persistence_manager.save_portfolio(
                    portfolio=st.session_state.portfolio,
                    user_id=current_user
                )

        if 'trades' not in st.session_state:
            current_user = st.session_state.username
            st.session_state.trades = self.persistence_manager.load_trades(user_id=current_user)  # load 时传递 user_id

        # --- 投资组合历史 (这个是易失性的，每次会话重新计算) ---
        if 'portfolio_history' not in st.session_state:
            current_user = st.session_state.get('username', 'Guest')
            # 从数据库加载最近 N 条历史记录
            st.session_state.portfolio_history = self.persistence_manager.load_portfolio_history(user_id=current_user,
                                                                                                 limit=1000)

        # --- 其他 UI 状态 (保持不变) ---
        if 'auto_refresh' not in st.session_state: st.session_state.auto_refresh = False
        if 'refresh_interval' not in st.session_state: st.session_state.refresh_interval = 60
        if 'last_refresh_time' not in st.session_state: st.session_state.last_refresh_time = time.time()
        # (确保添加了所有其他 session state 变量的初始化)

    def reset_to_default_portfolio(self):
        """[最终版] 将 session_state 重置为默认的游客模拟账户"""
        st.session_state.portfolio = {
            'cash': 100000.0, 'positions': {},
            'total_value': 100000.0, 'last_update': datetime.now().isoformat()
        }
        st.session_state.trades = []
        st.session_state.portfolio_history = []  # 也清空历史
        logger.info("Session state has been reset to the default guest portfolio.")



    def _update_portfolio(self, order_data, execution_result):
            """
            [最终版] 更新投资组合状态，并将变更写入 session_state 和持久化数据库。
            """
            if not execution_result or not execution_result.get('success'):
                logger.warning("_update_portfolio skipped: execution failed or result invalid.")
                return

            # --- 1. 从 session_state 获取当前投资组合的副本 ---
            portfolio = st.session_state.get('portfolio', {}).copy()
            positions = portfolio.setdefault('positions', {})

            # --- 2. 提取交易信息 ---
            symbol = order_data.get('symbol')
            quantity_from_order = order_data.get('quantity')
            direction = order_data.get('direction', 'Buy')
            exec_price = execution_result.get('price', order_data.get('price'))
            cost_or_proceeds = execution_result.get('total_cost', quantity_from_order * (exec_price or 0))
            commission = execution_result.get('commission', 0)

            if None in [symbol, quantity_from_order, exec_price]:
                logger.error(f"更新投资组合失败：缺少关键信息。Order={order_data}, Result={execution_result}")
                return

            logger.info(f"Updating portfolio: {direction} {quantity_from_order} {symbol} @ {exec_price:.2f}")

            # --- 3. 计算新的现金和持仓 ---
            # (这部分计算逻辑与您提供的代码完全相同，保持不变)
            if direction == "Buy":
                portfolio['cash'] -= (cost_or_proceeds + commission)
                position = positions.setdefault(symbol, {'quantity': 0, 'cost_basis': 0})
                old_quantity, old_cost_basis = position.get('quantity', 0), position.get('cost_basis', 0)
                new_quantity = old_quantity + quantity_from_order
                position['cost_basis'] = ((old_quantity * old_cost_basis) + (
                            quantity_from_order * exec_price)) / new_quantity if new_quantity != 0 else 0
                position['quantity'] = new_quantity
            elif direction == "Sell":
                portfolio['cash'] += (cost_or_proceeds - commission)
                if symbol in positions:
                    position = positions[symbol]
                    old_quantity = position.get('quantity', 0)
                    quantity_to_sell = min(quantity_from_order, old_quantity)
                    if quantity_to_sell < quantity_from_order: logger.warning(
                        f"Attempted to sell more than held for {symbol}.")
                    position['quantity'] -= quantity_to_sell
                    if position['quantity'] <= 1e-9:
                        del positions[symbol]
                else:
                    logger.error(f"Attempted to update a non-existent sell position for {symbol}.")

            # --- 4. 更新辅助信息和重新计算总价值 ---
            if symbol in positions:
                positions[symbol]['current_price'] = exec_price
            portfolio['last_update'] = datetime.now().isoformat()
            current_total_value = portfolio['cash']
            for pos_data in positions.values():
                current_total_value += pos_data.get('quantity', 0) * pos_data.get('current_price', 0)
            portfolio['total_value'] = current_total_value

            # --- 5. 构建新的交易和历史记录 ---
            new_trade_record = {
                'symbol': symbol, 'direction': direction, 'quantity': quantity_from_order,
                'price': exec_price, 'timestamp': execution_result.get('timestamp', datetime.now()),
                'total': cost_or_proceeds, 'commission': commission,
                'is_mock': execution_result.get('is_mock', True)
            }
            new_history_entry = {
                'timestamp': datetime.now(),
                'cash': portfolio['cash'],
                'positions': {s: p.copy() for s, p in portfolio.get('positions', {}).items()},
                'total_value': portfolio['total_value']
            }

            # --- 6. 将所有更新持久化 ---
            # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
            # a. 在使用前，从 session_state 获取当前用户 ID
            current_user = st.session_state.get('username', 'Guest')
            if not current_user:
                logger.error("无法持久化交易：未找到当前用户。")
                # 即使没有用户，也应该更新内存中的状态
                st.session_state.portfolio = portfolio
                st.session_state.trades.append(new_trade_record)
                st.session_state.portfolio_history.append(new_history_entry)
                return

            # b. 更新内存中的 session_state
            st.session_state.portfolio = portfolio
            st.session_state.trades.append(new_trade_record)
            st.session_state.portfolio_history.append(new_history_entry)

            # c. 将变更写入数据库，并传入 current_user
            self.persistence_manager.save_portfolio(portfolio, user_id=current_user)
            self.persistence_manager.add_trade(new_trade_record, user_id=current_user)
            self.persistence_manager.add_portfolio_history_entry(new_history_entry, user_id=current_user)
            # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^

            logger.info(
                f"投资组合为用户 '{current_user}' 更新并已持久化。现金: {portfolio['cash']:.2f}, 总价值: {portfolio['total_value']:.2f}")

    def reset_user_account(self) -> bool:
        """
        [新增] 重置当前已登录用户的持久化账户。
        这是一个更具破坏性的操作。
        """
        if not st.session_state.get('logged_in'):
            logger.warning("Attempted to reset user account, but no user is logged in.")
            return False

        username = st.session_state.get('username')
        logger.warning(f"PERFORMING FULL ACCOUNT RESET FOR USER: {username}")

        # 1. 创建初始投资组合
        initial_portfolio = {
            'cash': 100000.0, 'positions': {},
            'total_value': 100000.0, 'last_update': datetime.now().isoformat()
        }
        # 2. 更新 session_state
        st.session_state.portfolio = initial_portfolio
        st.session_state.trades = []
        st.session_state.portfolio_history = []

        # 3. 将新的初始状态写入数据库 (覆盖)
        self.persistence_manager.save_portfolio(
            portfolio=initial_portfolio,
            user_id=username
        )

        # 4. 清空该用户的交易历史表
        self.persistence_manager.clear_trades(user_id=username)
        logger.info(f"Account state and trade history for user '{username}' have been reset in the database.")
        return True

    def _create_mock_component(self, name):
        """创建模拟组件"""

        class MockComponent:
            def __init__(self):
                self.name = name
                logger.info(f"创建了 {name} 的模拟实现")

            def __getattr__(self, attr):
                def mock_method(*args, **kwargs):
                    logger.warning(f"{name}.{attr} 被调用，但此方法仅为模拟实现")
                    return {}

                return mock_method

        return MockComponent()

    def _setup_component_relationships(self):
        """设置组件之间的关联"""
        try:
            # 为custom_strategy设置依赖组件
            if all(k in self.components for k in
                   ["custom_strategy", "market_scanner", "technical_analyzer", "sentiment_analyzer"]):
                cs = self.components["custom_strategy"]
                cs.set_scanner(self.components["market_scanner"])
                cs.set_technical_analyzer(self.components["technical_analyzer"])
                cs.set_sentiment_analyzer(self.components["sentiment_analyzer"])
                logger.info("已设置custom_strategy的组件关联")
        except Exception as e:
            logger.error(f"设置组件关联失败: {e}")

    def _init_state(self):
        """初始化应用状态"""
        # 账户和投资组合
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = {
                'cash': 100000.0,
                'positions': {},
                'total_value': 100000.0,
                'last_update': datetime.now()
            }

        # 交易记录
        if 'trades' not in st.session_state:
            st.session_state.trades = []

        # 投资组合历史
        if 'portfolio_history' not in st.session_state:
            st.session_state.portfolio_history = []

        # 自动刷新设置
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = 60
        if 'last_refresh_time' not in st.session_state:
            st.session_state.last_refresh_time = time.time()

    def run(self):
        """运行系统UI界面，并将所有渲染职责交给 UIManager。"""


        # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
        # 移除此处的UI渲染调用，让 UIManager 全权负责
        # self.ui.render_refresh_controls()

        # UIManager 自己会决定在何处、如何渲染所有组件
        self.ui.render_sidebar(self)
        self.ui.render_main_tabs(self)
        # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^

        # 执行应用内自动交易逻辑 (这部分不变)
        self._run_in_app_autotrader()

        # 处理UI的自动刷新 (这部分不变)
        self._handle_smart_refresh()

    def render_autotrader_controls_in_sidebar(self):
        """
        [补全实现] 在侧边栏渲染自动交易引擎的总开关。
        这个方法由 ui_manager.py 中的 render_sidebar 调用。
        """
        st.sidebar.markdown("---")
        st.sidebar.header("🤖 " + translator.t('autotrader_engine_title', fallback="自动交易引擎"))

        # 初始化 session_state 中的总开关状态
        if "autotrader_engine_enabled" not in st.session_state:
            st.session_state.autotrader_engine_enabled = False

        # 渲染开关
        is_enabled = st.sidebar.toggle(
            translator.t('enable_autotrader_engine_label', fallback="开启自动交易引擎"),
            value=st.session_state.autotrader_engine_enabled,
            key="autotrader_engine_master_toggle",
            help=translator.t('autotrader_engine_help',
                              fallback="开启后，此浏览器页面将定期扫描并执行所有已启用的自动化策略。关闭此标签页将停止引擎。")
        )

        # 将开关状态同步回 session_state
        st.session_state.autotrader_engine_enabled = is_enabled

        # 根据状态显示不同的反馈信息
        if is_enabled:
            last_run_ts = st.session_state.get("autotrader_last_run_time")
            if last_run_ts:
                last_run_dt = datetime.fromtimestamp(last_run_ts)
                time_ago = datetime.now() - last_run_dt
                if time_ago.total_seconds() < 120:  # 两分钟内认为是正常的
                    status_text = f"✅ {translator.t('autotrader_status_running', fallback='引擎运行中')} (上次扫描: {int(time_ago.total_seconds())} 秒前)"
                else:
                    status_text = f"⚠️ {translator.t('autotrader_status_stale', fallback='引擎可能已暂停')} (上次扫描: {int(time_ago.total_seconds() / 60)} 分钟前)"
            else:
                status_text = f"✅ {translator.t('autotrader_status_pending', fallback='引擎已开启，等待首次扫描...')}"

            st.sidebar.caption(status_text)
        else:
            st.sidebar.caption(f"❌ {translator.t('autotrader_status_stopped', fallback='引擎已停止。')}")

        # in core/system.py -> class TradingSystem

    def _run_in_app_autotrader(self):
            """
            [最终完整版] 应用内自动交易调度器。
            - 修复了变量引用错误。
            - 包含完整的任务收集、数据预热、任务分发、交易决策和执行逻辑。
            """
            # 1. 检查总开关状态
            if not st.session_state.get("autotrader_engine_enabled", False):
                return

            # 2. 检查距离上次调度是否超过最小间隔
            now = time.time()
            last_run = st.session_state.get("autotrader_last_run_time", 0)
            interval = getattr(self.config, 'AUTOTRADER_INTERVAL_SECONDS', 60)

            if now - last_run < interval:
                return

            # 更新时间戳，防止下次刷新时立即再次运行
            st.session_state.autotrader_last_run_time = now

            st.toast("🤖 自动交易引擎正在扫描策略...")
            logger.info("[In-App AutoTrader] Tick Start: Scanning for enabled strategies...")

            try:
                # --- 3. 收集任务 ---
                enabled_strategies = self.persistence_manager.load_enabled_auto_strategies()
                if not enabled_strategies:
                    logger.info("[In-App AutoTrader] No enabled strategies found. Tick finished.")
                    return

                # 提取所有需要监控的唯一股票代码
                symbols_to_monitor = set()
                for s in enabled_strategies:
                    symbols_in_strat = s.get('symbols', [s.get('symbol')])
                    for symbol in symbols_in_strat:
                        if symbol: symbols_to_monitor.add(symbol)

                symbols_to_monitor = sorted(list(symbols_to_monitor))
                if not symbols_to_monitor:
                    logger.info(
                        "[In-App AutoTrader] No symbols to monitor across all enabled strategies. Tick finished.")
                    return

                logger.info(
                    f"[In-App AutoTrader] Found {len(enabled_strategies)} strategies. Monitoring symbols: {symbols_to_monitor}")

                # --- 4. SYNCHRONOUS Data Pre-heating ---
                logger.info("[In-App AutoTrader] Starting SYNCHRONOUS data pre-heating...")

                # Use a single ThreadPool to fetch both data types concurrently, but wait for completion.
                hist_data_batch = {}
                news_batch = {}

                with st.spinner(f"Pre-heating data for {len(symbols_to_monitor)} symbols..."):
                    with ThreadPoolExecutor(max_workers=self.batch_fetcher.max_workers) as executor:
                        # Submit historical data futures
                        hist_futures = {executor.submit(self.data_manager.get_historical_data, symbol, days=252,
                                                        interval='1d'): symbol for symbol in symbols_to_monitor}

                        # Submit news data futures
                        news_futures = {executor.submit(self.data_manager.get_news, symbol, num_articles=5): symbol for
                                        symbol in symbols_to_monitor}

                        # Process historical data as it completes
                        logger.info("Waiting for historical data fetches to complete...")
                        for future in as_completed(hist_futures):
                            symbol = hist_futures[future]
                            try:
                                data = future.result()
                                if data is not None and not data.empty:
                                    hist_data_batch[symbol] = data
                                    logger.debug(f"Successfully pre-heated historical data for {symbol}.")
                                else:
                                    logger.warning(
                                        f"Historical data fetch returned empty for {symbol} during pre-heating.")
                            except Exception as exc:
                                logger.error(f"Error fetching historical data for {symbol} during pre-heating: {exc}")

                        # Process news data as it completes
                        logger.info("Waiting for news fetches to complete...")
                        for future in as_completed(news_futures):
                            symbol = news_futures[future]
                            try:
                                data = future.result()
                                if data:
                                    news_batch[symbol] = data
                                    logger.debug(f"Successfully pre-heated news for {symbol}.")
                                else:
                                    logger.warning(f"News fetch returned empty for {symbol} during pre-heating.")
                            except Exception as exc:
                                logger.error(f"Error fetching news for {symbol} during pre-heating: {exc}")

                logger.info("[In-App AutoTrader] SYNCHRONOUS data pre-heating complete.")

                # --- 5. 遍历策略并为每个股票执行决策 ---
                for config in enabled_strategies:
                    strategy_id = config.get('strategy_id')
                    user_id = config.get('user_id')
                    core_type = config.get('core_type')  # <-- 从 config 字典中正确获取

                    symbols_in_this_strategy = config.get('symbols', [config.get('symbol')])

                    for symbol in symbols_in_this_strategy:
                        if not symbol: continue
                        logger.info(
                            f"--- Processing strategy: '{strategy_id}' for '{user_id}' on SYMBOL: '{symbol}' ---")

                        # 为当前股票准备上下文数据包
                        contextual_data = {
                            "historical_data": hist_data_batch.get(symbol),
                            "news": news_batch.get(symbol)
                        }

                        # 创建一个针对当前股票的临时配置副本
                        single_symbol_config = config.copy()
                        single_symbol_config['symbol'] = symbol

                        # --- 6. 任务分发：根据核心类型调用不同的信号生成器 ---
                        core_type = config.get('core_type')  # Get the new key

                        # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
                        # If the new key doesn't exist, check the old key for backward compatibility
                        if core_type is None:
                            old_type = config.get('type')
                            logger.warning(
                                f"[{strategy_id}] Strategy is using old format (type: '{old_type}'). Translating for compatibility.")
                            if old_type == 'ml_quant':
                                core_type = 'ml_model'
                            elif old_type == 'llm_trader':  # Assuming you might have saved LLM strategies this way too
                                core_type = 'llm_trader'
                        # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^

                        decision_result = {}
                        if core_type == "ml_model":
                            decision_result = self.strategy_manager.get_signal_for_autotrader(single_symbol_config)
                        elif core_type == "llm_trader":
                            decision_result = self.strategy_manager.get_llm_trader_signal(single_symbol_config,
                                                                                          contextual_data)
                        else:
                            logger.warning(
                                f"[{strategy_id}] Could not determine a valid core_type ('{core_type}'). Skipping symbol {symbol}.")
                            continue

                        if not decision_result or "error" in decision_result or "message" in decision_result:
                            error_msg = decision_result.get('error') or decision_result.get('message',
                                                                                            'Unknown signal generation error')
                            logger.error(f"[{strategy_id}] Failed to get signal for {symbol}: {error_msg}")
                            continue

                        # --- 7. 交易决策 ---
                        if core_type == "ml_model":
                            direction = decision_result.get('direction', -1)
                            final_decision = "BUY" if direction == 1 else ("SELL" if direction == 0 else "HOLD")
                        else:  # llm_trader
                            final_decision = decision_result.get("decision", "HOLD").upper()

                        confidence = decision_result.get("confidence", 0)
                        threshold = config.get("confidence_threshold", 0.7)
                        logger.info(
                            f"[{strategy_id}] Signal for {symbol}: Decision='{final_decision}', Confidence={confidence:.2f}, Threshold={threshold:.2f}")

                        if confidence < threshold:
                            logger.info(
                                f"[{strategy_id}] Decision for {symbol} reverted to HOLD because confidence is below threshold.")
                            final_decision = "HOLD"

                        # 获取最新持仓
                        portfolio = self.persistence_manager.load_portfolio(user_id)
                        has_position = portfolio.get('positions', {}).get(symbol, {}).get('quantity', 0) > 0

                        order_data = None
                        if final_decision == "BUY" and not has_position:
                            order_data = {"symbol": symbol, "quantity": config.get("trade_quantity"),
                                          "direction": "Buy", "order_type": "Market Order", "price": None}
                        elif final_decision == "SELL" and has_position:
                            order_data = {"symbol": symbol, "quantity": portfolio['positions'][symbol]['quantity'],
                                          "direction": "Sell", "order_type": "Market Order", "price": None}

                        # --- 8. 执行交易 ---
                        if order_data:
                            logger.warning(
                                f"[{strategy_id}] ACTION! Executing trade for '{user_id}' on {symbol}: {order_data}")
                            trade_result = self.execute_trade_for_user(order_data, user_id)
                            logger.info(f"[{strategy_id}] Trade execution result for {symbol}: {trade_result}")
                            if st.session_state.get(
                                    'username') == user_id and trade_result.get('success'):
                                st.rerun()
                        else:
                            logger.info(f"[{strategy_id}] No trade action required for {symbol} at this time.")

                    # --- 9. 更新整个策略的心跳（在处理完所有股票后）---
                    self.persistence_manager.update_strategy_last_executed(strategy_id)

            except Exception as e:
                logger.error(f"[In-App AutoTrader] A critical error occurred during the tick: {e}", exc_info=True)

    def __del__(self):
        """在对象被垃圾回收时尝试清理资源"""
        logger.info("TradingSystem instance being deleted. Attempting to stop managers.")
        if hasattr(self, 'futu_manager') and self.futu_manager:
            self.futu_manager.disconnect()

    def _handle_smart_refresh(self):
        """
        智能刷新。当前版本简化为与常规自动刷新相同。
        未来的版本可以基于 WebSocket 推送或事件来触发刷新。
        """
        if st.session_state.get('auto_refresh', False):
            current_time = time.time()
            last_refresh = st.session_state.get('last_refresh_time', 0)
            interval = st.session_state.get('refresh_interval', 60)

            if current_time - last_refresh >= interval:
                logger.debug(f"Auto-refresh triggered by timer (interval: {interval}s)")
                st.session_state.last_refresh_time = current_time
                st.rerun()

    def update_position_price(self, symbol: str, price: float):
        """由回调函数调用，实时更新持仓的当前价格"""
        if 'portfolio' in st.session_state and symbol in st.session_state.portfolio.get('positions', {}):
            st.session_state.portfolio['positions'][symbol]['current_price'] = price
            st.session_state.portfolio['last_update'] = datetime.now()
            # logger.debug(f"Live price update for position {symbol}: {price}")
            # 注意：这个更新不会立即反映在 UI 上，直到下一次 st.rerun()

    # ============= 功能方法 =============

    # 在system.py中修改get_stock_data方法
    def get_stock_data(self, symbol, days=90, interval="1d"):  # <--- 方法名改回 get_stock_data 以匹配 UI 调用
        """获取历史数据 (代理给 DataManager 的新方法)"""
        logger.debug(f"System getting historical data for {symbol}...")
        if self.data_manager:
            # ** 内部调用 get_historical_data **
            return self.data_manager.get_historical_data(symbol, days=days, interval=interval)

        logger.error("DataManager not available, cannot get historical data.")
        if hasattr(self, 'demo_data') and symbol in self.demo_data:
            logger.warning(f"Using demo data for {symbol} as DataManager is unavailable.")
            return self.demo_data[symbol].copy()
        return None



    def get_realtime_price(self, symbol):
        """获取实时价格 (代理给 DataManager)"""
        logger.debug(f"System getting realtime price for {symbol}...")
        if self.data_manager:
            price_data = self.data_manager.get_realtime_price(symbol)
            if price_data and isinstance(price_data.get('price'), (int, float)):
                return price_data['price']

        logger.error("DataManager not available, cannot get realtime price.")
        if hasattr(self, 'demo_data') and symbol in self.demo_data and not self.demo_data[symbol].empty:
            return self.demo_data[symbol]['close'].iloc[-1]
        return None

    def run_market_scan(self, criteria):
        """运行市场扫描 - 尝试调用真实扫描器 (用于调试)"""
        logger.debug(f"run_market_scan called with criteria: {criteria}")
        scanner = self.components.get('market_scanner')

        if scanner and hasattr(scanner, 'scan_market') and callable(scanner.scan_market):
            logger.info("尝试使用 MarketScanner 组件进行扫描...")
            try:
                # --- 尝试同步执行异步方法 ---
                # 注意：这在某些环境（包括Streamlit的某些版本/用法）下可能引发 RuntimeError
                # 如果引发 RuntimeError，说明我们需要更复杂的异步处理方式（如独立线程/服务）
                scan_results = asyncio.run(scanner.scan_market(criteria.get('market', 'US'), criteria))
                # --- 结束尝试 ---

                if isinstance(scan_results, list): # 确保返回的是列表
                     if scan_results:
                         logger.info(f"MarketScanner 返回了 {len(scan_results)} 个结果。")
                         return scan_results
                     else:
                         logger.info("MarketScanner 运行成功，但没有找到符合条件的结果。")
                         # 这里可以选择返回空列表或回退到模拟
                         # return []
                         logger.info("回退到模拟市场扫描 (真实扫描无结果)。")
                         return self._run_mock_market_scan(criteria)
                else:
                     logger.error(f"MarketScanner.scan_market 返回了非列表类型: {type(scan_results)}")
                     logger.info("回退到模拟市场扫描 (真实扫描返回异常)。")
                     return self._run_mock_market_scan(criteria)

            except RuntimeError as e_rt:
                 if "cannot run event loop while another loop is running" in str(e_rt):
                      logger.error("异步错误：无法在 Streamlit 的事件循环中直接运行 asyncio.run()。真实扫描器无法在此模式下执行。")
                 else:
                      logger.error(f"运行 MarketScanner 时发生运行时错误: {e_rt}", exc_info=True)
                 logger.info("回退到模拟市场扫描 (异步执行错误)。")
                 return self._run_mock_market_scan(criteria)
            except Exception as e:
                logger.error(f"运行 MarketScanner 时发生错误: {e}", exc_info=True)
                logger.info("回退到模拟市场扫描 (真实扫描器错误)。")
                return self._run_mock_market_scan(criteria)
        else:
            logger.warning("MarketScanner 未初始化或 scan_market 方法不存在。执行模拟扫描。")
            return self._run_mock_market_scan(criteria)

    def _run_real_market_scan(self, criteria):
        """使用真实组件运行市场扫描"""
        scan_results = []
        # 实际实现...
        return scan_results

    def _run_mock_market_scan(self, criteria):
        """运行模拟市场扫描"""
        results = []

        # 解析标准
        symbols = criteria.get('symbols', '').split(',')
        symbols = [s.strip().upper() for s in symbols if s.strip()]
        market = criteria.get('market', 'US')
        vol_threshold = criteria.get('vol_threshold', 10.0)
        price_threshold = criteria.get('price_threshold', 5.0)
        days = criteria.get('days', 5)

        # 使用默认股票池
        if not symbols:
            if market == "US":
                symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA']
            else:
                symbols = ['600519.SH', '000001.SZ', '600036.SH', '601318.SH']

        # 处理A股代码格式
        if market == "CN":
            symbols = [s if '.' in s else f"{s}.SH" for s in symbols]

        # 扫描股票
        for symbol in symbols:
            if symbol in self.demo_data:
                data = self.demo_data[symbol].copy()
                if len(data) < days:
                    continue

                # 取最近数据
                recent_data = data.iloc[-days:]

                # 安全计算指标
                if recent_data['volume'].iloc[0] != 0:
                    vol_change = ((recent_data['volume'].iloc[-1] / recent_data['volume'].iloc[0]) - 1) * 100
                else:
                    vol_change = 0

                if recent_data['close'].iloc[0] != 0:
                    price_change = ((recent_data['close'].iloc[-1] / recent_data['close'].iloc[0]) - 1) * 100
                else:
                    price_change = 0

                # 应用筛选条件
                if abs(vol_change) >= vol_threshold and abs(price_change) >= price_threshold:
                    results.append({
                        'symbol': symbol,
                        'vol_change': vol_change,
                        'price_change': price_change,
                        'last_price': recent_data['close'].iloc[-1],
                        'data': recent_data
                    })

        return results

    def execute_trade(self, order_data):
        """执行交易 (代理给 OrderExecutor)"""
        logger.debug(f"System executing trade: {order_data}")
        if self.order_executor is None:
            logger.error("Order Executor not initialized. Attempting mock trade.")
            mock_result = self._execute_mock_trade(order_data)
            if mock_result.get('success'):
                self._update_portfolio(order_data, mock_result)
            return mock_result

        try:
            if self.risk_manager:
                portfolio_state = st.session_state.get('portfolio', {})

                # --- 关键修改：将 TradingSystem 实例 (self) 传递给 risk_manager ---
                validation_result = self.risk_manager.validate_order(
                    order=order_data,
                    portfolio=portfolio_state,
                    system_ref=self  # <--- 在这里传递 self
                )

            result = self.order_executor.execute_order(
                symbol=order_data.get('symbol'),
                quantity=order_data.get('quantity'),
                price=order_data.get('price'),
                order_type=order_data.get('order_type'),
                direction=order_data.get('direction')
            )
            if result and result.get('success'):
                self._update_portfolio(order_data, result)
            return result
        except Exception as e:
            logger.error(f"Error during trade execution pipeline: {e}", exc_info=True)
            return {'success': False, 'message': f"交易执行出错: {e}"}


    def _execute_mock_trade(self, order_data):
        """模拟订单执行"""
        symbol = order_data['symbol']
        quantity = order_data['quantity']
        price = order_data['price']
        direction = order_data['direction']
        portfolio_snapshot = st.session_state.get('portfolio', {}) # 获取最新的portfolio快照
        current_positions = portfolio_snapshot.get('positions', {})

        # 获取当前语言设置
        lang = st.session_state.get("lang", "zh")

        # 检查市价单逻辑
        if order_data['order_type'] == "Market Order":
            current_price = self.get_realtime_price(symbol)
            if current_price is None:
                return {
                    'success': False,
                    'message': translator.t("price_fetch_failed", symbol=symbol)
                }
            price = current_price

        # 计算交易总额
        total_cost = abs(quantity) * price

        # 资金检查
        if direction == "Buy" and total_cost > st.session_state.portfolio['cash']:
            return {
                'success': False,
                'message': translator.t("insufficient_funds")
            }

        # 持仓检查
        if direction == "Sell":
            position_details = current_positions.get(symbol, {}) # 从快照获取
            held_quantity = position_details.get('quantity', 0)
            if quantity > held_quantity: # quantity 是用户输入的卖出量
                msg = translator.t("error_insufficient_position_mock", ...).format(needed=quantity, available=held_quantity)
                logger.warning(msg)
                return {'success': False, 'message': msg}
        # 返回成功结果
        return {
            'success': True,
            'symbol': symbol,
            'quantity': quantity if direction == "Buy" else -quantity,
            'price': price,
            'total_cost': total_cost,
            'timestamp': datetime.now(),
            'message': translator.t("order_success", direction=direction)
        }

    def reset_account(self):  # <--- 新增一个重置账户的方法
        """重置账户到初始状态并清空数据库"""
        logger.warning("PERFORMING FULL ACCOUNT RESET.")
        # 1. 创建初始投资组合
        initial_portfolio = {
            'cash': 100000.0, 'positions': {},
            'total_value': 100000.0, 'last_update': datetime.now()
        }
        # 2. 更新 session_state
        st.session_state.portfolio = initial_portfolio
        st.session_state.trades = []
        st.session_state.portfolio_history = []

        # 3. 将新的初始状态写入数据库 (覆盖)
        self.persistence_manager.save_portfolio(initial_portfolio)

        # 4. 清空交易历史表
        with self.persistence_manager._get_conn() as conn:
            conn.execute("DELETE FROM trade_history")
            conn.commit()
        logger.info("Account state and trade history have been reset in the database.")

    def analyze_sentiment(self, symbol):
        """分析情绪"""
        try:
            if "sentiment_analyzer" in self.components:
                # 使用真实情绪分析器
                sentiment_data = self.components["sentiment_analyzer"].analyze_market_sentiment(symbol)
                if sentiment_data:
                    return sentiment_data
        except Exception as e:
            logger.warning(f"情绪分析失败: {e}")

        # 降级为模拟情绪分析
        return self._mock_sentiment_analysis(symbol)

    def _mock_sentiment_analysis(self, symbol):
        """Mock sentiment analysis"""
        import random

        sentiment = {
            'composite_score': round(random.uniform(-1, 1), 2),
            'news_score': round(random.uniform(-1, 1), 2),
            'social_score': round(random.uniform(-1, 1), 2),
            'technical_score': round(random.uniform(-1, 1), 2),
            'sentiment_status': 'Neutral',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'analysis_details': {
                'news_count': random.randint(5, 30),
                'social_count': random.randint(10, 100),
                'indicators': {}
            }
        }

        score = sentiment['composite_score']
        if score > 0.7:
            sentiment['sentiment_status'] = 'Extremely Bullish'
        elif score > 0.3:
            sentiment['sentiment_status'] = 'Bullish'
        elif score > -0.3:
            sentiment['sentiment_status'] = 'Neutral'
        elif score > -0.7:
            sentiment['sentiment_status'] = 'Bearish'
        else:
            sentiment['sentiment_status'] = 'Extremely Bearish'

        return sentiment

    def _create_mock_data_for_symbol(self, symbol):
        """为不存在的股票创建模拟数据"""
        try:
            from core.utils.data_mock import create_stock_data

            if not hasattr(self, 'demo_data'):
                self.demo_data = {}

            self.demo_data[symbol] = create_stock_data(
                symbol=symbol,
                days=100,
                base_price=50.0 + hash(symbol) % 100,  # 基于股票代码的哈希生成不同的基础价格
                volatility=0.02
            )
            logger.info(f"成功为 {symbol} 创建了模拟数据")
            return True
        except Exception as e:
            logger.error(f"为 {symbol} 创建模拟数据失败: {e}")
            return False

    @st.cache_data(ttl=7200, show_spinner="正在获取最新新闻...")  # 缓存新闻 2 小时
    def get_news(self, symbol: str, num_articles: int = 20) -> List[Dict]:
        """
        使用 NewsAPI.org 获取与特定股票相关的新闻。
        Args:
            symbol (str): 股票代码。
            num_articles (int): 希望获取的文章数量上限。
        Returns:
            List[Dict]: 新闻文章列表，每个文章是一个字典。
        """
        news_api_key = getattr(self.config, 'NEWS_API_KEY', None)
        if not news_api_key:
            logger.warning("NewsAPI key not configured in config.py. Cannot fetch news.")
            return []

        # NewsAPI 对某些 Ticker (如 'SOUN') 可能找不到结果，可以尝试搜索公司全名
        # 这里为了通用性，我们先直接使用 symbol
        query = symbol

        # 构建 API 请求 URL
        # 使用 'everything' 端点可以搜索历史文章，'top-headlines' 只返回最新的头条
        base_url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'apiKey': news_api_key,
            'pageSize': num_articles,
            'sortBy': 'publishedAt',  # 按发布时间排序，最新的在前
            'language': 'en'  # 优先获取英文新闻，因为金融 LLM 通常在英文上训练得最好
        }

        logger.info(f"Fetching news for '{query}' from NewsAPI...")

        try:
            # 使用 self.requests_session (如果有) 或 requests
            session = getattr(self, 'requests_session', requests)
            response = session.get(base_url, params=params, timeout=10)  # 设置10秒超时

            # 检查 HTTP 状态码
            response.raise_for_status()  # 如果状态码不是 2xx，会抛出异常

            data = response.json()

            # 检查 API 返回的状态
            if data.get('status') == 'ok':
                articles = data.get('articles', [])
                logger.info(f"Successfully fetched {len(articles)} news articles for '{query}'.")
                # 返回我们需要的关键信息即可
                return [{'title': art.get('title'), 'description': art.get('description'),
                         'publishedAt': art.get('publishedAt'), 'url': art.get('url')} for art in articles]
            else:
                api_error_message = data.get('message', 'Unknown API error')
                logger.error(f"NewsAPI returned an error for query '{query}': {api_error_message}")
                return []

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request to NewsAPI failed for query '{query}': {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred during news fetching for '{query}': {e}", exc_info=True)
            return []

    def login_user(self, username, password) -> bool:
        """
        [最终版] 处理用户登录。成功后，加载数据并覆盖 session_state。
        """
        VALID_PASSWORDS = {"user": "1234", "admin": "admin"}
        if username in VALID_PASSWORDS and password == VALID_PASSWORDS[username]:
            st.session_state.logged_in = True
            st.session_state.username = username

            # 从数据库加载该用户的数据
            portfolio_from_db = self.persistence_manager.load_portfolio(user_id=username)
            trades_from_db = self.persistence_manager.load_trades(user_id=username)

            if portfolio_from_db:
                st.session_state.portfolio = portfolio_from_db
            else:  # 如果数据库没有该用户的数据，则创建一个新的并保存
                self.reset_to_default_portfolio()  # 使用默认模板
                self.persistence_manager.save_portfolio(
                    portfolio=st.session_state.portfolio,
                    user_id=username
                )

            st.session_state.trades = trades_from_db

            logger.info(f"User '{username}' logged in. Portfolio and trades loaded from DB.")
            return True
        return False

    def logout_user(self):
        """[最终版] 处理用户登出。重置为默认的游客账户。"""
        st.session_state.logged_in = False
        st.session_state.username = "Guest"
        self.reset_to_default_portfolio() # 用默认账户数据覆盖当前状态
        logger.info("User logged out. Session state reset to guest portfolio.")

    def reset_to_default_portfolio(self):
        """[辅助方法] 将 session_state 重置为默认的游客模拟账户"""
        st.session_state.portfolio = {
            'cash': 100000.0, 'positions': {},
            'total_value': 100000.0, 'last_update': datetime.now().isoformat()
        }
        st.session_state.trades = []

    def execute_trade_for_user(self, order_data: Dict, user_id: str):
        """
        [新增] 为指定用户执行交易，这是一个非 UI 的、服务化的方法。
        """
        logger.info(f"[AutoTrader Service] Executing trade for user '{user_id}': {order_data}")
        portfolio = self.persistence_manager.load_portfolio(user_id)
        if portfolio is None:
            logger.error(f"Cannot execute trade for '{user_id}', portfolio not found.")
            return {"success": False, "message": "Portfolio not found."}

        validation = self.risk_manager.validate_order(order_data, portfolio, system_ref=self)
        if not validation.get('valid'):
            logger.warning(f"Trade for '{user_id}' REJECTED by risk manager: {validation.get('reason')}")
            return {"success": False, "message": validation.get('reason')}

        result = self.order_executor.execute_order(**order_data)

        if result and result.get('success'):
            self._update_portfolio_for_user(order_data, result, user_id, portfolio)
        return result

    def _update_portfolio_for_user(self, order_data: Dict, execution_result: Dict, user_id: str, portfolio: Dict):
        """
        [新增] _update_portfolio 的一个变体，用于更新一个传入的投资组合对象并持久化。
        """
        symbol, quantity, direction = order_data.get('symbol'), order_data.get('quantity'), order_data.get('direction')
        exec_price = execution_result.get('price', order_data.get('price'))
        cost_or_proceeds = execution_result.get('total_cost', quantity * (exec_price or 0))
        commission = execution_result.get('commission', 0)

        logger.info(
            f"[_update_portfolio_for_user] Updating for '{user_id}': {direction} {quantity} {symbol} @ {exec_price:.2f}")
        positions = portfolio.setdefault('positions', {})

        if direction.lower() == "buy":
            portfolio['cash'] -= (cost_or_proceeds + commission)
            pos = positions.setdefault(symbol, {'quantity': 0, 'cost_basis': 0})
            new_qty = pos.get('quantity', 0) + quantity
            pos['cost_basis'] = ((pos.get('quantity', 0) * pos.get('cost_basis', 0)) + (
                        quantity * exec_price)) / new_qty if new_qty > 0 else 0
            pos['quantity'] = new_qty
        elif direction.lower() == "sell":
            portfolio['cash'] += (cost_or_proceeds - commission)
            if symbol in positions:
                pos = positions[symbol]
                pos['quantity'] -= min(quantity, pos.get('quantity', 0))
                if pos['quantity'] <= 1e-9: del positions[symbol]

        if symbol in positions: positions[symbol]['current_price'] = exec_price
        portfolio['last_update'] = datetime.now().isoformat()
        portfolio['total_value'] = portfolio.get('cash', 0) + sum(
            p.get('quantity', 0) * p.get('current_price', 0) for p in positions.values())

        new_trade_record = {'symbol': symbol, 'direction': direction, 'quantity': quantity, 'price': exec_price,
                            'timestamp': datetime.now(), 'total': cost_or_proceeds, 'commission': commission,
                            'is_mock': execution_result.get('is_mock', True)}

        try:
            self.persistence_manager.save_portfolio(portfolio, user_id=user_id)
            self.persistence_manager.add_trade(new_trade_record, user_id=user_id)
            logger.info(f"Portfolio for user '{user_id}' updated and persisted via AutoTrader Service.")
        except Exception as e:
            logger.error(f"Failed to persist portfolio/trade update for user '{user_id}': {e}", exc_info=True)

    def __del__(self):
        """在对象被垃圾回收时尝试清理资源，主要是关闭WebSocket。"""
        logger.info("TradingSystem instance being deleted. Attempting to stop WebSocketManager.")
        if hasattr(self, 'data_manager') and self.data_manager and hasattr(self.data_manager, 'websocket_manager'):
            if self.data_manager.websocket_manager:
                self.data_manager.websocket_manager.stop()