# app/main.py

import os
import sys
import asyncio
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import yfinance as yf
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import plotly.graph_objs as go  # 添加这行

# 添加项目根目录到Python路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from core.data.fetcher import DataFetcher
from core.analysis.technical import TechnicalAnalyzer
from core.analysis.sentiment import SentimentAnalyzer
from core.strategy.ml_strategy import MLStrategy
from core.strategy.custom_strategy import CustomStrategy
from core.trading.executor import OrderExecutor  # 更正导入路径
from core.risk.manager import RiskManager
from core.analysis.performance import PerformanceAnalyzer
from core.backtesting.engine import BacktestEngine
from core.market.scanner import MarketScanner
from config import Config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app_components(config, scanner, data_fetcher):
    """创建应用程序组件"""
    try:
        # 初始化分析器组件
        technical_analyzer = TechnicalAnalyzer(config)
        sentiment_analyzer = SentimentAnalyzer(config)
        performance_analyzer = PerformanceAnalyzer(config)

        # 初始化策略组件
        ml_strategy = MLStrategy(config)
        custom_strategy = CustomStrategy()
        custom_strategy.set_scanner(scanner)

        # 初始化交易和风险组件
        risk_manager = RiskManager(config)
        order_executor = OrderExecutor(config, risk_manager)

        # 初始化回测引擎
        backtest_engine = BacktestEngine(config)

        # 组装所有组件
        components = {
            'config': config,
            'scanner': scanner,  # 这里是 MarketScanner 的实例
            'data_fetcher': data_fetcher,
            'technical_analyzer': technical_analyzer,
            'sentiment_analyzer': sentiment_analyzer,
            'performance_analyzer': performance_analyzer,
            'ml_strategy': ml_strategy,
            'custom_strategy': custom_strategy,
            'order_executor': order_executor,
            'risk_manager': risk_manager,
            'backtest_engine': backtest_engine
        }

        logger.info("All components created successfully")
        return components

    except Exception as e:
        logger.error(f"Error creating components: {e}")
        raise


# app/main.py

class TradingApp:
    def __init__(self):
        """初始化交易应用"""
        self.config = {
            'MAX_WORKERS': 4,
            # 扫描器配置
            'SCANNER_CONFIG': {
                'delay': 60,
                'markets': ['US'],
                'symbols': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'],
                'enable_alpha_vantage': True,
                'enable_tushare': True,
                'default_markets': ['US', 'CN']
            },
            'API_CONFIG': {
                'endpoints': {
                    'market_data': 'your_market_data_endpoint',
                    'minute_data': 'your_minute_data_endpoint',
                    'symbols': 'your_symbols_endpoint'
                }
            },
            # 添加风险管理配置
            'MAX_POSITION_SIZE': 0.1,
            'MAX_DRAWDOWN': 0.2,
            'VAR_LIMIT': 0.05,
            'VOLATILITY_LIMIT': 0.3,
            'RISK_FREE_RATE': 0.02,

            'RISK_CONFIG': {
                'MAX_POSITION_SIZE': 0.1,
                'MAX_SINGLE_LOSS': 0.02,
                'STOP_LOSS': 0.05,
                'TAKE_PROFIT': 0.1,
                'MAX_DRAWDOWN': 0.2,
                'RISK_FREE_RATE': 0.02,
                'POSITION_LIMITS': {
                    'US': 0.8,
                    'CN': 0.8
                }
            },
            # 添加其他可能需要的配置参数
            'MAX_POSITION_SIZE': 0.1,  # 直接在根级别也添加
            'PRICE_CACHE_DURATION': 60,  # 价格缓存持续时间（秒）
            'MA_PERIODS': [5, 10, 20, 50],  # 移动平均线周期
            'RSI_PERIOD': 14,  # RSI周期
            'MACD_PARAMS': {  # MACD参数
                'fast': 12,
                'slow': 26,
                'signal': 9
            },
            'BOLLINGER_PARAMS': {  # 布林带参数
                'period': 20,
                'std_dev': 2
            },
            'MARKET_HOURS': {  # 市场交易时间
                'US': {
                    'market_open': '09:30',
                    'market_close': '16:00'
                },
                'CN': {
                    'morning_open': '09:30',
                    'morning_close': '11:30',
                    'afternoon_open': '13:00',
                    'afternoon_close': '15:00'
                }
            },

                # 机器学习策略配置
                'ML_CONFIG': {
                    'MODEL_PATH': 'models/xgb_model.joblib',
                    'FEATURE_COLUMNS': [
                        'close', 'volume', 'ma_5', 'ma_20', 'rsi', 'macd',
                        'macd_signal', 'upper_band', 'lower_band'
                    ],
                    'TARGET_COLUMN': 'target',
                    'TRAIN_TEST_SPLIT': 0.8,
                    'PREDICTION_HORIZON': 5,
                    'RETRAIN_INTERVAL': 7,  # days
                    'MIN_TRAINING_SAMPLES': 1000,
                    'HYPERPARAMETERS': {
                        'n_estimators': 100,
                        'max_depth': 10,
                        'min_samples_split': 2,
                        'min_samples_leaf': 1
                    }
                },

                # 模型路径
                'MODEL_PATH': 'models/xgb_model.joblib',  # 在根级别也添加

                # 其他模型相关配置
                'MODEL_CONFIG': {
                    'input_size': 10,
                    'hidden_size': 64,
                    'output_size': 3,
                    'num_layers': 2,
                    'dropout': 0.2
                }
            },
        self.config['SCANNER_CONFIG'].update({
            'enable_alpha_vantage': True,
            'enable_tushare': True,
            'default_markets': ['US', 'CN'],
            'volume_threshold': 1.5,
            'technical_filters': {
                'enable_ma': True,
                'enable_rsi': False,
                'enable_macd': False
            }
        })

        self.executor = ThreadPoolExecutor(max_workers=4)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.last_update_time = None
        self.update_interval = 5

        self.DEFAULT_CONFIG['SCREENER_CONFIG'] = self.SCREENER_CONFIG

        self.logger = logging.getLogger(__name__)

        # 初始化应用状态
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = 60
        if 'last_refresh_time' not in st.session_state:
            st.session_state.last_refresh_time = time.time()

        # 初始化所有组件
        self.loop.run_until_complete(self.setup_components())
        self.initialize_state()

    # app/main.py 中的 TradingApp 类

    async def setup_components(self):
        """异步初始化系统组件"""
        try:
            # 1. 初始化市场扫描器
            scanner_config = {
                'SCANNER_CONFIG': {
                    'delay': 60,
                    'markets': ['US'],
                    'symbols': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'],
                    'api_endpoints': {
                        'market_data': 'your_market_data_endpoint',
                        'minute_data': 'your_minute_data_endpoint',
                        'symbols': 'your_symbols_endpoint'
                    }
                },
                'API_CONFIG': self.config.get('API_CONFIG', {})
            }

            self.market_scanner = MarketScanner(config=scanner_config)
            await self.market_scanner.initialize()  # 先初始化
            await self.market_scanner.start()  # 再启动
            self.logger.info("Market scanner initialized")

            # 2. 数据和分析组件
            try:
                self.data_fetcher = DataFetcher(self.config)
                self.logger.info("Data fetcher initialized")
            except Exception as e:
                self.logger.error(f"Error initializing data fetcher: {e}")
                raise

            try:
                self.technical_analyzer = TechnicalAnalyzer(self.config)
                self.logger.info("Technical analyzer initialized")
            except Exception as e:
                self.logger.error(f"Error initializing technical analyzer: {e}")
                raise

            try:
                self.sentiment_analyzer = SentimentAnalyzer(self.config)
                self.logger.info("Sentiment analyzer initialized")
            except Exception as e:
                self.logger.error(f"Error initializing sentiment analyzer: {e}")
                raise

            # 3. 风险管理器
            try:
                self.risk_manager = RiskManager(self.config)
                self.logger.info("Risk manager initialized")
            except Exception as e:
                self.logger.error(f"Error initializing risk manager: {str(e)}")
                self.logger.error(f"Config being passed: {self.config}")  # 打印配置内容
                raise

            # 4. 策略组件
            try:
                self.ml_strategy = MLStrategy(self.config)
                self.custom_strategy = CustomStrategy()
                self.logger.info("Strategy components initialized")
            except Exception as e:
                self.logger.error(f"Error initializing strategy components: {e}")
                raise

            # 5. 订单执行器
            try:
                self.order_executor = OrderExecutor(
                    config=self.config,
                    risk_manager=self.risk_manager
                )
                self.logger.info("Order executor initialized")
            except Exception as e:
                self.logger.error(f"Error initializing order executor: {e}")
                raise

            # 6. 分析组件
            try:
                self.performance_analyzer = PerformanceAnalyzer(config=self.config)
                self.backtest_engine = BacktestEngine(self.config)
                self.logger.info("Analysis components initialized")
            except Exception as e:
                self.logger.error(f"Error initializing analysis components: {e}")
                raise

            # 设置扫描器关联
            self.data_fetcher.set_scanner(self.market_scanner)

            # 为 custom_strategy 设置必要的组件
            try:
                self.custom_strategy.set_scanner(self.market_scanner)
                self.custom_strategy.set_technical_analyzer(self.technical_analyzer)
                self.custom_strategy.set_sentiment_analyzer(self.sentiment_analyzer)
                self.logger.info("Custom strategy setup completed")
            except Exception as e:
                self.logger.error(f"Error setting up custom strategy: {e}")
                raise

            self.logger.info("All components initialized successfully")

        except Exception as e:
            self.logger.error(f"Error setting up components: {str(e)}")
            self.logger.error(f"Current config: {self.config}")  # 打印当前配置
            raise

    def initialize_state(self):
        """初始化应用状态"""
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = {
                'cash': 100000.0,
                'positions': {},
                'total_value': 100000.0,
                'last_update': datetime.now()
            }
        if 'trades' not in st.session_state:
            st.session_state.trades = []
        if 'portfolio_history' not in st.session_state:
            st.session_state.portfolio_history = []
        if 'risk_metrics' not in st.session_state:
            st.session_state.risk_metrics = {}
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = {}

    def run(self):
        """运行应用"""
        try:
            st.set_page_config(
                page_title="智能交易系统",
                page_icon="📈",
                layout="wide"
            )

            # 错误恢复机制
            if 'error_state' in st.session_state and st.session_state.error_state:
                if st.button("重置系统状态"):
                    for key in ['error_state', 'app']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
                return
            # 在页面最上方添加自动刷新控制
            self.render_refresh_controls()

            # 侧边栏
            self.render_sidebar()

            # 主要标签页
            self.render_main_area()

            # 处理自动刷新
            if st.session_state.auto_refresh:
                current_time = time.time()
                if current_time - st.session_state.last_refresh_time >= st.session_state.refresh_interval:
                    st.session_state.last_refresh_time = current_time
                    time.sleep(0.1)
                    st.rerun()

        except Exception as e:
            st.session_state.error_state = True
            self.logger.error(f"Error running application: {e}")
            st.error("系统运行出错，请点击'重置系统状态'按钮重试")


        except Exception as e:
            logger.error(f"Error running application: {e}")
            st.error("应用运行出错，请检查日志或联系管理员")

    def render_refresh_controls(self):
        """渲染刷新控制"""
        with st.container():
            col1, col2 = st.columns([1, 3])

            with col1:
                st.session_state.auto_refresh = st.checkbox(
                    "启用自动刷新",
                    value=st.session_state.auto_refresh,
                    key="auto_refresh_checkbox"
                )

            with col2:
                if st.session_state.auto_refresh:
                    st.session_state.refresh_interval = st.slider(
                        "刷新间隔(秒)",
                        min_value=5,
                        max_value=300,
                        value=st.session_state.refresh_interval,
                        key="refresh_interval_slider"
                    )

                    # 显示下次刷新时间
                    time_to_next = max(0, st.session_state.refresh_interval -
                                       (time.time() - st.session_state.last_refresh_time))
                    st.info(f"下次刷新在 {int(time_to_next)} 秒后")

    def display_main_content(self):
        """显示主要内容"""
        try:
            # 使用 st.empty() 创建占位符来更新内容
            main_container = st.empty()

            with main_container.container():
                # 添加所有现有的显示逻辑
                self.display_market_status()
                self.display_portfolio()
                self.display_trading_signals()
                # ... 其他显示函数 ...

        except Exception as e:
            st.error(f"显示内容时出错: {str(e)}")

    def display_market_status(self):
        """显示市场状态"""
        try:
            status = "开市" if self._is_market_open() else "休市"
            st.info(f"市场状态: {status}")
        except Exception as e:
            st.warning(f"获取市场状态失败: {str(e)}")

    def _is_market_open(self):
        """检查市场是否开放"""
        # 实现市场开放检查逻辑
        return True  # 示例返回值

    def render_sidebar(self):
        """渲染侧边栏"""
        with st.sidebar:
            st.title("智能交易系统")

            # 市场选择
            market = st.selectbox(
                "选择市场",
                ["美股", "A股"],
                key="market_selector"
            )

            # 账户信息
            st.subheader("账户信息")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "可用资金",
                    f"${st.session_state.portfolio['cash']:,.2f}"
                )
            with col2:
                st.metric(
                    "总资产",
                    f"${st.session_state.portfolio['total_value']:,.2f}"
                )

            # 风险设置
            st.subheader("风险控制")
            max_position = st.slider(
                "最大持仓比例",
                0.0, 1.0, 0.2,
                key="max_position_size"
            )
            stop_loss = st.slider(
                "止损比例",
                0.0, 0.5, 0.1,
                key="stop_loss"
            )

            # 系统状态
            st.subheader("系统状态")
            last_update = st.session_state.portfolio['last_update']
            st.text(f"最后更新: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")

    def render_main_area(self):
        """渲染主要区域"""
        tabs = st.tabs([
            "市场监控",
            "高级筛选",
            "交易执行",
            "持仓管理",
            "交易策略",
            "回测分析",
            "绩效报告",
            "自动止损止盈",  # 新增
            "市场情绪"  # 新增
        ])

        with tabs[0]:
            self.market_monitoring_tab()
        with tabs[1]:  # 新增筛选标签页
            self.enhanced_screener_tab()
        with tabs[2]:
            self.trading_execution_tab()
        with tabs[3]:
            self.portfolio_management_tab()
        with tabs[4]:
            self.trading_strategy_tab()
        with tabs[5]:
            self.backtest_analysis_tab()
        with tabs[6]:
            self.performance_report_tab()
        with tabs[7]:
            self.auto_stop_settings_tab()  # 新增
        with tabs[8]:
            self.market_sentiment_analysis_tab()  # 新增

    async def get_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """异步获取股票数据"""
        try:
            # 首先检查symbol是否在有效的交易标的中
            if symbol not in self.market_scanner.base_scanner.active_symbols:
                logger.warning(f"Symbol {symbol} is not in active universe")
                return None

            data = await self.data_fetcher.get_historical_data(
                symbol=symbol,
                start_date=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d')
            )
            return data
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            return None

    def market_monitoring_tab(self):
        """市场监控标签页"""
        st.header("市场监控")

        # 市场选择
        market = st.session_state.get("market_selector", "美股")
        market_code = "US" if market == "美股" else "CN"

        # 股票代码输入
        symbol = st.text_input("输入股票代码", key="monitor_symbol")

        if symbol:
            # 处理A股后缀
            if market_code == "CN" and not symbol.endswith(('.SH', '.SZ')):
                symbol += '.SH'

            # 获取股票数据
            data = self.loop.run_until_complete(self.get_stock_data(symbol))
            if data is not None and not data.empty:
                # 显示股票信息
                self.display_stock_info(symbol, data)

                # 获取技术分析结果
                try:
                    analysis = self.technical_analyzer.analyze(data)

                    # 创建图表
                    fig = go.Figure()

                    # 添加K线
                    fig.add_trace(go.Candlestick(
                        x=data.index,
                        open=data['open'],
                        high=data['high'],
                        low=data['low'],
                        close=data['close'],
                        name='K线'
                    ))

                    # 添加移动平均线
                    if 'ma' in analysis:
                        ma_periods = [5, 10, 20, 30, 60]
                        colors = ['blue', 'green', 'red', 'purple', 'orange']
                        for ma, period, color in zip(analysis['ma'], ma_periods, colors):
                            fig.add_trace(go.Scatter(
                                x=data.index,
                                y=ma,
                                name=f'MA{period}',
                                line=dict(color=color, width=1)
                            ))

                    # 添加布林带
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

                    # 更新图表布局
                    fig.update_layout(
                        title=f"{symbol} 价格与技术指标",
                        xaxis_rangeslider_visible=False,
                        height=600
                    )

                    # 显示图表
                    st.plotly_chart(fig, use_container_width=True)

                    # 显示技术分析详情
                    self.display_technical_analysis(data)
                except Exception as e:
                    st.error(f"技术分析出错: {str(e)}")
            else:
                st.warning("无法获取该股票数据")

    def run_enhanced_scan(self, symbols: str, market: str, **params) -> List[Dict]:
                """执行增强版扫描"""
                symbols_list = [s.strip().upper() for s in symbols.split(',')]
                results = []

                for symbol in symbols_list:
                    try:
                        # 处理A股后缀
                        if market == "CN" and not symbol.endswith(('.SH', '.SZ')):
                            symbol += '.SH'

                        # 使用原有DataFetcher获取数据
                        data = self.loop.run_until_complete(
                            self.data_fetcher.get_historical_data(
                                symbol=symbol,
                                market=market,
                                days=params['days'],
                                interval='1d'
                            )
                        )

                        if data is None or data.empty:
                            continue

                        # 基础量价筛选
                        vol_cond = (data['volume'].iloc[-1] >
                                    data['volume'].mean() * params['vol_threshold'])
                        if not vol_cond:
                            continue

                        # 技术指标筛选
                        if params.get('tech_params'):
                            analysis = self.technical_analyzer.analyze(data)
                            if 'ma_short' in params['tech_params']:
                                ma_cross = (analysis['ma'][0][-1] > analysis['ma'][1][-1])
                                if not ma_cross:
                                    continue

                        # 通过所有筛选条件
                        results.append({
                            'symbol': symbol,
                            'data': data,
                            'analysis': analysis if params.get('tech_params') else None
                        })

                    except Exception as e:
                        self.logger.warning(f"扫描{symbol}时出错: {str(e)}")
                        continue

                return results

    def display_scan_results(self, results: List[Dict]):
                """显示扫描结果"""
                if not results:
                    st.info("未找到符合条件标的")
                    return

                # 显示概览表格
                df = pd.DataFrame([{
                    '代码': r['symbol'],
                    '当前价': r['data']['close'].iloc[-1],
                    '量能变化%': f"{(r['data']['volume'].iloc[-1] / r['data']['volume'].mean() - 1) * 100:.1f}",
                    '5日均线': r['analysis']['ma'][0][-1] if r.get('analysis') else 'N/A',
                    '20日均线': r['analysis']['ma'][1][-1] if r.get('analysis') else 'N/A'
                } for r in results])

                st.dataframe(df.style.format({
                    '当前价': "{:.2f}",
                    '量能变化%': "{:.1f}%",
                    '5日均线': "{:.2f}",
                    '20日均线': "{:.2f}"
                }), use_container_width=True)

                # 添加图表展示
                selected = st.selectbox(
                    "查看详情",
                    [r['symbol'] for r in results],
                    key="screener_detail"
                )
                if selected:
                    result = next(r for r in results if r['symbol'] == selected)
                    self.display_stock_chart(result['data'], result.get('analysis'))

    def display_stock_chart(self, data: pd.DataFrame, analysis: Optional[Dict] = None):
        """显示股票图表（复用原有可视化逻辑）"""
        fig = go.Figure()

        # K线图
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='K线'
        ))

        # 技术指标
        if analysis:
            if 'ma' in analysis:
                for period, values in zip([5, 20], analysis['ma']):
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=values,
                        name=f'MA{period}',
                        line=dict(width=1)
                    ))

        fig.update_layout(
            title="价格与技术指标",
            xaxis_rangeslider_visible=False,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    def perform_technical_analysis(self, data: pd.DataFrame) -> Dict:
        """执行技术分析"""
        try:
            analysis_results = self.technical_analyzer.analyze(data)
            signals = self.technical_analyzer.generate_signals(data)
            market_regime = self.technical_analyzer.get_market_regime(data)
            support_resistance = self.technical_analyzer.get_support_resistance(data)

            return {
                'analysis': analysis_results,
                'signals': signals,
                'market_regime': market_regime,
                'support_resistance': support_resistance
            }
        except Exception as e:
            self.logger.error(f"Error in technical analysis: {str(e)}")
            return {}

    def display_stock_info(self, symbol: str, data: pd.DataFrame):
        """显示股票基本信息"""
        try:
            col1, col2, col3 = st.columns(3)

            with col1:
                current_price = data['close'].iloc[-1]
                price_change = data['close'].iloc[-1] - data['close'].iloc[-2]
                price_change_pct = price_change / data['close'].iloc[-2] * 100
                st.metric(
                    "当前价格",
                    f"${current_price:.2f}",
                    f"{price_change_pct:.2f}%"
                )

            with col2:
                volume = data['volume'].iloc[-1]
                volume_change = (data['volume'].iloc[-1] / data['volume'].iloc[-2] - 1) * 100
                st.metric(
                    "成交量",
                    f"{volume:,.0f}",
                    f"{volume_change:.2f}%"
                )

            with col3:
                volatility = data['close'].pct_change().std() * np.sqrt(252) * 100
                st.metric("波动率", f"{volatility:.2f}%")

        except Exception as e:
            logger.error(f"Error displaying stock info: {e}")
            st.error("无法显示股票信息")

    def display_sentiment_analysis(self, symbol: str):
        """显示情绪分析"""
        try:
            # 获取情绪分析数据
            sentiment_data = self.loop.run_until_complete(
                self.sentiment_analyzer.analyze_market_sentiment(symbol)
            )

            st.subheader("市场情绪分析")

            # 显示综合情绪指标
            col1, col2 = st.columns(2)
            with col1:
                sentiment_color = "red" if sentiment_data['composite_score'] < 0 else "green"
                st.metric(
                    "综合情绪指数",
                    f"{sentiment_data['composite_score']:.2f}",
                    sentiment_data['sentiment_status']
                )

            with col2:
                st.metric(
                    "情绪状态",
                    sentiment_data['sentiment_status']
                )

            # 显示分项情绪得分
            st.subheader("分项情绪分析")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "新闻情绪",
                    f"{sentiment_data['news_score']:.2f}"
                )
            with col2:
                st.metric(
                    "社交媒体情绪",
                    f"{sentiment_data['social_score']:.2f}"
                )
            with col3:
                st.metric(
                    "技术指标情绪",
                    f"{sentiment_data['technical_score']:.2f}"
                )

            # 显示分析详情
            if sentiment_data['analysis_details']:
                st.subheader("分析详情")
                st.write(f"分析新闻数量: {sentiment_data['analysis_details']['news_count']}")
                st.write(f"社交媒体数据量: {sentiment_data['analysis_details']['social_count']}")
                if sentiment_data['analysis_details']['indicators']:
                    st.write("技术指标:", sentiment_data['analysis_details']['indicators'])

        except Exception as e:
            self.logger.error(f"Error displaying sentiment analysis: {e}")
            st.error("情绪分析显示出错")

    def display_technical_analysis(self, data: pd.DataFrame):
        # 计算简单均线（不依赖TechnicalAnalyzer）
        data['ma5'] = data['close'].rolling(5).mean()
        data['ma20'] = data['close'].rolling(20).mean()

        # 显示图表
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['close'], name='价格'))
        fig.add_trace(go.Scatter(x=data.index, y=data['ma5'], name='5日均线'))
        fig.add_trace(go.Scatter(x=data.index, y=data['ma20'], name='20日均线'))
        st.plotly_chart(fig)

        """显示技术分析"""
        try:
            st.subheader("技术分析")

            # 计算技术指标
            analysis_results = self.technical_analyzer.analyze(data)

            # 显示主要指标
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("MA趋势", str(analysis_results.get('ma_trend', 'N/A')))
            with col2:
                rsi_value = analysis_results.get('rsi', pd.Series([0])).iloc[-1]
                st.metric("RSI", f"{rsi_value:.2f}")
            with col3:
                macd_value = analysis_results.get('macd', pd.Series([0])).iloc[-1]
                st.metric("MACD", f"{macd_value:.2f}")

            # 绘制技术分析图表
            try:
                fig = self.technical_analyzer.plot_indicators(data)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                self.logger.error(f"Error plotting indicators: {e}")
                st.warning("无法显示技术分析图表")

        except Exception as e:
            self.logger.error(f"Error in technical analysis: {e}")
            st.error("技术分析显示出错")

    def display_prediction_analysis(self, data: pd.DataFrame):
        """显示预测分析"""
        try:
            st.subheader("预测分析")

            # 首先进行技术分析
            analysis_results = self.technical_analyzer.analyze(data)

            # 将分析结果转换为DataFrame
            prediction_data = pd.DataFrame({
                key: value.values if isinstance(value, pd.Series) else value
                for key, value in analysis_results.items()
                if key in self.ml_strategy.required_features
            })

            # 确保数据格式正确
            prediction_data = prediction_data.astype(float)

            # 获取ML预测结果
            prediction = self.ml_strategy.predict(prediction_data)

            if prediction and 'direction' in prediction:
                col1, col2 = st.columns(2)
                with col1:
                    direction_text = {
                        1: "上涨",
                        -1: "下跌",
                        0: "持平"
                    }.get(prediction['direction'], "未知")
                    st.metric("预测方向", direction_text)
                with col2:
                    st.metric("置信度", f"{prediction.get('confidence', 0):.2f}%")

                # 显示预测详情
                if 'details' in prediction and prediction['details']:
                    st.write("预测详情:", prediction['details'])
            else:
                st.warning("无法生成预测结果")

        except Exception as e:
            self.logger.error(f"Error in prediction analysis: {str(e)}")
            st.error("预测分析出错")

    def trading_execution_tab(self):
        """交易执行标签页"""
        st.header("交易执行")

        # 交易表单
        with st.form("trade_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                symbol = st.text_input("股票代码")
                quantity = st.number_input("数量", min_value=1, value=100)

            with col2:
                order_type = st.selectbox(
                    "订单类型",
                    ["市价单", "限价单"]
                )
                if order_type == "限价单":
                    price = st.number_input("价格", min_value=0.01)
                else:
                    price = None

            with col3:
                trade_action = st.selectbox(
                    "交易方向",
                    ["买入", "卖出"]
                )

            if st.form_submit_button("提交订单"):
                if trade_action == "卖出":
                    quantity = -quantity
                self.loop.run_until_complete(
                    self.execute_trade(symbol, quantity, order_type, price)
                )

        # 显示活跃订单
        st.subheader("活跃订单")
        if st.session_state.trades:
            self.display_active_orders()
        else:
            st.info("当前没有活跃订单")

        # 显示活跃订单
        st.subheader("活跃订单")
        if st.session_state.trades:
            self.display_active_orders()
        else:
            st.info("当前没有活跃订单")

    async def execute_trade(self, symbol: str, quantity: int, order_type: str, price: Optional[float] = None):
        """执行交易"""
        try:
            # 获取当前价格
            current_price = price if price is not None else await self.get_current_price(symbol)
            if current_price is None:
                st.error("无法获取当前价格")
                return

            # 计算交易总额
            total_cost = abs(quantity) * current_price

            # 检查资金是否足够（买入时）
            if quantity > 0 and total_cost > st.session_state.portfolio['cash']:
                st.error("可用资金不足")
                return

            # 检查持仓是否足够（卖出时）
            if quantity < 0:
                current_position = st.session_state.portfolio['positions'].get(symbol, 0)
                if abs(quantity) > current_position:
                    st.error("持仓不足")
                    return

            # 检查风险限制
            risk_check = await self.risk_manager.check_trade_risk(
                symbol=symbol,
                quantity=quantity,
                price=current_price,
                portfolio=st.session_state.portfolio
            )

            if not risk_check['allowed']:
                st.error(f"交易被风险控制拒绝: {risk_check['reason']}")
                return

            # 执行交易
            trade_result = await self.order_executor.execute_order(
                symbol=symbol,
                quantity=quantity,
                price=current_price,
                order_type=order_type
            )

            if trade_result['success']:
                # 更新投资组合
                await self.update_portfolio(trade_result)
                st.success("交易执行成功")
            else:
                st.error(f"交易执行失败: {trade_result['message']}")

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            st.error("交易执行出错")

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        try:
            data = await self.get_stock_data(symbol)
            return data['close'].iloc[-1] if data is not None else None
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return None

    async def update_portfolio(self, trade_result: Dict):
        """更新投资组合"""
        try:
            symbol = trade_result['symbol']
            quantity = trade_result['quantity']
            total_cost = trade_result['total_cost']

            # 更新现金
            if quantity > 0:  # 买入
                st.session_state.portfolio['cash'] -= total_cost
            else:  # 卖出
                st.session_state.portfolio['cash'] += total_cost

            # 更新持仓
            if symbol not in st.session_state.portfolio['positions']:
                st.session_state.portfolio['positions'][symbol] = 0
            st.session_state.portfolio['positions'][symbol] += quantity

            # 如果持仓变为0，删除该持仓记录
            if st.session_state.portfolio['positions'][symbol] == 0:
                del st.session_state.portfolio['positions'][symbol]

            # 更新总资产价值
            await self.update_portfolio_value()

            # 记录交易
            st.session_state.trades.append(trade_result)

            # 更新最后更新时间
            st.session_state.portfolio['last_update'] = datetime.now()

            # 记录投资组合历史
            history_entry = {
                'timestamp': datetime.now(),
                'total_value': st.session_state.portfolio['total_value'],
                'cash': st.session_state.portfolio['cash'],
                'positions': st.session_state.portfolio['positions'].copy()
            }
            st.session_state.portfolio_history.append(history_entry)

        except Exception as e:
            logger.error(f"Error updating portfolio: {e}")
            raise Exception("Error message")


    def portfolio_management_tab(self):
        """投资组合管理标签页"""
        st.header("投资组合管理")

        # 显示当前持仓概况
        self.display_portfolio_summary()

        # 显示持仓详情
        self.display_portfolio_details()

        # 显示历史交易记录
        self.display_trade_history()

    def display_portfolio_summary(self):
        """显示投资组合概况"""
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "可用资金",
                f"${st.session_state.portfolio['cash']:,.2f}"
            )

        with col2:
            total_positions_value = sum(
                position['quantity'] * position['current_price']
                for position in st.session_state.portfolio['positions'].values()
            )
            st.metric(
                "持仓市值",
                f"${total_positions_value:,.2f}"
            )

        with col3:
            total_value = st.session_state.portfolio['cash'] + total_positions_value
            st.metric(
                "总资产",
                f"${total_value:,.2f}"
            )

    def update_price_data(self, symbol: str):
        """更新价格数据"""
        current_time = time.time()

        # 检查是否需要更新
        if (self.last_update_time is None or
                current_time - self.last_update_time >= self.update_interval):

            # 获取实时数据
            price_data = self.data_fetcher.get_realtime_price(symbol)

            if price_data:
                if price_data.get('delayed', False):
                    st.warning("显示的是延迟数据")

                self.last_update_time = current_time
                return price_data

        return None

    def display_portfolio_details(self):
        """显示持仓详情"""
        st.subheader("当前持仓")

        if st.session_state.portfolio['positions']:
            # 创建持仓数据表
            positions_data = []
            for symbol, position in st.session_state.portfolio['positions'].items():
                positions_data.append({
                    "股票代码": symbol,
                    "持仓数量": position['quantity'],
                    "当前价格": f"${position['current_price']:.2f}",
                    "市值": f"${position['quantity'] * position['current_price']:.2f}",
                    "成本价": f"${position['cost_basis']:.2f}",
                    "盈亏": f"${(position['current_price'] - position['cost_basis']) * position['quantity']:.2f}"
                })

            st.dataframe(pd.DataFrame(positions_data))
        else:
            st.info("当前没有持仓")

    def display_trade_history(self):
        """显示交易历史"""
        st.subheader("交易历史")

        if st.session_state.trades:
            trades_df = pd.DataFrame(st.session_state.trades)
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df = trades_df.sort_values('timestamp', ascending=False)
            st.dataframe(trades_df)
        else:
            st.info("暂无交易记录")

    def update_analysis(self):
            """更新分析数据"""
            try:
                if hasattr(self, 'current_data') and not self.current_data.empty:
                    analysis_results = self.perform_technical_analysis(self.current_data)
                    self.current_analysis = analysis_results
                    self.logger.info("Technical analysis updated successfully")
                else:
                    self.logger.warning("No data available for analysis")
            except Exception as e:
                self.logger.error(f"Error updating analysis: {str(e)}")

    def display_risk_metrics(self):
            """显示风险指标"""
            try:
                # 更新风险指标
                risk_metrics = self.risk_manager.calculate_risk_metrics(
                    portfolio=st.session_state.portfolio,
                    history=st.session_state.portfolio_history
                )
                st.session_state.risk_metrics = risk_metrics

                # 显示风险指标
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "投资组合Beta",
                        f"{risk_metrics.get('portfolio_beta', 0):.2f}"
                    )

                with col2:
                    st.metric(
                        "波动率",
                        f"{risk_metrics.get('volatility', 0):.2f}%"
                    )

                with col3:
                    st.metric(
                        "VaR (95%)",
                        f"${risk_metrics.get('var_95', 0):,.2f}"
                    )

            except Exception as e:
                logger.error(f"Error displaying risk metrics: {e}")
                st.error("风险指标显示出错")

    def backtest_analysis_tab(self):
            """回测分析标签页"""
            st.header("回测分析")

            # 回测参数设置
            with st.form("backtest_form"):
                col1, col2 = st.columns(2)

                with col1:
                    start_date = st.date_input(
                        "开始日期",
                        datetime.now() - timedelta(days=365)
                    )
                    symbols = st.text_input(
                        "股票代码（用逗号分隔）",
                        "AAPL,GOOGL,MSFT"
                    ).split(',')

                with col2:
                    end_date = st.date_input(
                        "结束日期",
                        datetime.now()
                    )
                    initial_capital = st.number_input(
                        "初始资金",
                        value=100000.0
                    )

                if st.form_submit_button("运行回测"):
                    self.run_backtest(
                        symbols=symbols,
                        start_date=start_date,
                        end_date=end_date,
                        initial_capital=initial_capital
                    )

    def run_backtest(self, symbols: List[str], start_date: datetime,
                         end_date: datetime, initial_capital: float):
            """运行回测"""
            try:
                # 运行回测
                backtest_results = self.backtest_engine.run(
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital
                )

                # 显示回测结果
                self.display_backtest_results(backtest_results)

            except Exception as e:
                logger.error(f"Error in backtest: {e}")
                st.error("回测执行出错")

    def display_backtest_results(self, results: Dict):
            """显示回测结果"""
            try:
                st.subheader("回测结果")

                # 显示主要指标
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "总收益率",
                        f"{results['total_return']:.2f}%"
                    )
                with col2:
                    st.metric(
                        "年化收益率",
                        f"{results['annual_return']:.2f}%"
                    )
                with col3:
                    st.metric(
                        "夏普比率",
                        f"{results['sharpe_ratio']:.2f}"
                    )
                with col4:
                    st.metric(
                        "最大回撤",
                        f"{results['max_drawdown']:.2f}%"
                    )

                # 绘制回测曲线
                if 'equity_curve' in results:
                    st.plotly_chart(
                        results['equity_curve'],
                        use_container_width=True
                    )

                # 显示详细的交易记录
                if 'trades' in results:
                    st.subheader("交易记录")
                    st.dataframe(results['trades'])

            except Exception as e:
                logger.error(f"Error displaying backtest results: {e}")
                st.error("回测结果显示出错")

    def performance_report_tab(self):
            """绩效报告标签页"""
            st.header("绩效报告")

            # 计算并更新性能指标
            self.update_performance_metrics()

            # 显示绩效概览
            self.display_performance_overview()

            # 显示详细分析
            self.display_detailed_analysis()

    def update_performance_metrics(self):
            """更新性能指标"""
            try:
                history_df = pd.DataFrame(st.session_state.portfolio_history)
                if not history_df.empty:
                    metrics = self.performance_analyzer.calculate_metrics(history_df)
                    st.session_state.performance_metrics = metrics
            except Exception as e:
                logger.error(f"Error updating performance metrics: {e}")
                st.error("性能指标更新出错")

    def display_performance_overview(self):
            """显示绩效概览"""
            try:
                metrics = st.session_state.performance_metrics

                st.subheader("绩效概览")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "累计收益",
                        f"{metrics.get('total_return', 0):.2f}%"
                    )
                with col2:
                    st.metric(
                        "年化收益",
                        f"{metrics.get('annual_return', 0):.2f}%"
                    )
                with col3:
                    st.metric(
                        "信息比率",
                        f"{metrics.get('information_ratio', 0):.2f}"
                    )
                with col4:
                    st.metric(
                        "胜率",
                        f"{metrics.get('win_rate', 0):.2f}%"
                    )

            except Exception as e:
                logger.error(f"Error displaying performance overview: {e}")
                st.error("绩效概览显示出错")

    def display_detailed_analysis(self):
            """显示详细分析"""
            try:
                st.subheader("详细分析")

                # 绘制收益分布图
                if hasattr(self.performance_analyzer, 'plot_return_distribution'):
                    fig = self.performance_analyzer.plot_return_distribution(
                        pd.DataFrame(st.session_state.portfolio_history)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # 显示月度收益表
                if hasattr(self.performance_analyzer, 'calculate_monthly_returns'):
                    monthly_returns = self.performance_analyzer.calculate_monthly_returns(
                        pd.DataFrame(st.session_state.portfolio_history)
                    )
                    st.dataframe(monthly_returns)

            except Exception as e:
                logger.error(f"Error displaying detailed analysis: {e}")
                st.error("详细分析显示出错")

    def auto_stop_settings_tab(self):
        """自动止损止盈设置标签页"""
        st.subheader("自动止损止盈设置")

        # 获取当前持仓
        positions = st.session_state.portfolio.get('positions', {})
        if not positions:
            st.warning("当前没有持仓")
            return

        # 选择股票
        symbol = st.selectbox(
            "选择股票",
            options=list(positions.keys()),
            format_func=lambda x: f"{x} ({positions[x].get('name', '')})"
        )

        # 启用/禁用自动止损止盈
        enabled = st.checkbox(
            "启用自动止损止盈",
            value=self.custom_strategy.stop_settings.get(symbol, {}).get('auto_stop_enabled', False)
        )

        if enabled:
            col1, col2 = st.columns(2)
            with col1:
                stop_loss = st.number_input(
                    "止损比例(%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=5.0,
                    step=0.1
                )

            with col2:
                take_profit = st.number_input(
                    "止盈比例(%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=10.0,
                    step=0.1
                )

            if st.button("保存设置"):
                self.custom_strategy.set_auto_stop_settings(
                    symbol=symbol,
                    stop_loss_pct=stop_loss,
                    take_profit_pct=take_profit,
                    enabled=enabled
                )
                st.success("设置已保存")

        # 显示当前设置
        if symbol in self.custom_strategy.stop_settings:
            settings = self.custom_strategy.stop_settings[symbol]
            st.write("当前设置:")
            st.json(settings)

    def market_sentiment_analysis_tab(self):
        """市场情绪分析标签页"""
        st.subheader("市场情绪分析")

        # 市场选择
        market = st.selectbox(
            "选择市场",
            ["CN", "US"],
            help="选择要分析的市场"
        )

        if st.button("分析市场情绪"):
            with st.spinner("正在分析市场情绪..."):
                try:
                    # 使用 custom_strategy 进行市场情绪分析
                    sentiment_results = self.loop.run_until_complete(
                        self.custom_strategy.get_market_sentiment_core(market)
                    )

                    if sentiment_results:
                        self.display_market_sentiment_results(sentiment_results)
                    else:
                        st.error("无法获取市场情绪数据")

                except Exception as e:
                    st.error(f"分析过程出错: {str(e)}")

    def display_market_sentiment_results(self, sentiment_results):
        """显示市场情绪分析结果"""
        # 显示市场整体情绪
        st.metric(
            "市场整体情绪",
            f"{sentiment_results['market_sentiment']:.2f}",
            help="-1(极度悲观) 到 1(极度乐观)"
        )

        # 显示核心股票情绪
        st.subheader("核心股票情绪排名")
        if sentiment_results['core_stocks']:
            sentiment_df = pd.DataFrame(sentiment_results['core_stocks'])

            # 格式化显示
            display_df = sentiment_df.copy()
            for col in ['sentiment_score', 'news_score', 'social_score',
                        'impact_score', 'price_change', 'volatility']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(3)

            st.dataframe(display_df)

            # 绘制情绪分布图
            fig = go.Figure()

            # 添加综合情绪柱状图
            fig.add_trace(go.Bar(
                x=sentiment_df['symbol'],
                y=sentiment_df['sentiment_score'],
                name='综合情绪',
                marker_color='lightblue'
            ))

            # 添加其他指标线
            for col, name in [
                ('news_score', '新闻情绪'),
                ('social_score', '社交情绪'),
                ('impact_score', '影响力')
            ]:
                if col in sentiment_df.columns:
                    fig.add_trace(go.Scatter(
                        x=sentiment_df['symbol'],
                        y=sentiment_df[col],
                        name=name,
                        mode='lines+markers'
                    ))

            fig.update_layout(
                title=f"{market}市场核心股票情绪分布",
                xaxis_title="股票代码",
                yaxis_title="情绪得分",
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

            # 导出功能
            csv = sentiment_df.to_csv(index=False)
            st.download_button(
                label="导出分析结果",
                data=csv,
                file_name=f"market_sentiment_{market}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

    def trading_strategy_tab(self):
        """交易策略标签页"""
        st.header("交易策略")

        # 选择策略类型
        strategy_type = st.selectbox(
            "策略类型",
            ["技术分析策略", "机器学习策略", "情绪分析策略"]
        )

        if strategy_type == "技术分析策略":
            self.technical_strategy_section()
        elif strategy_type == "机器学习策略":
            self.ml_strategy_section()
        elif strategy_type == "情绪分析策略":
            self.sentiment_strategy_section()

    def technical_strategy_section(self):
        """技术分析策略设置区域"""
        st.subheader("技术分析策略设置")

        col1, col2 = st.columns(2)

        with col1:
            # MA策略设置
            st.write("##### 移动平均线策略")
            ma_short = st.number_input("短期MA周期", min_value=5, value=5, step=1)
            ma_long = st.number_input("长期MA周期", min_value=10, value=20, step=1)

            # RSI策略设置
            st.write("##### RSI策略")
            rsi_period = st.number_input("RSI周期", min_value=1, value=14, step=1)
            rsi_upper = st.number_input("RSI上限", min_value=50, value=70, step=1)
            rsi_lower = st.number_input("RSI下限", min_value=1, value=30, step=1)

        with col2:
            # MACD策略设置
            st.write("##### MACD策略")
            macd_fast = st.number_input("MACD快线周期", min_value=1, value=12, step=1)
            macd_slow = st.number_input("MACD慢线周期", min_value=1, value=26, step=1)
            macd_signal = st.number_input("MACD信号线周期", min_value=1, value=9, step=1)

            # 布林带策略设置
            st.write("##### 布林带策略")
            bb_period = st.number_input("布林带周期", min_value=1, value=20, step=1)
            bb_std = st.number_input("标准差倍数", min_value=0.1, value=2.0, step=0.1)

        # 保存策略设置
        if st.button("保存技术分析策略设置"):
            try:
                strategy_settings = {
                    'ma': {'short': ma_short, 'long': ma_long},
                    'rsi': {'period': rsi_period, 'upper': rsi_upper, 'lower': rsi_lower},
                    'macd': {'fast': macd_fast, 'slow': macd_slow, 'signal': macd_signal},
                    'bollinger': {'period': bb_period, 'std': bb_std}
                }
                # 保存到配置
                self.config.update_strategy_settings('technical', strategy_settings)
                st.success("策略设置已保存")
            except Exception as e:
                st.error(f"保存策略设置失败: {str(e)}")

    def ml_strategy_section(self):
        """机器学习策略设置区域"""
        st.subheader("机器学习策略设置")

        col1, col2 = st.columns(2)

        with col1:
            # 模型设置
            st.write("##### 模型参数")
            model_type = st.selectbox(
                "模型类型",
                ["RandomForest", "XGBoost", "LSTM"]
            )
            prediction_horizon = st.number_input(
                "预测周期（天）",
                min_value=1,
                value=5,
                step=1
            )

            # 特征设置
            st.write("##### 特征选择")
            use_technical = st.checkbox("使用技术指标", value=True)
            use_fundamental = st.checkbox("使用基本面数据", value=False)
            use_sentiment = st.checkbox("使用情绪数据", value=True)

        with col2:
            # 训练设置
            st.write("##### 训练参数")
            train_period = st.number_input(
                "训练数据期限（天）",
                min_value=30,
                value=365,
                step=30
            )
            retrain_freq = st.number_input(
                "重新训练频率（天）",
                min_value=1,
                value=30,
                step=1
            )

            # 验证设置
            st.write("##### 验证设置")
            validation_size = st.slider(
                "验证集比例",
                min_value=0.1,
                max_value=0.3,
                value=0.2,
                step=0.05
            )

        # 保存策略设置
        if st.button("保存机器学习策略设置"):
            try:
                strategy_settings = {
                    'model': {
                        'type': model_type,
                        'prediction_horizon': prediction_horizon
                    },
                    'features': {
                        'technical': use_technical,
                        'fundamental': use_fundamental,
                        'sentiment': use_sentiment
                    },
                    'training': {
                        'period': train_period,
                        'retrain_freq': retrain_freq,
                        'validation_size': validation_size
                    }
                }
                # 保存到配置
                self.config.update_strategy_settings('ml', strategy_settings)
                st.success("策略设置已保存")
            except Exception as e:
                st.error(f"保存策略设置失败: {str(e)}")

    def sentiment_strategy_section(self):
        """情绪分析策略设置区域"""
        st.subheader("情绪分析策略设置")

        col1, col2 = st.columns(2)

        with col1:
            # 数据源设置
            st.write("##### 数据源设置")
            use_news = st.checkbox("新闻数据", value=True)
            use_social = st.checkbox("社交媒体", value=True)
            use_market = st.checkbox("市场数据", value=True)

            # 时间窗口设置
            st.write("##### 时间窗口")
            lookback_period = st.number_input(
                "回看期（天）",
                min_value=1,
                value=7,
                step=1
            )
            update_freq = st.number_input(
                "更新频率（分钟）",
                min_value=1,
                value=60,
                step=5
            )

        with col2:
            # 信号设置
            st.write("##### 信号设置")
            sentiment_threshold = st.slider(
                "情绪阈值",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1
            )

            # 权重设置
            st.write("##### 来源权重")
            news_weight = st.slider("新闻权重", 0.0, 1.0, 0.4, 0.1)
            social_weight = st.slider("社交媒体权重", 0.0, 1.0, 0.3, 0.1)
            market_weight = st.slider("市场数据权重", 0.0, 1.0, 0.3, 0.1)

        # 保存策略设置
        if st.button("保存情绪分析策略设置"):
            try:
                strategy_settings = {
                    'data_sources': {
                        'news': use_news,
                        'social': use_social,
                        'market': use_market
                    },
                    'timing': {
                        'lookback_period': lookback_period,
                        'update_freq': update_freq
                    },
                    'signals': {
                        'threshold': sentiment_threshold
                    },
                    'weights': {
                        'news': news_weight,
                        'social': social_weight,
                        'market': market_weight
                    }
                }
                # 保存到配置
                self.config.update_strategy_settings('sentiment', strategy_settings)
                st.success("策略设置已保存")
            except Exception as e:
                st.error(f"保存策略设置失败: {str(e)}")

    async def cleanup(self):
        """清理所有资源"""
        try:
            # 确保在正确的事件循环中清理资源
            if hasattr(self, 'market_scanner'):
                try:
                    await self.market_scanner.stop()
                except Exception as e:
                    self.logger.error(f"Error stopping market scanner: {e}")

            if hasattr(self, 'data_fetcher'):
                try:
                    await self.data_fetcher.cleanup()
                except Exception as e:
                    self.logger.error(f"Error cleaning up data fetcher: {e}")

            if hasattr(self, 'order_executor'):
                try:
                    await self.order_executor.cleanup()
                except Exception as e:
                    self.logger.error(f"Error cleaning up order executor: {e}")

            # 清理线程池
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)

            self.logger.info("所有资源已清理完成")

        except Exception as e:
            self.logger.error(f"清理资源时出错: {e}")
            raise

    def get_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """带错误处理的股票数据获取"""
        try:
            # 检查symbol是否在有效池中
            if not hasattr(self.market_scanner, 'active_symbols'):
                self.logger.error("MarketScanner未正确初始化")
                return None

            if symbol not in self.market_scanner.active_symbols:
                self.logger.warning(f"Symbol {symbol} not in active universe")
                return None

            return self.loop.run_until_complete(
                self.data_fetcher.get_historical_data(
                    symbol=symbol,
                    start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                    end_date=datetime.now().strftime('%Y-%m-%d')
                )
            )
        except Exception as e:
            self.logger.error(f"Error fetching stock data: {str(e)}")
            return None

def main():
    """主函数"""
    app = TradingApp()
    app.run()

if __name__ == "__main__":
    main()