# core/system.py
import streamlit as st
import logging
import time
from datetime import datetime, timedelta
import importlib
import os
from dotenv import load_dotenv
import sys
from pathlib import Path

# 导入核心组件
from core.config import Config
from core.analysis.portfolio import PortfolioAnalyzer
from core.ui.manager import UIManager
from core.utils.data_mock import create_demo_data
from core.data.manager import DataManager
from core.alert.manager import AlertManager
from core.strategy.unified_strategy import UnifiedStrategy
from core.data.direct_fetcher import DirectDataFetcher

logger = logging.getLogger(__name__)


def load_environment():
    env_file = ".env.txt"  # 注意文件名包含.txt后缀
    if os.path.exists(env_file):
        load_dotenv(env_file)
        print(f"已加载环境变量文件: {env_file}")
        # 验证变量是否正确加载
        print(f"TUSHARE_TOKEN: {os.environ.get('TUSHARE_TOKEN', '未设置')[:5]}...")
    else:
        print(f"警告: 环境变量文件不存在: {env_file}")


if __name__ == "__main__":
    load_environment()


class TradingSystem:
    """Trading System Core Class"""

    def __init__(self):
        """Initialize trading system"""
        logger.info("Initializing trading system...")

        # Load configuration
        self.config = Config()

        # Initialize components dictionary
        self.components = {}

        # Create UI manager
        self.ui = UIManager()

        # Initialize data manager
        self.data_manager = DataManager(self.config.get("data_sources", {}))

        # Initialize components
        self._init_components()

        # Get risk manager from components
        self.risk_manager = self.components.get('risk_manager')

        # Initialize trader
        try:
            from core.trading.trader import Trader
            self.trader = Trader()
            logger.info("Trader initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import Trader class: {e}")
            self.trader = None

        # Initialize alert system
        self.alert_manager = AlertManager(self)

        # Initialize portfolio analyzer
        self.portfolio_analyzer = PortfolioAnalyzer(self)

        # Initialize strategy manager
        try:
            self.strategy_manager = UnifiedStrategy(self)
        except Exception as e:
            logger.error(f"Strategy manager initialization failed: {e}")

            class SimpleStrategyManager:
                def __init__(self, system):
                    self.system = system

                def render_strategy_ui(self, system):
                    st.header("📈 Strategy Trading")
                    st.warning("Strategy manager failed to load, please check dependencies and logs")
                    st.info("Try reloading the page or contact system administrator")

            self.strategy_manager = SimpleStrategyManager(self)
            logger.info("Created simple strategy manager as fallback")

        # Initialize session state
        self._init_session_state()

        # Initialize application state
        self._init_state()

        # Create demo data
        self.demo_data = create_demo_data()

        logger.info("Trading system initialization completed")

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
        """Initialize session state variables"""
        # Portfolio
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = {
                'cash': 100000.0,
                'positions': {},
                'total_value': 100000.0,
                'last_update': datetime.now()
            }

        # Trade history
        if 'trades' not in st.session_state:
            st.session_state.trades = []

        # Portfolio history
        if 'portfolio_history' not in st.session_state:
            st.session_state.portfolio_history = []

        # Auto-refresh settings
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = 60  # Default 60 seconds
        if 'last_refresh_time' not in st.session_state:
            st.session_state.last_refresh_time = time.time()

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
        """运行系统"""
        # 渲染刷新控件
        self.ui.render_refresh_controls()

        # 渲染侧边栏
        self.ui.render_sidebar()

        # 渲染主内容区
        self.ui.render_main_tabs(self)

        # 处理自动刷新
        self._handle_auto_refresh()

    def _handle_auto_refresh(self):
        """处理自动刷新逻辑"""
        if st.session_state.auto_refresh:
            current_time = time.time()
            if current_time - st.session_state.last_refresh_time >= st.session_state.refresh_interval:
                st.session_state.last_refresh_time = current_time
                st.rerun()

    # ============= 功能方法 =============

    # 在system.py中修改get_stock_data方法
    def get_stock_data(self, symbol):
        """获取股票数据 - 改进版"""
        logger.info(f"尝试获取股票 {symbol} 的数据")

        # 1. 尝试直接获取器
        try:
            # 导入直接数据获取器


            # 检查是否已有实例
            if not hasattr(self, '_direct_fetcher'):
                self._direct_fetcher = DirectDataFetcher()

            # 获取数据
            data = self._direct_fetcher.get_stock_data(symbol)

            if data is not None and not data.empty:
                logger.info(f"直接获取器成功获取 {symbol} 数据")
                return data
        except Exception as e:
            logger.warning(f"直接获取器获取 {symbol} 数据失败: {e}")

        # 2. 尝试通过组件系统
        try:
            if "data_fetcher" in self.components:
                logger.info(f"通过组件尝试获取 {symbol} 数据")
                data = self.components["data_fetcher"].get_historical_data(
                    symbol=symbol,
                    start_date=(datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d'),
                    end_date=datetime.now().strftime('%Y-%m-%d')
                )
                if data is not None and not data.empty:
                    logger.info(f"组件系统成功获取 {symbol} 数据")
                    return data
        except Exception as e:
            logger.warning(f"组件系统获取 {symbol} 数据失败: {e}")

        # 3. 尝试使用模拟数据
        if hasattr(self, 'demo_data') and symbol in self.demo_data:
            logger.info(f"使用模拟数据获取 {symbol}")
            return self.demo_data[symbol]

        # 4. 所有方法都失败
        logger.error(f"无法获取 {symbol} 的数据")
        import pandas as pd
        return pd.DataFrame()  # 返回空DataFrame

    def get_realtime_price(self, symbol):
        """获取实时价格"""
        try:
            if "data_fetcher" in self.components:
                price_data = self.components["data_fetcher"].get_realtime_price(symbol)
                if price_data and 'price' in price_data:
                    return price_data['price']
        except Exception as e:
            logger.warning(f"获取{symbol}实时价格失败: {e}")

        # 降级为使用demo数据的最新价格
        if symbol in self.demo_data:
            return self.demo_data[symbol]['close'].iloc[-1]
        return None

    def run_market_scan(self, criteria):
        """运行市场扫描"""
        try:
            if "market_scanner" in self.components and "custom_strategy" in self.components:
                # 使用真实扫描器
                return self._run_real_market_scan(criteria)
        except Exception as e:
            logger.warning(f"使用真实扫描器失败: {e}")

        # 降级为模拟扫描
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
        """执行交易"""
        try:
            if "order_executor" in self.components:
                # 使用真实执行器执行订单
                result = self._execute_real_trade(order_data)
                if result.get('success'):
                    self._update_portfolio(order_data, result)
                return result
        except Exception as e:
            logger.error(f"执行订单失败: {e}")

        # 降级为模拟执行
        mock_result = self._execute_mock_trade(order_data)
        if mock_result.get('success'):
            self._update_portfolio(order_data, mock_result)
        return mock_result

    def _execute_real_trade(self, order_data):
        """使用真实执行器执行订单"""
        try:
            executor = self.components["order_executor"]
            result = executor.execute_order(
                symbol=order_data['symbol'],
                quantity=order_data['quantity'],
                price=order_data['price'],
                order_type=order_data['order_type']
            )
            return result
        except Exception as e:
            logger.error(f"真实交易执行错误: {e}")
            return {'success': False, 'message': f"交易执行错误: {e}"}

    def _execute_mock_trade(self, order_data):
        """Mock order execution"""
        symbol = order_data['symbol']
        quantity = order_data['quantity']
        price = order_data['price']
        direction = order_data['direction']

        if order_data['order_type'] == "Market Order":
            current_price = self.get_realtime_price(symbol)
            if current_price is None:
                return {'success': False, 'message': f"Failed to get price data for {symbol}"}
            price = current_price

        total_cost = abs(quantity) * price

        if direction == "Buy" and total_cost > st.session_state.portfolio['cash']:
            return {'success': False, 'message': "Insufficient funds"}

        if direction == "Sell":
            position = st.session_state.portfolio['positions'].get(symbol, {})
            if not position or position.get('quantity', 0) < quantity:
                return {'success': False, 'message': "Insufficient position"}

        return {
            'success': True,
            'symbol': symbol,
            'quantity': quantity if direction == "Buy" else -quantity,
            'price': price,
            'total_cost': total_cost,
            'timestamp': datetime.now(),
            'message': f"{direction} order executed successfully"
        }

    def _update_portfolio(self, order_data, result):
        """更新投资组合"""
        symbol = order_data['symbol']
        direction = order_data['direction']
        quantity = order_data['quantity']
        price = result['price']

        # 更新现金
        if direction == "买入":
            st.session_state.portfolio['cash'] -= result['total_cost']
        else:
            st.session_state.portfolio['cash'] += result['total_cost']

        # 更新持仓
        positions = st.session_state.portfolio['positions']
        if direction == "买入":
            if symbol not in positions:
                positions[symbol] = {
                    'quantity': 0,
                    'cost_basis': 0,
                    'current_price': price
                }

            # 更新持仓均价和数量
            old_quantity = positions[symbol]['quantity']
            old_cost_basis = positions[symbol]['cost_basis']
            new_quantity = old_quantity + quantity

            positions[symbol]['quantity'] = new_quantity
            positions[symbol]['cost_basis'] = (old_quantity * old_cost_basis + quantity * price) / new_quantity
            positions[symbol]['current_price'] = price
        else:
            # 卖出
            positions[symbol]['quantity'] -= quantity
            positions[symbol]['current_price'] = price

            # 如果持仓为0，删除该持仓
            if positions[symbol]['quantity'] <= 0:
                del positions[symbol]

        # 添加交易记录
        st.session_state.trades.append({
            'symbol': symbol,
            'direction': direction,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now(),
            'total': quantity * price
        })

        # 更新投资组合历史
        st.session_state.portfolio_history.append({
            'timestamp': datetime.now(),
            'cash': st.session_state.portfolio['cash'],
            'positions': {s: p.copy() for s, p in positions.items()}
        })

        # 更新最后更新时间
        st.session_state.portfolio['last_update'] = datetime.now()

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