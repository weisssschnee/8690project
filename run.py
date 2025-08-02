# run.py - 简化版
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objs as go

# 引入必要的类
from core.risk.manager import RiskManager
from core.market.scanner import MarketScanner
from core.data.fetcher import DataFetcher
from core.analysis.technical import TechnicalAnalyzer
from core.analysis.sentiment import SentimentAnalyzer
from core.trading.executor import OrderExecutor
from core.strategy.custom_strategy import CustomStrategy
from core.analysis.performance import PerformanceAnalyzer


class SimpleTradingApp:
    """简化版交易应用，确保基本功能可用"""

    def __init__(self):
        """初始化简化版应用"""
        # 设置基本配置
        self.config = self._create_default_config()

        # 初始化组件
        self._setup_components()

        # 初始化状态
        self._initialize_state()

        # 设置示例数据
        self._setup_demo_data()

    def _create_default_config(self):
        """创建默认配置"""
        return {
            'MAX_WORKERS': 4,
            'SCANNER_CONFIG': {
                'delay': 60,
                'markets': ['US'],
                'symbols': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
            },
            'API_CONFIG': {
                'endpoints': {
                    'market_data': 'your_market_data_endpoint',
                    'minute_data': 'your_minute_data_endpoint',
                    'symbols': 'your_symbols_endpoint'
                }
            },
            'RISK_CONFIG': {
                'MAX_POSITION_SIZE': 0.1,
                'MAX_SINGLE_LOSS': 0.02,
                'STOP_LOSS': 0.05,
                'TAKE_PROFIT': 0.1,
                'MAX_DRAWDOWN': 0.2,
                'RISK_FREE_RATE': 0.02,
                'POSITION_LIMITS': {'US': 0.8, 'CN': 0.8}
            },
            'MAX_POSITION_SIZE': 0.1
        }

    def _setup_components(self):
        """设置核心组件"""
        # 为简单起见，我们只创建必要的组件实例
        # 实际上这些会从相应的模块导入

        # 模拟组件
        self.risk_manager = RiskManager(self.config)
        self.technical_analyzer = TechnicalAnalyzer(self.config)

    def _initialize_state(self):
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

    def _setup_demo_data(self):
        """设置示例数据"""
        self.demo_data = {}
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA']

        for symbol in symbols:
            # 基础价格
            base_price = {
                'AAPL': 150, 'GOOGL': 2800, 'MSFT': 300,
                'AMZN': 3300, 'META': 300, 'TSLA': 800, 'NVDA': 400
            }.get(symbol, 100)

            # 生成时间序列数据
            dates = pd.date_range(end=datetime.now(), periods=100)
            prices = np.random.normal(base_price, base_price * 0.01, 100)
            volumes = np.random.normal(1000000, 200000, 100)

            # 创建DataFrame
            df = pd.DataFrame({
                'open': prices,
                'high': prices + np.random.uniform(0, base_price * 0.01, 100),
                'low': prices - np.random.uniform(0, base_price * 0.01, 100),
                'close': prices + np.random.uniform(-base_price * 0.005, base_price * 0.005, 100),
                'volume': np.abs(volumes)
            }, index=dates)

            self.demo_data[symbol] = df

    def run(self):
        """运行应用"""
        st.set_page_config(page_title="智能交易系统", page_icon="📈", layout="wide")

        # 侧边栏
        self._render_sidebar()

        # 主区域标签页
        tabs = st.tabs([
            "市场监控", "股票筛选", "交易执行", "持仓管理", "情绪分析"
        ])

        with tabs[0]:
            self._market_monitor_tab()
        with tabs[1]:
            self._stock_screener_tab()
        with tabs[2]:
            self._trading_execution_tab()
        with tabs[3]:
            self._portfolio_management_tab()
        with tabs[4]:
            self._sentiment_analysis_tab()

    def _render_sidebar(self):
        """渲染侧边栏"""
        with st.sidebar:
            st.title("智能交易系统")

            # 账户信息
            st.subheader("账户信息")
            st.metric("可用资金", f"${st.session_state.portfolio['cash']:,.2f}")
            st.metric("总资产", f"${st.session_state.portfolio['total_value']:,.2f}")

            # 市场选择
            st.subheader("市场设置")
            market = st.selectbox("选择市场", ["美股", "A股"], key="sidebar_market")

            # 简单设置
            st.subheader("系统设置")
            max_position = st.slider("最大持仓比例", 0.0, 1.0, 0.2, key="max_position")
            st.session_state.use_realtime_data = st.checkbox("使用实时数据", value=False)

            # 显示版本信息
            st.markdown("---")
            st.caption("智能交易系统 v1.0")

    def _market_monitor_tab(self):
        """市场监控标签页"""
        st.header("📈 市场监控")

        col1, col2 = st.columns([1, 3])

        with col1:
            # 股票输入
            symbol = st.text_input("股票代码", "AAPL")

            # 市场状态
            market_open = self._is_market_open()
            status_color = "green" if market_open else "red"
            st.markdown(f"市场状态: <span style='color:{status_color};font-weight:bold'>"
                        f"{'开市' if market_open else '休市'}</span>", unsafe_allow_html=True)

            # 基本操作按钮
            if st.button("获取实时数据"):
                st.session_state.last_symbol = symbol
                st.session_state.show_stock_data = True

        with col2:
            if symbol:
                self._display_stock_data(symbol)

    def _stock_screener_tab(self):
        """股票筛选标签页"""
        st.header("🔍 股票筛选")

        col1, col2 = st.columns(2)

        with col1:
            market = st.selectbox("交易市场", ["美股", "A股"], key="screener_market")
            symbols = st.text_input("股票代码（逗号分隔，留空查询全部）",
                                    "AAPL,MSFT,GOOGL" if market == "美股" else "600519.SH,000001.SZ",
                                    key="screener_symbols")

        with col2:
            days = st.slider("分析周期（天）", 1, 30, 5, key="screener_days")
            vol_threshold = st.number_input("成交量变化阈值(%)", 0.0, 100.0, 10.0, key="screener_vol")
            price_change = st.number_input("价格波动阈值(%)", 0.0, 50.0, 5.0, key="screener_price")

        if st.button("开始扫描", type="primary"):
            with st.spinner("正在扫描市场..."):
                results = self._run_stock_scan(symbols, market, days, vol_threshold, price_change)
                if results:
                    self._display_scan_results(results)
                else:
                    st.info("未找到符合条件的股票")

    def _trading_execution_tab(self):
        """交易执行标签页"""
        st.header("💰 交易执行")

        # 交易表单
        with st.form("trade_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                symbol = st.text_input("股票代码", "AAPL")
                quantity = st.number_input("数量", min_value=1, value=100)

            with col2:
                order_type = st.selectbox("订单类型", ["市价单", "限价单"])
                price = st.number_input("价格(限价单)", min_value=0.1, value=100.0)

            with col3:
                direction = st.radio("交易方向", ["买入", "卖出"])
                stop_loss = st.number_input("止损比例(%)", min_value=0.0, value=5.0)

            submitted = st.form_submit_button("提交订单")
            if submitted:
                # 处理订单
                result = self._submit_order(symbol, quantity, direction, order_type, price, stop_loss)
                if result.get('success'):
                    st.success(result.get('message', '订单已提交'))
                else:
                    st.error(result.get('message', '订单提交失败'))

        # 显示活跃订单
        st.subheader("最近交易")
        if st.session_state.trades:
            trades_df = pd.DataFrame(st.session_state.trades[-5:])
            st.dataframe(trades_df)
        else:
            st.info("暂无交易记录")

    def _portfolio_management_tab(self):
        """持仓管理标签页"""
        st.header("📊 持仓管理")

        # 持仓概况
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("可用资金", f"${st.session_state.portfolio['cash']:,.2f}")
        with col2:
            positions_value = sum(
                p.get('quantity', 0) * p.get('current_price', 0)
                for p in st.session_state.portfolio.get('positions', {}).values()
            )
            st.metric("持仓市值", f"${positions_value:,.2f}")
        with col3:
            total_value = st.session_state.portfolio['cash'] + positions_value
            st.metric("总资产", f"${total_value:,.2f}")

        # 持仓列表
        st.subheader("当前持仓")
        positions = st.session_state.portfolio.get('positions', {})
        if positions:
            positions_data = []
            for symbol, position in positions.items():
                positions_data.append({
                    "股票代码": symbol,
                    "持仓数量": position.get('quantity', 0),
                    "平均成本": position.get('cost_basis', 0),
                    "当前价格": position.get('current_price', 0),
                    "市值": position.get('quantity', 0) * position.get('current_price', 0),
                    "盈亏比例": (position.get('current_price', 0) / position.get('cost_basis', 1) - 1) * 100
                })

            df = pd.DataFrame(positions_data)
            st.dataframe(df)
        else:
            st.info("当前没有持仓")

    def _sentiment_analysis_tab(self):
        """情绪分析标签页"""
        st.header("🧠 市场情绪分析")

        col1, col2 = st.columns(2)

        with col1:
            market = st.selectbox("选择市场", ["US", "CN"], key="sentiment_market")

        with col2:
            symbol = st.text_input("输入股票代码", "AAPL", key="sentiment_symbol")

        # 股票情绪分析
        if symbol:
            if st.button("分析情绪"):
                with st.spinner("正在分析情绪数据..."):
                    sentiment_data = self._get_mock_sentiment(symbol)
                    self._display_sentiment(symbol, sentiment_data)

    def _display_stock_data(self, symbol):
        """显示股票数据"""
        # 获取股票数据
        if symbol in self.demo_data:
            data = self.demo_data[symbol].copy()

            # 显示基本信息
            current_price = data['close'].iloc[-1]
            prev_price = data['close'].iloc[-2]
            price_change = (current_price - prev_price) / prev_price * 100

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("当前价格", f"${current_price:.2f}", f"{price_change:.2f}%")
            with col2:
                st.metric("成交量", f"{int(data['volume'].iloc[-1]):,}")
            with col3:
                st.metric("52周范围", f"${data['low'].min():.2f} - ${data['high'].max():.2f}")

            # 显示K线图
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='K线'
            ))

            # 添加均线
            ma5 = data['close'].rolling(window=5).mean()
            ma20 = data['close'].rolling(window=20).mean()

            fig.add_trace(go.Scatter(
                x=data.index, y=ma5, name='MA5', line=dict(color='blue', width=1)
            ))
            fig.add_trace(go.Scatter(
                x=data.index, y=ma20, name='MA20', line=dict(color='orange', width=1)
            ))

            fig.update_layout(
                title=f'{symbol} 股价走势',
                yaxis_title='价格',
                xaxis_rangeslider_visible=False,
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # 技术分析
            st.subheader("技术分析")
            col1, col2, col3 = st.columns(3)

            # 计算技术指标
            rsi = 50 + np.random.normal(0, 10)  # 模拟RSI

            with col1:
                st.metric("RSI(14)", f"{rsi:.2f}")
            with col2:
                st.metric("MACD", "0.25")
            with col3:
                trend = "上升" if price_change > 0 else "下降"
                st.metric("趋势", trend)
        else:
            st.error(f"未找到{symbol}的数据")

    def _run_stock_scan(self, symbols, market, days, vol_threshold, price_threshold):
        """执行股票扫描"""
        results = []

        # 解析股票代码
        symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]

        # 处理A股代码格式
        if market == "A股":
            symbol_list = [s if s.endswith(('.SH', '.SZ')) else f"{s}.SH" for s in symbol_list]

        # 如果未指定股票，使用默认列表
        if not symbol_list:
            if market == "美股":
                symbol_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA']
            else:
                symbol_list = ['600519.SH', '000001.SZ', '600036.SH', '601318.SH']

        # 扫描股票
        for symbol in symbol_list:
            # 从演示数据获取
            if symbol in self.demo_data:
                data = self.demo_data[symbol].copy().tail(days)

                # 计算指标
                vol_change = ((data['volume'].iloc[-1] / data['volume'].iloc[0]) - 1) * 100
                price_change = ((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100

                # 应用筛选条件
                if abs(vol_change) >= vol_threshold and abs(price_change) >= price_threshold:
                    results.append({
                        'symbol': symbol,
                        'vol_change': vol_change,
                        'price_change': price_change,
                        'last_price': data['close'].iloc[-1],
                        'data': data
                    })

        return results

    def _display_scan_results(self, results):
        """显示扫描结果"""
        # 创建结果表格
        df = pd.DataFrame([{
            '代码': r['symbol'],
            '当前价': f"${r['last_price']:.2f}",
            '价格变化': f"{r['price_change']:.1f}%",
            '成交量变化': f"{r['vol_change']:.1f}%"
        } for r in results])

        st.dataframe(df, use_container_width=True)

        # 选择查看详情
        if len(results) > 0:
            selected = st.selectbox(
                "选择股票查看详情",
                options=[r['symbol'] for r in results]
            )

            if selected:
                selected_data = next(r for r in results if r['symbol'] == selected)
                data = selected_data['data']

                # 绘制图表
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name='K线'
                ))

                fig.update_layout(
                    title=f'{selected} 价格走势',
                    yaxis_title='价格',
                    xaxis_rangeslider_visible=False,
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

    def _submit_order(self, symbol, quantity, direction, order_type, price, stop_loss):
        """提交交易订单"""
        # 获取当前价格
        current_price = self._get_stock_price(symbol)
        if not current_price:
            return {'success': False, 'message': f"无法获取{symbol}的价格数据"}

        # 计算交易总额
        actual_price = price if order_type == "限价单" else current_price
        total_cost = quantity * actual_price

        # 检查资金是否足够
        if direction == "买入" and total_cost > st.session_state.portfolio['cash']:
            return {'success': False, 'message': "可用资金不足"}

        # 检查持仓是否足够
        if direction == "卖出":
            position = st.session_state.portfolio.get('positions', {}).get(symbol, {})
            if not position or position.get('quantity', 0) < quantity:
                return {'success': False, 'message': "持仓不足"}

        # 模拟订单执行
        order = {
            'symbol': symbol,
            'quantity': quantity if direction == "买入" else -quantity,
            'price': actual_price,
            'type': order_type,
            'timestamp': datetime.now(),
            'status': 'completed',
            'direction': direction
        }

        # 更新现金
        if direction == "买入":
            st.session_state.portfolio['cash'] -= total_cost
        else:
            st.session_state.portfolio['cash'] += total_cost

        # 更新持仓
        positions = st.session_state.portfolio.get('positions', {})
        if direction == "买入":
            if symbol not in positions:
                positions[symbol] = {
                    'quantity': 0,
                    'cost_basis': 0,
                    'current_price': actual_price
                }

            # 更新持仓均价
            old_value = positions[symbol]['quantity'] * positions[symbol]['cost_basis']
            new_value = quantity * actual_price
            new_quantity = positions[symbol]['quantity'] + quantity

            positions[symbol]['quantity'] = new_quantity
            positions[symbol]['cost_basis'] = (old_value + new_value) / new_quantity
            positions[symbol]['current_price'] = actual_price
        else:
            # 卖出
            positions[symbol]['quantity'] -= quantity
            positions[symbol]['current_price'] = actual_price

            # 如果持仓为0，删除该持仓
            if positions[symbol]['quantity'] <= 0:
                del positions[symbol]

        st.session_state.portfolio['positions'] = positions

        # 记录交易
        st.session_state.trades.append(order)

        return {'success': True, 'message': f"{direction} {quantity} 股 {symbol} 成功，价格: ${actual_price:.2f}"}

    def _get_stock_price(self, symbol):
        """获取股票价格"""
        if symbol in self.demo_data:
            return self.demo_data[symbol]['close'].iloc[-1]
        return None

    def _is_market_open(self):
        """检查市场是否开放"""
        # 简单模拟市场状态
        now = datetime.now()
        weekday = now.weekday()
        hour = now.hour

        # 周一至周五，9:30-16:00视为开市
        return weekday < 5 and 9 <= hour < 16

    def _get_mock_sentiment(self, symbol):
        """获取模拟的情绪数据"""
        import random

        # 模拟情绪数据
        sentiment = {
            'score': random.uniform(-1, 1),
            'news_score': random.uniform(-1, 1),
            'social_score': random.uniform(-1, 1),
            'technical_score': random.uniform(-1, 1),
            'news_count': random.randint(5, 30),
            'social_mentions': random.randint(10, 100),
            'timestamp': datetime.now()
        }

        return sentiment

    def _display_sentiment(self, symbol, data):
        """显示情绪分析结果"""
        st.subheader(f"{symbol} 情绪分析")

        # 计算综合情绪状态
        score = data['score']
        sentiment_status = "极度乐观" if score > 0.7 else \
            "乐观" if score > 0.3 else \
                "中性" if score > -0.3 else \
                    "悲观" if score > -0.7 else "极度悲观"

        # 显示情绪分数
        col1, col2 = st.columns(2)
        with col1:
            # 使用颜色指示情绪
            color = "green" if score > 0.3 else "orange" if score > -0.3 else "red"
            st.markdown(f"### 情绪得分: <span style='color:{color}'>{score:.2f}</span>", unsafe_allow_html=True)
            st.markdown(f"### 情绪状态: <span style='color:{color}'>{sentiment_status}</span>", unsafe_allow_html=True)

        with col2:
            # 显示细分情绪得分
            st.metric("新闻情绪", f"{data['news_score']:.2f}")
            st.metric("社交媒体情绪", f"{data['social_score']:.2f}")
            st.metric("技术指标情绪", f"{data['technical_score']:.2f}")

        # 绘制雷达图
        categories = ['新闻情绪', '社交情绪', '技术情绪']
        values = [data['news_score'], data['social_score'], data['technical_score']]

        # 归一化值到[0,1]范围
        normalized_values = [(v + 1) / 2 for v in values]

        # 确保雷达图是闭合的
        categories.append(categories[0])
        normalized_values.append(normalized_values[0])

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=normalized_values,
            theta=categories,
            fill='toself',
            name='情绪分析'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # 添加分析说明
        st.subheader("分析详情")
        st.write(f"分析时间: {data['timestamp']}")
        st.write(f"分析新闻数量: {data['news_count']}")
        st.write(f"社交媒体提及: {data['social_mentions']}")


# 主函数
def main():
    app = SimpleTradingApp()
    app.run()


if __name__ == "__main__":
    main()