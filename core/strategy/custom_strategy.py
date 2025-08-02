# core/strategy/custom_strategy.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import asyncio
from scipy import stats
import plotly.graph_objs as go
import aiohttp
import os


class CustomStrategy:
    def __init__(self):
        self.scanner = None
        self.technical_analyzer = None
        self.sentiment_analyzer = None
        self.stop_settings = {}
        self.logger = logging.getLogger(__name__)
        self.signals = {}  # 用于存储策略信号
        self.initialized = False
        self.market_scanner = None
        self.data_fetcher = None

        # 策略状态存储
        self.price_triggers = {}
        self.volume_price_triggers = {}
        self.stop_settings = {}
        self.market_scan_settings = {}

        # 活跃交易状态
        self.active_trades = {}

    def set_scanner(self, scanner):
        """设置市场扫描器"""
        self.scanner = scanner

    def set_technical_analyzer(self, analyzer):
        """设置技术分析器"""
        self.technical_analyzer = analyzer

    def set_sentiment_analyzer(self, analyzer):
        """设置情绪分析器"""
        self.sentiment_analyzer = analyzer

    def set_auto_stop_settings(self, symbol: str, stop_loss_pct: float,
                             take_profit_pct: float, enabled: bool = True):
        """设置自动止损止盈"""
        self.stop_settings[symbol] = {
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'enabled': enabled
        }

    async def initialize(self):
        """异步初始化策略"""
        if self.initialized:
            return

        try:
            if self.scanner is None:
                raise ValueError("Scanner not set")

            # 在这里添加其他初始化逻辑
            self.initialized = True
            self.logger.info("Custom strategy initialized successfully")

        except Exception as e:
            self.logger.error(f"Strategy initialization error: {e}")
            raise

    async def generate_signals(self, symbol: str, data: pd.DataFrame) -> Dict:
        """生成交易信号"""
        if not self.initialized:
            await self.initialize()

        try:
            # 基础信号生成逻辑
            signals = {
                'buy': [],
                'sell': [],
                # 其他信号...
            }

            # 检查价格触发信号
            price_signals = await self._check_price_triggers()
            if price_signals:
                signals['price_triggers'] = price_signals

            # 检查量价触发信号
            volume_price_signals = await self._check_volume_price_triggers()
            if volume_price_signals:
                signals['volume_price_triggers'] = volume_price_signals

            # 检查止损止盈信号
            stop_signals = await self.monitor_positions()
            if stop_signals:
                signals['stop_signals'] = stop_signals

            return signals

        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {e}")
            return {}

    def set_price_trigger(self, symbol: str, price_triggers: Dict[str, float]):
        """
        设置价格触发器
        price_triggers格式: {
            'buy_above': 价格,  # 价格上涨到此价位买入
            'buy_below': 价格,  # 价格下跌到此价位买入
            'sell_above': 价格, # 价格上涨到此价位卖出
            'sell_below': 价格  # 价格下跌到此价位卖出
        }
        """
        if symbol not in self.price_triggers:
            self.price_triggers[symbol] = {}
        self.price_triggers[symbol].update(price_triggers)

    def set_volume_price_trigger(self, symbol: str, settings: Dict):
        """
        设置量价触发器
        settings格式: {
            'volume_percent': float,  # 成交量百分比阈值
            'price_change': float,    # 价格变化百分比
            'lookback_minutes': int,  # 回看时间（分钟）
            'direction': str         # 'up' 或 'down'
        }
        """
        if symbol not in self.volume_price_triggers:
            self.volume_price_triggers[symbol] = {}
        self.volume_price_triggers[symbol].update(settings)

    def set_stop_loss_profit(self, symbol: str, position_settings: Dict):
        """
        设置止损止盈
        position_settings格式: {
            'stop_loss_type': str,          # 'fixed' 或 'trailing'
            'stop_loss_value': float,       # 止损价格或百分比
            'take_profit_type': str,        # 'fixed' 或 'trailing'
            'take_profit_value': float,     # 止盈价格或百分比
            'enabled': bool,                # 是否启用
            'partial_exit': bool,           # 是否允许部分退出
            'exit_portions': List[Dict],    # 分批退出设置
            'time_stops': Dict,             # 时间止损设置
            'volume_stops': Dict            # 量能止损设置
        }
        """
        if symbol not in self.stop_settings:
            self.stop_settings[symbol] = {}

        self.stop_settings[symbol].update({
            'position_settings': position_settings,
            'initial_price': None,  # 将在开仓时设置
            'highest_price': None,  # 用于追踪止损
            'lowest_price': None,  # 用于追踪止损
            'exit_records': [],  # 记录部分退出历史
            'last_update': datetime.now()
        })

    async def monitor_positions(self) -> List[Dict]:
        """监控所有持仓的止损止盈状态"""
        signals = []

        for symbol, settings in self.stop_settings.items():
            if not settings.get('position_settings', {}).get('enabled', False):
                continue

            current_price = await self._get_current_price(symbol)
            if current_price is None:
                continue

            position_settings = settings['position_settings']
            initial_price = settings['initial_price']

            if initial_price is None:
                continue

            # 更新最高/最低价格
            settings['highest_price'] = max(settings.get('highest_price', current_price), current_price)
            settings['lowest_price'] = min(settings.get('lowest_price', current_price), current_price)

            # 检查止损条件
            stop_loss_signal = self._check_stop_loss(
                symbol, current_price, settings
            )
            if stop_loss_signal:
                signals.append(stop_loss_signal)

            # 检查止盈条件
            take_profit_signal = self._check_take_profit(
                symbol, current_price, settings
            )
            if take_profit_signal:
                signals.append(take_profit_signal)

            # 检查时间止损
            time_stop_signal = self._check_time_stops(
                symbol, settings
            )
            if time_stop_signal:
                signals.append(time_stop_signal)

            # 检查量能止损
            volume_stop_signal = await self._check_volume_stops(
                symbol, settings
            )
            if volume_stop_signal:
                signals.append(volume_stop_signal)

        return signals

    def _check_stop_loss(self, symbol: str, current_price: float, settings: Dict) -> Optional[Dict]:
        """检查止损条件"""
        position_settings = settings['position_settings']
        initial_price = settings['initial_price']

        if position_settings['stop_loss_type'] == 'fixed':
            stop_price = position_settings['stop_loss_value']
            if current_price <= stop_price:
                return {
                    'symbol': symbol,
                    'action': 'sell',
                    'reason': 'fixed_stop_loss',
                    'price': current_price
                }

        elif position_settings['stop_loss_type'] == 'trailing':
            stop_percentage = position_settings['stop_loss_value']
            highest_price = settings['highest_price']
            stop_price = highest_price * (1 - stop_percentage / 100)

            if current_price <= stop_price:
                return {
                    'symbol': symbol,
                    'action': 'sell',
                    'reason': 'trailing_stop_loss',
                    'price': current_price
                }

        return None

    def _check_take_profit(self, symbol: str, current_price: float, settings: Dict) -> Optional[Dict]:
        """检查止盈条件"""
        position_settings = settings['position_settings']
        initial_price = settings['initial_price']

        if position_settings['partial_exit']:
            exit_portions = position_settings['exit_portions']
            for portion in exit_portions:
                if not portion.get('executed', False):
                    target_price = initial_price * (1 + portion['profit_target'] / 100)
                    if current_price >= target_price:
                        portion['executed'] = True
                        return {
                            'symbol': symbol,
                            'action': 'sell',
                            'reason': 'partial_take_profit',
                            'price': current_price,
                            'portion': portion['size']
                        }

        elif position_settings['take_profit_type'] == 'fixed':
            profit_price = position_settings['take_profit_value']
            if current_price >= profit_price:
                return {
                    'symbol': symbol,
                    'action': 'sell',
                    'reason': 'fixed_take_profit',
                    'price': current_price
                }

        elif position_settings['take_profit_type'] == 'trailing':
            profit_percentage = position_settings['take_profit_value']
            lowest_price = settings['lowest_price']
            profit_price = lowest_price * (1 + profit_percentage / 100)

            if current_price <= profit_price:
                return {
                    'symbol': symbol,
                    'action': 'sell',
                    'reason': 'trailing_take_profit',
                    'price': current_price
                }

        return None

    def _check_time_stops(self, symbol: str, settings: Dict) -> Optional[Dict]:
        """检查时间止损条件"""
        time_stops = settings['position_settings'].get('time_stops', {})
        if not time_stops:
            return None

        current_time = datetime.now()
        position_time = settings['last_update']

        # 检查持仓时间限制
        if 'max_hold_time' in time_stops:
            max_hold_minutes = time_stops['max_hold_time']
            if (current_time - position_time).total_seconds() / 60 >= max_hold_minutes:
                return {
                    'symbol': symbol,
                    'action': 'sell',
                    'reason': 'time_stop',
                    'price': None
                }

        return None

    async def _check_volume_stops(self, symbol: str, settings: Dict) -> Optional[Dict]:
        """检查量能止损条件"""
        volume_stops = settings['position_settings'].get('volume_stops', {})
        if not volume_stops:
            return None

        # 获取最新成交量数据
        current_volume = await self._get_current_volume(symbol)
        if current_volume is None:
            return None

        # 检查成交量条件
        if 'min_volume' in volume_stops and current_volume < volume_stops['min_volume']:
            return {
                'symbol': symbol,
                'action': 'sell',
                'reason': 'volume_stop',
                'price': None
            }

        return None

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        try:
            data = await self.get_minute_data(symbol, period=1)
            if data is not None and not data.empty:
                return data['close'].iloc[-1]
            return None
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return None

    async def _get_current_volume(self, symbol: str) -> Optional[float]:
        """获取当前成交量"""
        try:
            data = await self.get_minute_data(symbol, period=1)
            if data is not None and not data.empty:
                return data['volume'].iloc[-1]
            return None
        except Exception as e:
            self.logger.error(f"Error getting current volume: {e}")
            return None

    async def get_minute_data(self, symbol: str, period: int) -> Optional[pd.DataFrame]:
        """获取分钟级数据"""
        try:
            # 使用scanner获取数据
            if self.scanner:
                data = await self.scanner.get_minute_data(symbol, period)
                if data is not None and not data.empty:
                    return self._standardize_data(data)
            return None
        except Exception as e:
            self.logger.error(f"Error fetching minute data for {symbol}: {e}")
            return None

    def _standardize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """标准化数据格式"""
        try:
            standardized = pd.DataFrame()

            # 处理时间戳
            if isinstance(data.index, pd.DatetimeIndex):
                standardized['timestamp'] = data.index
            elif 'timestamp' in data.columns:
                standardized['timestamp'] = pd.to_datetime(data['timestamp'])
            elif 'date' in data.columns:
                standardized['timestamp'] = pd.to_datetime(data['date'])

            # 复制价格和成交量数据
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in data.columns:
                    standardized[col] = data[col]
                elif col.capitalize() in data.columns:
                    standardized[col] = data[col.capitalize()]
                else:
                    standardized[col] = 0  # 设置默认值

            return standardized

        except Exception as e:
            self.logger.error(f"Error standardizing data: {e}")
            raise

    async def _check_price_triggers(self) -> List[Dict]:
        """检查价格触发条件"""
        signals = []

        for symbol, triggers in self.price_triggers.items():
            try:
                current_price = await self._get_current_price(symbol)
                if current_price is None:
                    continue

                # 检查买入条件
                if 'buy_above' in triggers and current_price >= triggers['buy_above']:
                    signals.append(self._format_trade_signal({
                        'symbol': symbol,
                        'action': 'buy',
                        'reason': 'price_trigger_above',
                        'price': current_price
                    }))

                if 'buy_below' in triggers and current_price <= triggers['buy_below']:
                    signals.append(self._format_trade_signal({
                        'symbol': symbol,
                        'action': 'buy',
                        'reason': 'price_trigger_below',
                        'price': current_price
                    }))

                # 检查卖出条件
                if 'sell_above' in triggers and current_price >= triggers['sell_above']:
                    signals.append(self._format_trade_signal({
                        'symbol': symbol,
                        'action': 'sell',
                        'reason': 'price_trigger_above',
                        'price': current_price
                    }))

                if 'sell_below' in triggers and current_price <= triggers['sell_below']:
                    signals.append(self._format_trade_signal({
                        'symbol': symbol,
                        'action': 'sell',
                        'reason': 'price_trigger_below',
                        'price': current_price
                    }))

            except Exception as e:
                self.logger.error(f"Error checking price triggers for {symbol}: {e}")

        return signals

    async def _check_volume_price_triggers(self) -> List[Dict]:
        """检查量价触发条件"""
        signals = []

        for symbol, settings in self.volume_price_triggers.items():
            try:
                # 获取分钟数据
                data = await self.get_minute_data(
                    symbol=symbol,
                    period=settings['lookback_minutes']
                )

                if data is None or data.empty:
                    continue

                # 计算成交量变化
                current_volume = data['volume'].iloc[-1]
                avg_volume = data['volume'].mean()
                volume_change = (current_volume / avg_volume - 1) * 100

                # 计算价格变化
                price_change = (
                                       data['close'].iloc[-1] / data['close'].iloc[0] - 1
                               ) * 100

                # 检查触发条件
                volume_condition = volume_change >= settings['volume_percent']
                price_condition = (
                        (settings['direction'] == 'up' and price_change >= settings['price_change']) or
                        (settings['direction'] == 'down' and price_change <= -settings['price_change'])
                )

                if volume_condition and price_condition:
                    signals.append(self._format_trade_signal({
                        'symbol': symbol,
                        'action': 'buy' if settings['direction'] == 'up' else 'sell',
                        'reason': 'volume_price_trigger',
                        'price': data['close'].iloc[-1],
                        'volume': current_volume,
                        'details': {
                            'volume_change': volume_change,
                            'price_change': price_change
                        }
                    }))

            except Exception as e:
                self.logger.error(f"Error checking volume price triggers for {symbol}: {e}")

        return signals

    def _format_trade_signal(self, signal: Dict) -> Dict:
        """格式化交易信号"""
        return {
            'timestamp': datetime.now(),
            'symbol': signal['symbol'],
            'action': signal['action'],
            'reason': signal['reason'],
            'price': signal.get('price'),
            'volume': signal.get('volume'),
            'details': signal.get('details', {})
        }

    def get_strategy_status(self) -> Dict:
        """获取策略状态概览"""
        return {
            'active_triggers': {
                'price': len(self.price_triggers),
                'volume_price': len(self.volume_price_triggers),
                'stop_settings': len(self.stop_settings)
            },
            'monitoring_symbols': list(set(
                list(self.price_triggers.keys()) +
                list(self.volume_price_triggers.keys()) +
                list(self.stop_settings.keys())
            )),
            'last_update': datetime.now()
        }

    async def get_market_sentiment_core(self, symbol: str) -> Dict:
        """获取市场情绪核心数据"""
        try:
            if not self.sentiment_analyzer:
                self.logger.warning("Sentiment analyzer not initialized")
                return {
                    'status': 'neutral',
                    'score': 0.5,
                    'confidence': 0.0,
                    'signals': {},
                    'timestamp': datetime.now()
                }

            sentiment_data = await self.sentiment_analyzer.get_sentiment(symbol)

            # 如果无法获取情绪数据，返回中性结果
            if not sentiment_data:
                return {
                    'status': 'neutral',
                    'score': 0.5,
                    'confidence': 0.0,
                    'signals': {},
                    'timestamp': datetime.now()
                }

            return sentiment_data

        except Exception as e:
            self.logger.error(f"Error getting market sentiment core: {e}")
            return {
                'status': 'error',
                'score': 0.5,
                'confidence': 0.0,
                'signals': {},
                'timestamp': datetime.now(),
                'error': str(e)
            }

    async def search_symbols(self, query: str) -> List[Dict]:
        """全网搜索股票符号"""
        try:
            if not query or len(query) < 1:
                return []

            query = query.upper().strip()
            results = []

            # 尝试从API搜索
            try:
                if self.api_endpoints.get('symbol_search'):
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                                self.api_endpoints['symbol_search'],
                                params={'query': query}
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                if data and 'symbols' in data:
                                    return data['symbols']
            except Exception as e:
                self.logger.warning(f"API search failed: {e}")

            # 如果API搜索失败，使用多个数据源进行搜索
            search_tasks = [
                self._search_us_stocks(query),
                self._search_global_stocks(query),
                self._search_crypto(query)
            ]

            results = []
            for task in asyncio.as_completed(search_tasks):
                try:
                    data = await task
                    if data:
                        results.extend(data)
                except Exception as e:
                    self.logger.error(f"Error in search task: {e}")

            # 去重并排序
            seen = set()
            unique_results = []
            for item in results:
                symbol = item.get('symbol')
                if symbol and symbol not in seen:
                    seen.add(symbol)
                    unique_results.append(item)

            return unique_results[:20]  # 限制返回结果数量

        except Exception as e:
            self.logger.error(f"Error searching symbols: {e}")
            return []

    async def _search_us_stocks(self, query: str) -> List[Dict]:
        """搜索美股"""
        try:
            # 使用多个数据源
            sources = [
                ('US_STOCKS', self.api_endpoints.get('us_stocks_search')),
                ('NYSE', self.api_endpoints.get('nyse_search')),
                ('NASDAQ', self.api_endpoints.get('nasdaq_search'))
            ]

            results = []
            for source_name, endpoint in sources:
                if endpoint:
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                    endpoint,
                                    params={'q': query}
                            ) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    if data:
                                        for item in data:
                                            results.append({
                                                'symbol': item.get('symbol'),
                                                'name': item.get('name'),
                                                'exchange': source_name,
                                                'type': 'stock',
                                                'country': 'US'
                                            })
                    except Exception as e:
                        self.logger.warning(f"Error searching {source_name}: {e}")

            # 如果API都失败了，使用本地模糊搜索
            if not results:
                demo_stocks = {
                    'AAPL': 'Apple Inc.',
                    'GOOGL': 'Alphabet Inc.',
                    'MSFT': 'Microsoft Corporation',
                    'AMZN': 'Amazon.com Inc.',
                    'META': 'Meta Platforms Inc.',
                    'TSLA': 'Tesla Inc.',
                    'NVDA': 'NVIDIA Corporation',
                    'JPM': 'JPMorgan Chase & Co.',
                    'V': 'Visa Inc.',
                    'WMT': 'Walmart Inc.'
                }

                for symbol, name in demo_stocks.items():
                    if query in symbol or query.lower() in name.lower():
                        results.append({
                            'symbol': symbol,
                            'name': name,
                            'exchange': 'US_DEMO',
                            'type': 'stock',
                            'country': 'US'
                        })

            return results

        except Exception as e:
            self.logger.error(f"Error in US stocks search: {e}")
            return []

    async def _search_global_stocks(self, query: str) -> List[Dict]:
        """搜索全球股票"""
        try:
            results = []

            # 使用配置的全球市场数据源
            global_endpoints = self.api_endpoints.get('global_markets', {})

            for market, endpoint in global_endpoints.items():
                if endpoint:
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                    endpoint,
                                    params={'query': query}
                            ) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    if data:
                                        results.extend(data)
                    except Exception as e:
                        self.logger.warning(f"Error searching {market}: {e}")

            return results

        except Exception as e:
            self.logger.error(f"Error in global stocks search: {e}")
            return []

    async def _search_crypto(self, query: str) -> List[Dict]:
        """搜索加密货币"""
        try:
            results = []

            crypto_endpoint = self.api_endpoints.get('crypto_search')
            if crypto_endpoint:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                                crypto_endpoint,
                                params={'query': query}
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                if data:
                                    results.extend(data)
                except Exception as e:
                    self.logger.warning(f"Error searching crypto: {e}")

            return results

        except Exception as e:
            self.logger.error(f"Error in crypto search: {e}")
            return []

    async def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """获取股票详细信息"""
        try:
            # 尝试从不同数据源获取信息
            sources = [
                ('quote', self.api_endpoints.get('quote')),
                ('profile', self.api_endpoints.get('company_profile')),
                ('stats', self.api_endpoints.get('key_stats'))
            ]

            info = {}
            for source_type, endpoint in sources:
                if endpoint:
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                    endpoint,
                                    params={'symbol': symbol}
                            ) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    if data:
                                        info[source_type] = data
                    except Exception as e:
                        self.logger.warning(f"Error getting {source_type} for {symbol}: {e}")

            # 如果没有获取到数据，使用演示数据
            if not info and symbol in self.demo_data:
                data = self.demo_data[symbol]
                latest = data.iloc[-1]
                info = {
                    'symbol': symbol,
                    'price': latest['close'],
                    'change': (latest['close'] - data.iloc[-2]['close']) / data.iloc[-2]['close'],
                    'volume': latest['volume'],
                    'timestamp': latest['timestamp']
                }

            return info or None

        except Exception as e:
            self.logger.error(f"Error getting symbol info: {e}")
            return None

