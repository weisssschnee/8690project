# core/market/scanner.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
import logging
from datetime import datetime, timedelta
from .base_scanner import BaseMarketScanner
import asyncio
import aiohttp


class MarketScanner(BaseMarketScanner):
    def __init__(self, config=None):
        # 确保配置是字典
        config = config or {}

        # 调用父类的初始化
        super().__init__(config)

        # 保存对自身的引用作为base_scanner

        # API 端点设置
        self.api_endpoints = self.config.get('API_CONFIG', {}).get('endpoints', {})

        # 设置演示数据
        self._setup_demo_data()

        # 从父类的SCANNER_CONFIG获取配置
        self.scan_delay = self.SCANNER_CONFIG.get('delay', 60)
        self.max_workers = self.config.get('MAX_WORKERS', 4)

        # 缓存设置
        self.data_cache = {}
        self.cache_timeout = timedelta(minutes=5)
        self.last_update = {}

        # 扫描任务
        self.scan_task = None
        self.logger = logging.getLogger(__name__)

        # 初始化标志
        self.initialized = False
        self.is_running = False
        self.active_symbols = set()

    def _setup_demo_data(self):
        """设置演示数据"""
        self.demo_data = {}
        demo_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']

        for symbol in demo_symbols:
            # 创建初始价格
            base_price = {
                'AAPL': 150, 'GOOGL': 2800,
                'MSFT': 300, 'AMZN': 3300,
                'META': 300
            }.get(symbol, 100)

            # 生成时间序列数据
            times = pd.date_range(end=datetime.now(), periods=100, freq='1min')
            prices = np.random.normal(loc=base_price, scale=base_price * 0.01, size=100)
            volumes = np.random.normal(loc=1000000, scale=200000, size=100)

            # 创建DataFrame
            df = pd.DataFrame({
                'timestamp': times,
                'open': prices,
                'high': prices + np.random.uniform(0, base_price * 0.01, 100),
                'low': prices - np.random.uniform(0, base_price * 0.01, 100),
                'close': prices + np.random.uniform(-base_price * 0.005, base_price * 0.005, 100),
                'volume': np.abs(volumes)
            })

            # 确保价格的合理性
            df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, base_price * 0.005, 100)
            df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, base_price * 0.005, 100)

            self.demo_data[symbol] = df

    async def start_scanning(self):
        """启动市场扫描"""
        if not self.initialized:
            await self.initialize()

        if not self.is_running:
            self.is_running = True
            self.scan_task = asyncio.create_task(self._continuous_scan())
            self.logger.info("Market scanning started")

    async def stop_scanning(self):
        """停止市场扫描"""
        self.is_running = False
        if self.scan_task:
            try:
                self.scan_task.cancel()
                await self.scan_task
            except asyncio.CancelledError:
                pass
            self.scan_task = None
        self.logger.info("Market scanning stopped")

    async def start(self):
        """启动扫描器"""
        try:
            if not self.initialized:
                await self.initialize()

            self.is_running = True
            self.scan_task = asyncio.create_task(self._continuous_scan())
            self.logger.info("Market scanner started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start market scanner: {e}")
            raise

    async def stop(self):
        """停止扫描器"""
        try:
            self.logger.info("Stopping market scanner...")
            self.is_running = False

            if self.scan_task:
                self.scan_task.cancel()
                try:
                    await self.scan_task
                except asyncio.CancelledError:
                    self.logger.info("扫描任务被取消")
                finally:
                    self.scan_task = None

            self.logger.info("Market scanner stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping market scanner: {e}")
            raise

    async def initialize(self):
        try:
            # 优先使用配置中的预定义股票池
            if self.config.get('SCANNER_CONFIG', {}).get('symbols'):
                self.active_symbols = set(self.config['SCANNER_CONFIG']['symbols'])
                self.logger.info("使用配置中的预定义股票池")
            else:
                # 备用逻辑：尝试从API获取（保持原逻辑但简化）
                symbols = await self._fetch_symbols_from_api()
                self.active_symbols = symbols if symbols else set(['AAPL', 'MSFT'])
        except Exception as e:
            self.logger.error(f"初始化失败: {e}")
            self.active_symbols = set(['AAPL', 'MSFT'])  # 硬编码兜底

    async def _continuous_scan(self):
        """持续扫描市场"""
        try:
            while self.is_running:
                scan_params = {'period': 1}  # 默认扫描参数
                await self.scan_market('US', scan_params)  # 默认扫描美股市场
                await asyncio.sleep(self.SCANNER_CONFIG.get('delay', 60))
        except asyncio.CancelledError:
            self.logger.info("Continuous scan cancelled")
        except Exception as e:
            self.logger.error(f"Error in continuous scan: {e}")

    async def scan_market(self, market: str, scan_params: Dict) -> List[Dict]:
        """扫描市场"""
        try:
            if not self.active_symbols:
                await self.initialize()

            results = []
            scan_start_time = datetime.now()

            # 使用并发获取数据
            tasks = []
            for symbol in self.active_symbols:
                task = asyncio.create_task(self.get_minute_data(symbol, scan_params.get('period', 1)))
                tasks.append((symbol, task))

            for symbol, task in tasks:
                try:
                    data = await task
                    if data is not None and not data.empty:
                        analysis = self._analyze_data(data, scan_params)
                        if analysis['matched']:
                            results.append({
                                'symbol': symbol,
                                'data': analysis,
                                'timestamp': datetime.now()
                            })
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {e}")

            scan_duration = (datetime.now() - scan_start_time).total_seconds()
            self.logger.info(f"Market scan completed in {scan_duration:.2f} seconds")
            return results

        except Exception as e:
            self.logger.error(f"Error scanning market: {e}")
            return []

    async def get_minute_data(self, symbol: str, period: int) -> Optional[pd.DataFrame]:
        """获取分钟级数据"""
        try:
            # 检查缓存
            if symbol in self.data_cache:
                cache_time = self.last_update.get(symbol)
                if cache_time and datetime.now() - cache_time < self.cache_timeout:
                    return self.data_cache[symbol]

            # 如果没有配置API endpoints或者是演示模式，使用演示数据
            if not self.api_endpoints.get('minute_data') or symbol in self.demo_data:
                if symbol in self.demo_data:
                    df = self.demo_data[symbol].copy()
                    self.data_cache[symbol] = df
                    self.last_update[symbol] = datetime.now()
                    return df
                return None

            # 如果有API配置，尝试获取实际数据
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                            self.api_endpoints['minute_data'],
                            params={'symbol': symbol, 'period': period}
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            df = pd.DataFrame(data)

                            # 更新缓存
                            self.data_cache[symbol] = df
                            self.last_update[symbol] = datetime.now()

                            return df
                        else:
                            self.logger.warning(f"Failed to get data for {symbol}, status: {response.status}")
                            # 尝试使用演示数据作为后备
                            if symbol in self.demo_data:
                                return self.demo_data[symbol].copy()
                            return None
            except aiohttp.ClientError as e:
                self.logger.error(f"Network error fetching data for {symbol}: {e}")
                # 尝试使用演示数据作为后备
                if symbol in self.demo_data:
                    return self.demo_data[symbol].copy()
                return None

        except Exception as e:
            self.logger.error(f"Error fetching minute data for {symbol}: {e}")
            return None

    def _analyze_data(self, data: pd.DataFrame, params: Dict) -> Dict:
        """分析数据"""
        try:
            # 基本技术指标计算
            if len(data) < 2:
                return {'matched': False, 'reason': 'Insufficient data'}

            # 计算基本指标
            data['returns'] = data['close'].pct_change()
            data['volatility'] = data['returns'].rolling(window=20).std()
            data['ma_20'] = data['close'].rolling(window=20).mean()
            data['ma_50'] = data['close'].rolling(window=50).mean()

            # 获取最新数据
            latest = data.iloc[-1]

            # 分析条件
            conditions = {
                'price_above_ma': latest['close'] > latest['ma_20'],
                'volume_spike': latest['volume'] > data['volume'].mean() * 1.5,
                'volatility_normal': latest['volatility'] < data['volatility'].mean() * 2
            }

            # 汇总分析结果
            matched = all(conditions.values())

            return {
                'matched': matched,
                'conditions': conditions,
                'price': latest['close'],
                'volume': latest['volume'],
                'volatility': latest['volatility'],
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"Error analyzing data: {e}")
            return {'matched': False, 'reason': str(e)}

    async def get_symbols(self, market: str) -> List[str]:
        """获取市场中的所有交易对"""
        try:
            # 首先检查配置中是否有预定义的symbols
            config_symbols = self.SCANNER_CONFIG.get('symbols', [])
            if config_symbols:
                self.logger.info("Using symbols from config")
                return config_symbols

            # 尝试使用API获取
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                            f"{self.api_endpoints['symbols']}",
                            params={'market': market}
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            symbols = data.get('symbols', [])
                            if symbols:
                                return symbols
            except Exception as e:
                self.logger.warning(f"Failed to get symbols from API: {e}")

            # 如果API获取失败，返回默认股票池
            default_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
            self.logger.info("Using default symbol list")
            return default_symbols

        except Exception as e:
            self.logger.error(f"Error getting symbols: {e}")
            return []

    async def get_market_data(self, market: str) -> Dict:
        """获取市场数据"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                        f"{self.api_endpoints['market_data']}",
                        params={'market': market}
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        self.logger.warning(f"Failed to get market data, status: {response.status}")
                        return {}
        except aiohttp.ClientError as e:
            self.logger.error(f"Network error getting market data: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return {}

    def get_active_symbols(self) -> Set[str]:
        """获取当前活跃的交易标的"""
        return self.active_symbols

    async def cleanup(self):
        """清理资源"""
        try:
            await self.stop()
            self.data_cache.clear()
            self.last_update.clear()
            self.active_symbols.clear()
            self.initialized = False
            self.logger.info("Market scanner cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during market scanner cleanup: {e}")
            raise