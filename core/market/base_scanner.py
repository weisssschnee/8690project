# core/market/base_scanner.py

import logging
from datetime import datetime, timedelta
import asyncio
import aiohttp
from typing import Dict, List, Optional, Set
import pandas as pd


class BaseMarketScanner:
    def __init__(self, config=None):
        """
        初始化基础市场扫描器
        Args:
            config: 配置字典，包含扫描器配置
        """
        # 确保config是字典类型
        if config is None:
            config = {}

        # 确保config中有SCANNER_CONFIG
        if 'SCANNER_CONFIG' not in config:
            config['SCANNER_CONFIG'] = {
                'delay': 60,
                'markets': ['US'],
                'symbols': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
            }

        # 确保config中有API_CONFIG
        if 'API_CONFIG' not in config:
            config['API_CONFIG'] = {
                'endpoints': {
                    'market_data': 'your_market_data_endpoint',
                    'minute_data': 'your_minute_data_endpoint',
                    'symbols': 'your_symbols_endpoint'
                }
            }

        self.config = config
        self.logger = logging.getLogger(__name__)
        self.active_symbols: Set[str] = set()
        self.is_running = False
        self.initialized = False
        self.cache = {}

    async def initialize(self):
        """初始化扫描器"""
        try:
            # 获取初始股票池
            self.active_symbols = await self.get_symbols()
            if not self.active_symbols:
                self.active_symbols = set(self.config['SCANNER_CONFIG']['symbols'])

            self.initialized = True
            self.logger.info("Market scanner initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize market scanner: {e}")
            raise

    async def get_minute_data(self, symbol: str, period: int) -> Optional[pd.DataFrame]:
        """获取分钟级数据"""
        try:
            # 基础实现，应在子类中重写
            return None
        except Exception as e:
            self.logger.error(f"Error fetching minute data for {symbol}: {e}")
            return None

    async def get_market_data(self, market: str) -> Dict:
        """获取市场数据"""
        try:
            # 基础实现，应在子类中重写
            return {}
        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            return {}

    async def scan_market(self, market: str, scan_params: Dict) -> List[Dict]:
        """扫描市场"""
        try:
            # 基础实现，应在子类中重写
            return []
        except Exception as e:
            self.logger.error(f"Error scanning market: {e}")
            return []

    async def get_symbols(self) -> set:
        """获取交易标的列表"""
        try:
            try:
                # 尝试从API获取
                symbols = await self._fetch_symbols_from_api()
                if symbols:
                    return symbols
                self.logger.warning("Failed to get symbols from API: " +
                                    self.config['API_CONFIG']['endpoints']['symbols'])
                self.logger.info("Using default symbol list")
                return set(self.config['SCANNER_CONFIG']['symbols'])
            except Exception as e:
                self.logger.warning(f"Failed to fetch symbols from API: {e}")
                return set(self.config['SCANNER_CONFIG']['symbols'])
        except Exception as e:
            self.logger.error(f"Error getting symbols: {e}")
            return set()

    async def _fetch_symbols_from_api(self) -> Set[str]:
        """从API获取股票列表"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.config['API_CONFIG']['endpoints']['symbols']) as response:
                    if response.status == 200:
                        data = await response.json()
                        return set(data)
            return set()
        except Exception as e:
            self.logger.warning(f"Failed to fetch symbols from API: {e}")
            return set()

    def clear_cache(self):
        """清除缓存数据"""
        self.cache.clear()

    async def update_cache(self, symbol: str, data: pd.DataFrame):
        """更新缓存数据"""
        self.cache[symbol] = {
            'data': data,
            'timestamp': datetime.now()
        }

    def get_cached_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取缓存的数据"""
        if symbol in self.cache:
            cache_age = datetime.now() - self.cache[symbol]['timestamp']
            if cache_age.total_seconds() < 60:  # 1分钟缓存
                return self.cache[symbol]['data']
        return None

    async def get_historical_data(self, symbol: str, start_time: datetime,
                                  end_time: datetime) -> Optional[pd.DataFrame]:
        """获取历史数据"""
        try:
            # 基础实现，应在子类中重写
            return None
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None

    async def get_ticker_info(self, symbol: str) -> Dict:
        """获取交易对信息"""
        try:
            # 基础实现，应在子类中重写
            return {}
        except Exception as e:
            self.logger.error(f"Error fetching ticker info for {symbol}: {e}")
            return {}

    async def get_market_status(self, market: str) -> Dict:
        """获取市场状态"""
        try:
            # 基础实现，应在子类中重写
            return {
                'market': market,
                'status': 'unknown',
                'timestamp': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"Error fetching market status: {e}")
            return {}

    @property
    def SCANNER_CONFIG(self):
        """确保SCANNER_CONFIG始终可访问"""
        return self.config['SCANNER_CONFIG']