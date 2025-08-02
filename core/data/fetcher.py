# core/data/fetcher.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import tushare as ts
import logging
from typing import Optional, Dict, List, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiohttp
import time
import pytz
from finnhub import Client as FinnhubClient
from polygon import RESTClient


# core/data/fetcher.py

class DataFetcher:
    def __init__(self, config):
        """初始化数据获取器"""
        self.config = config or {}  # 确保 config 是字典
        self.logger = logging.getLogger(__name__)

        # 使用 dict.get() 方法获取 MAX_WORKERS
        max_workers = self.config.get('MAX_WORKERS', 4)  # 默认值为 4
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        self.data_cache = {}
        self.price_cache = {}
        self.last_update_time = {}

        # 设置默认值
        self.ts_pro = None
        self.news_api_key = None
        self.finnhub_client = None
        self.polygon_client = None
        self.scanner = None
        self.session = None

        # 初始化APIs
        self._init_apis()

    def _init_apis(self):
        """初始化各数据源API"""
        try:
            # 初始化Tushare
            tushare_token = self.config.get('TUSHARE_TOKEN')
            if tushare_token:
                self.ts_pro = ts.pro_api(tushare_token)
                self.logger.info("Tushare API initialized successfully")
            else:
                self.logger.warning("Tushare token not found")

            # 初始化NewsAPI
            news_api_key = self.config.get('NEWS_API_KEY')
            if news_api_key:
                self.news_api_key = news_api_key
                self.logger.info("News API key found")
            else:
                self.logger.warning("News API key not found")

            # 初始化Finnhub
            finnhub_key = self.config.get('FINNHUB_KEY')
            if finnhub_key:
                self.finnhub_client = FinnhubClient(api_key=finnhub_key)
                self.logger.info("Finnhub API initialized successfully")
            else:
                self.logger.warning("Finnhub key not found")

            # 初始化Polygon
            polygon_key = self.config.get('POLYGON_KEY')
            if polygon_key:
                self.polygon_client = RESTClient(polygon_key)
                self.logger.info("Polygon API initialized successfully")
            else:
                self.logger.warning("Polygon key not found")

        except Exception as e:
            self.logger.error(f"API initialization error: {e}")

    def set_scanner(self, scanner):
        """设置市场扫描器"""
        try:
            self._scanner_instance = scanner  # 保存scanner实例
            self.scanner = scanner  # 直接设置scanner实例
            self.scanner_initialized = True  # 直接标记为已初始化
            self.logger.info("Scanner instance saved and initialized")
        except Exception as e:
            self.logger.error(f"Error setting scanner: {e}")
            self.scanner = None
            self.scanner_initialized = False

    async def _init_scanner(self):
        """异步初始化扫描器（保留但简化）"""
        if not self.scanner_initialized and self._scanner_instance:
            try:
                self.scanner = self._scanner_instance
                self.scanner_initialized = True
                self.logger.info("Scanner initialized successfully")
            except Exception as e:
                self.logger.error(f"Scanner initialization error: {e}")

    async def get_historical_data(
            self,
            symbol: str,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            interval: str = '1d'
    ) -> Optional[pd.DataFrame]:
        """带验证的数据获取方法"""
        try:
            # 验证symbol格式
            if not self._validate_symbol(symbol):
                self.logger.warning(f"无效的股票代码格式: {symbol}")
                return None

            # 缓存检查
            cache_key = self._get_cache_key(symbol, start_date, end_date, interval)
            if cache_key in self.data_cache:
                return self.data_cache[cache_key]

            # 实际获取数据
            if self._is_cn_stock(symbol):
                data = await self._get_cn_stock_data(symbol, start_date, end_date)
            else:
                data = await self._get_us_stock_data(symbol, start_date, end_date, interval)

            if data is not None:
                data = await self._add_technical_indicators(data)
                self.data_cache[cache_key] = data
                return data

        except Exception as e:
            self.logger.error(f"获取数据失败: {symbol}, 错误: {str(e)}")
            return None

    def _validate_symbol(self, symbol: str) -> bool:
        """验证股票代码格式"""
        if not symbol:
            return False
        if self._is_cn_stock(symbol):
            return symbol.endswith(('.SH', '.SZ'))
        return True  # 美股不检查后缀

    async def _fetch_with_retry(self, session, url, params=None, headers=None, max_retries=3):
        """带重试的异步请求"""
        for attempt in range(max_retries):
            try:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()

            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:  # 最后一次尝试
                    raise  # 重试耗尽，抛出异常

                await asyncio.sleep(1 * (attempt + 1))  # 指数退避

        return None  # 所有重试都失败

    def _get_cache_key(self, symbol: str, start_date: Optional[str] = None,
                       end_date: Optional[str] = None, interval: str = '1d') -> str:
        """生成缓存键"""
        return f"{symbol}_{start_date}_{end_date}_{interval}"

    def _should_update_cache(self, symbol: str) -> bool:
        """检查是否需要更新缓存"""
        if symbol not in self.last_update_time:
            return True

        cache_age = time.time() - self.last_update_time[symbol]
        return cache_age >= self.config.PRICE_CACHE_DURATION

    def get_realtime_price(self, symbol: str) -> Optional[Dict]:
        # 检查缓存有效性（时间戳 + 市场状态）
        if symbol in self.price_cache:
            cache_age = time.time() - self.last_update_time[symbol]
            is_market_open = self._is_market_open('US' if not self._is_cn_stock(symbol) else 'CN')
            # 休市时延长缓存时间
            if (is_market_open and cache_age < 5) or (not is_market_open and cache_age < 300):
                return self.price_cache[symbol]
        """获取实时价格数据"""
        try:
            # 检查缓存
            if symbol in self.price_cache and not self._should_update_cache(symbol):
                return self.price_cache[symbol]

            price_data = None
            market = 'CN' if self._is_cn_stock(symbol) else 'US'
            is_market_open = self._is_market_open(market)

            if market == 'CN':
                price_data = self._get_cn_realtime_price(symbol, is_market_open)
            else:
                price_data = self._get_us_realtime_price(symbol, is_market_open)

            if price_data:
                # 更新缓存
                self.price_cache[symbol] = price_data
                self.last_update_time[symbol] = time.time()
                return price_data

            return None

        except Exception as e:
            self.logger.info(f"正在获取 {symbol} 实时数据，当前缓存状态: {self.price_cache.get(symbol)}")
            return None

    def _get_cn_realtime_price(self, symbol: str, is_market_open: bool) -> Optional[Dict]:
        """获取A股实时价格"""
        try:
            if not self.ts_pro:
                return None

            df = self.ts_pro.daily(ts_code=symbol, start_date=datetime.now().strftime('%Y%m%d'))
            if not df.empty:
                return {
                    'price': float(df.iloc[0]['close']),
                    'change': float(df.iloc[0]['change']),
                    'volume': float(df.iloc[0]['vol']),
                    'timestamp': time.time(),
                    'delayed': not is_market_open,
                    'source': 'tushare'
                }
        except Exception as e:
            self.logger.warning(f"Error fetching CN stock price: {str(e)}")
        return None

    def _get_us_realtime_price(self, symbol: str, is_market_open: bool) -> Optional[Dict]:
        # 优先使用YFinance（免费稳定）
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'price': info.get('regularMarketPrice', 0),
                'change': info.get('regularMarketChangePercent', 0),
                'volume': info.get('regularMarketVolume', 0),
                'timestamp': time.time(),
                'delayed': not is_market_open,
                'source': 'yfinance'
            }
        except Exception as e:
            self.logger.error(f"YFinance获取失败: {e}")
            return None

    def _get_finnhub_price(self, symbol: str, is_market_open: bool) -> Optional[Dict]:
        """从Finnhub获取价格"""
        try:
            quote = self.finnhub_client.quote(symbol)
            if quote and quote['c'] > 0:
                return {
                    'price': quote['c'],
                    'change': ((quote['c'] - quote['pc']) / quote['pc']) * 100,
                    'volume': quote.get('v', 0),
                    'timestamp': time.time(),
                    'delayed': not is_market_open,
                    'source': 'finnhub'
                }
        except Exception as e:
            self.logger.warning(f"Finnhub API error: {str(e)}")
        return None

    def _get_polygon_price(self, symbol: str, is_market_open: bool) -> Optional[Dict]:
        """从Polygon获取价格"""
        try:
            last_trade = self.polygon_client.get_last_trade(symbol)
            if last_trade:
                return {
                    'price': last_trade.price,
                    'volume': last_trade.size,
                    'timestamp': last_trade.timestamp / 1000,
                    'delayed': not is_market_open,
                    'source': 'polygon'
                }
        except Exception as e:
            self.logger.warning(f"Polygon API error: {str(e)}")
        return None


    def _is_market_open(self, market: str = 'US') -> bool:
        """检查市场是否开放"""
        now = datetime.now(pytz.timezone('America/New_York' if market == 'US' else 'Asia/Shanghai'))
        current_time = now.time()

        # 检查是否是工作日
        if now.weekday() >= 5:
            return False

        # 获取市场时间配置
        market_hours = self.config.MARKET_HOURS.get(market, {})

        if market == 'CN':
            morning_open = market_hours.get('morning_open')
            morning_close = market_hours.get('morning_close')
            afternoon_open = market_hours.get('afternoon_open')
            afternoon_close = market_hours.get('afternoon_close')

            return ((morning_open <= current_time <= morning_close) or
                    (afternoon_open <= current_time <= afternoon_close))
        else:
            market_open = market_hours.get('market_open')
            market_close = market_hours.get('market_close')

            return market_open <= current_time <= market_close

    def _is_cn_stock(self, symbol: str) -> bool:
        """判断是否为中国股票"""
        return symbol.endswith(('.SH', '.SZ'))

    async def _get_cn_stock_data(
            self,
            symbol: str,
            start_date: Optional[str],
            end_date: Optional[str]
    ) -> Optional[pd.DataFrame]:
        """异步获取中国股票数据"""
        if not self.ts_pro:
            self.logger.error("Tushare API not initialized")
            return None

        try:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                self.executor,
                lambda: self.ts_pro.daily(
                    ts_code=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
            )
            return await self._format_cn_data(df)
        except Exception as e:
            self.logger.error(f"Error fetching CN stock data: {e}")
            return None

    async def _get_us_stock_data(
            self,
            symbol: str,
            start_date: Optional[str],
            end_date: Optional[str],
            interval: str
    ) -> Optional[pd.DataFrame]:
        """异步获取美股数据"""
        try:
            loop = asyncio.get_event_loop()
            ticker = yf.Ticker(symbol)
            df = await loop.run_in_executor(
                self.executor,
                lambda: ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval
                )
            )
            return await self._format_us_data(df)
        except Exception as e:
            self.logger.error(f"Error fetching US stock data: {e}")
            return None

    async def _fetch_news(self, symbol: str) -> List[Dict]:
        """获取新闻数据"""
        try:
            # 首先尝试使用 NewsAPI
            try:
                articles = await self._fetch_from_newsapi(symbol)
                if articles:
                    return articles
            except Exception as e:
                self.logger.warning(f"NewsAPI failed: {e}")

            # 如果 NewsAPI 失败，使用备用 RSS 源
            try:
                articles = await self._fetch_from_rss(symbol)
                return articles
            except Exception as e:
                self.logger.warning(f"RSS feeds failed: {e}")

            return []

        except Exception as e:
            self.logger.error(f"Error fetching news: {e}")
            return []

    async def _fetch_from_newsapi(self, symbol: str) -> List[Dict]:
        """从 NewsAPI 获取新闻"""
        try:
            if not self.news_api_key:
                raise ValueError("News API key not found")

            url = "https://newsapi.org/v2/everything"
            params = {
                'q': symbol,
                'apiKey': self.news_api_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 10
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('articles', [])
                    elif response.status == 401:
                        self.logger.error("News API authentication failed. Switching to fallback source.")
                        return await self._fetch_from_rss(symbol)
                    else:
                        self.logger.error(f"News API request failed: {response.status}")
                        return await self._fetch_from_rss(symbol)

        except Exception as e:
            self.logger.error(f"Error in NewsAPI request: {e}")
            return await self._fetch_from_rss(symbol)

    async def _fetch_from_rss(self, symbol: str) -> List[Dict]:
        """从 RSS feeds 获取新闻"""
        try:
            import feedparser
            from datetime import datetime
            import pytz

            articles = []
            feeds = self.config.NEWS_SOURCES['fallback']['feeds']

            for feed_url in feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:5]:  # 每个源取前5条
                        # 检查新闻是否包含目标股票代码
                        if symbol.lower() in entry.title.lower() or \
                                (hasattr(entry, 'description') and
                                 symbol.lower() in entry.description.lower()):

                            # 转换发布时间为ISO格式
                            try:
                                published = datetime.strptime(
                                    entry.published,
                                    '%a, %d %b %Y %H:%M:%S %z'
                                ).isoformat()
                            except:
                                published = datetime.now(pytz.UTC).isoformat()

                            articles.append({
                                'title': entry.title,
                                'description': entry.description if hasattr(entry, 'description') else '',
                                'url': entry.link if hasattr(entry, 'link') else '',
                                'publishedAt': published,
                                'source': {
                                    'name': feed.feed.title if hasattr(feed, 'feed') else 'RSS Feed'
                                }
                            })

                except Exception as e:
                    self.logger.warning(f"Error parsing RSS feed {feed_url}: {e}")
                    continue

            return articles[:10]  # 最多返回10条新闻

        except Exception as e:
            self.logger.error(f"Error fetching RSS news: {e}")
            return []

    async def _format_cn_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """格式化中国股票数据"""
        df = df.rename(columns={
            'trade_date': 'date',
            'vol': 'volume'
        })
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date').sort_index()

    async def _format_us_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """格式化美股数据"""
        return df.rename(columns={
            'Volume': 'volume',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close'
        })

    async def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        try:
            # 移动平均线
            for period in self.config.MA_PERIODS:
                df[f'ma{period}'] = df['close'].rolling(window=period).mean()

            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.config.RSI_PERIOD).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.RSI_PERIOD).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = df['close'].ewm(span=self.config.MACD_PARAMS['fast']).mean()
            exp2 = df['close'].ewm(span=self.config.MACD_PARAMS['slow']).mean()
            df['macd'] = exp1 - exp2
            df['signal'] = df['macd'].ewm(span=self.config.MACD_PARAMS['signal']).mean()
            df['hist'] = df['macd'] - df['signal']

            # Bollinger Bands
            df['bollinger_middle'] = df['close'].rolling(
                window=self.config.BOLLINGER_PARAMS['period']
            ).mean()
            std = df['close'].rolling(
                window=self.config.BOLLINGER_PARAMS['period']
            ).std()
            df['bollinger_upper'] = df['bollinger_middle'] + (
                    std * self.config.BOLLINGER_PARAMS['std_dev']
            )
            df['bollinger_lower'] = df['bollinger_middle'] - (
                    std * self.config.BOLLINGER_PARAMS['std_dev']
            )

            return df

        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return df

    async def cleanup(self):
        """清理资源"""
        try:
            if self.session:
                await self.session.close()
            self.logger.info("DataFetcher resources cleaned up")
        except Exception as e:
            self.logger.error(f"Error cleaning up DataFetcher: {e}")
            raise

    def clear_cache(self):
        """清除数据缓存"""
        self.data_cache.clear()
        self.price_cache.clear()
        self.last_update_time.clear()
        self.logger.info("Data cache cleared")