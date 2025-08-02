# core/data/direct_fetcher.py
import os
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DirectDataFetcher:
    """直接调用API的数据获取器，绕过复杂的组件架构"""

    def __init__(self):
        """初始化数据获取器"""
        self.api_preference = ["yfinance", "alpha_vantage", "polygon", "finnhub", "tushare", "akshare"]
        self._check_apis()

    def _check_apis(self):
        """检查哪些API可用"""
        self.available_apis = []

        # 检查yfinance
        try:
            import akshare as ak
            self.available_apis.append("akshare")
        except ImportError:
            pass

        # 检查yfinance
        try:
            import yfinance
            self.available_apis.append("yfinance")
        except ImportError:
            pass

        # 检查polygon
        if os.environ.get("POLYGON_KEY"):
            try:
                from polygon import RESTClient
                self.available_apis.append("polygon")
            except ImportError:
                pass

        # 检查tushare
        if os.environ.get("TUSHARE_TOKEN"):
            try:
                import tushare
                self.available_apis.append("tushare")
            except ImportError:
                pass

        # 检查finnhub
        if os.environ.get("FINNHUB_KEY"):
            try:
                import finnhub
                self.available_apis.append("finnhub")
            except ImportError:
                pass

        # 检查alpha_vantage
        if os.environ.get("ALPHA_VANTAGE_KEY"):
            self.available_apis.append("alpha_vantage")

        logger.info(f"可用的API: {self.available_apis}")

    def get_stock_data(self, symbol, days=100):
        """获取股票数据，尝试所有可用API"""
        for api in self.api_preference:
            if api in self.available_apis:
                try:
                    method = getattr(self, f"_get_from_{api}")
                    data = method(symbol, days)
                    if data is not None and not data.empty:
                        logger.info(f"成功通过{api}获取{symbol}数据")
                        return data
                except Exception as e:
                    logger.warning(f"{api}获取{symbol}数据失败: {e}")

        logger.error(f"所有API都无法获取{symbol}数据")
        return pd.DataFrame()

    def _get_from_yfinance(self, symbol, days):
        """从yfinance获取数据"""
        import yfinance as yf
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        data = yf.download(
            symbol,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )

        if data.empty:
            return None

        # 添加symbol列并确保日期是索引
        data['symbol'] = symbol
        return data

    def _get_from_polygon(self, symbol, days):
        """从Polygon获取数据"""
        from polygon import RESTClient
        import pandas as pd

        api_key = os.environ.get("POLYGON_KEY")
        client = RESTClient(api_key)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # 获取历史数据
        aggs = client.get_aggs(
            ticker=symbol,
            multiplier=1,
            timespan="day",
            from_=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d')
        )

        if not aggs:
            return None

        # 转换为DataFrame
        df = pd.DataFrame([{
            'date': datetime.fromtimestamp(agg.timestamp / 1000),
            'open': agg.open,
            'high': agg.high,
            'low': agg.low,
            'close': agg.close,
            'volume': agg.volume,
            'symbol': symbol
        } for agg in aggs])

        df.set_index('date', inplace=True)
        return df

    def _get_from_tushare(self, symbol, days):
        """从Tushare获取数据"""
        import tushare as ts

        # 根据股票代码判断是否为A股
        is_cn_stock = symbol.endswith('.SZ') or symbol.endswith('.SH')
        if not is_cn_stock:
            return None

        api_key = os.environ.get("TUSHARE_TOKEN")
        ts.set_token(api_key)
        pro = ts.pro_api()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        df = pro.daily(
            ts_code=symbol,
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d')
        )

        if df.empty:
            return None

        # 转换列名以匹配标准格式
        df = df.rename(columns={
            'trade_date': 'date',
            'vol': 'volume'
        })

        # 格式化日期并设为索引
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df['symbol'] = symbol

        return df

    def _get_from_finnhub(self, symbol, days):
        """从Finnhub获取数据"""
        import finnhub
        import pandas as pd

        api_key = os.environ.get("FINNHUB_KEY")
        client = finnhub.Client(api_key=api_key)

        end_date = int(datetime.now().timestamp())
        start_date = int((datetime.now() - timedelta(days=days)).timestamp())

        data = client.stock_candles(
            symbol,
            'D',  # 日线数据
            start_date,
            end_date
        )

        if data['s'] != 'ok':
            return None

        df = pd.DataFrame({
            'date': pd.to_datetime(data['t'], unit='s'),
            'open': data['o'],
            'high': data['h'],
            'low': data['l'],
            'close': data['c'],
            'volume': data['v']
        })

        df.set_index('date', inplace=True)
        df['symbol'] = symbol

        return df

    def _get_from_alpha_vantage(self, symbol, days):
        """从Alpha Vantage获取数据"""
        import requests
        import pandas as pd

        api_key = os.environ.get("ALPHA_VANTAGE_KEY")
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=full'

        r = requests.get(url)
        data = r.json()

        if "Error Message" in data or "Time Series (Daily)" not in data:
            return None

        # 解析数据
        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame(time_series).T

        # 转换列名
        df = df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        })

        # 转换数据类型
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col])
        df['volume'] = pd.to_numeric(df['volume'], downcast='integer')

        # 设置索引
        df.index = pd.to_datetime(df.index)
        df.index.name = 'date'

        # 按日期筛选
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df = df[df.index >= start_date]

        df['symbol'] = symbol

        return df