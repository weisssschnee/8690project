# core/utils/data_mock.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def create_demo_data():
    """创建示例股票数据"""
    try:
        demo_data = {}
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA',
                   '600519.SH', '000001.SZ', '600036.SH', '601318.SH']

        for symbol in symbols:
            # 设置基本参数
            if symbol.endswith(('.SH', '.SZ')):
                # A股价格范围
                base_price = {
                    '600519.SH': 1800, '000001.SZ': 15,
                    '600036.SH': 35, '601318.SH': 40
                }.get(symbol, 50)
                vol_base = 10000000  # A股成交量通常较大
            else:
                # 美股价格范围
                base_price = {
                    'AAPL': 150, 'GOOGL': 150, 'MSFT': 300,
                    'AMZN': 3300, 'META': 300, 'TSLA': 800, 'NVDA': 100
                }.get(symbol, 100)
                vol_base = 1000000

            # 生成日期序列
            dates = pd.date_range(end=datetime.now(), periods=100, freq='D')

            # 生成价格序列
            np.random.seed(hash(symbol) % 10000)  # 不同股票有不同种子
            prices = np.random.normal(base_price, base_price * 0.01, 100)
            # 添加一些趋势
            trend = np.linspace(-0.05, 0.05, 100) * base_price
            prices = prices + trend

            # 生成交易量
            volumes = np.random.normal(vol_base, vol_base * 0.2, 100)
            # 有时成交量和价格相关
            if np.random.random() > 0.5:
                volume_trend = np.array([0.5 if p > prices[i - 1] else -0.2 for i, p in enumerate(prices)])
                volume_trend[0] = 0
                volumes = volumes * (1 + volume_trend)

            # 创建OHLC数据
            high = prices + np.random.uniform(0, base_price * 0.01, 100)
            low = prices - np.random.uniform(0, base_price * 0.01, 100)

            # 确保最高价大于收盘价和开盘价，最低价小于收盘价和开盘价
            opens = prices * (1 + np.random.uniform(-0.005, 0.005, 100))
            for i in range(len(prices)):
                high[i] = max(high[i], prices[i], opens[i])
                low[i] = min(low[i], prices[i], opens[i])

            # 创建DataFrame
            df = pd.DataFrame({
                'open': opens,
                'high': high,
                'low': low,
                'close': prices,
                'volume': np.abs(volumes)
            }, index=dates)

            demo_data[symbol] = df

        logger.info(f"创建了{len(demo_data)}个示例股票数据")
        return demo_data

    except Exception as e:
        logger.error(f"生成示例数据失败: {e}")
        return {}  # 返回空字典

