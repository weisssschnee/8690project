import concurrent.futures
import time
from typing import Dict, Optional, List
import pandas as pd


class OptimizedDataManager:
    def __init__(self, config):
        self.config = config
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        # 简单的内存缓存
        self._cache = {}
        self._cache_ttl = 5  # 5秒缓存

    def get_realtime_price_fast(self, symbol: str) -> Optional[Dict]:
        """优化的实时价格获取"""
        # 检查缓存
        cache_key = f"price_{symbol}"
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached_data

        # 并行尝试多个数据源
        futures = []

        # 为不同市场使用不同数据源优先级
        if symbol.endswith('.SH') or symbol.endswith('.SZ'):
            # A股
            sources = ['akshare', 'tushare']
        else:
            # 美股
            sources = ['finnhub', 'polygon', 'alpha_vantage']

        for source in sources:
            future = self.executor.submit(self._fetch_price_from_source, source, symbol)
            futures.append((future, source))

        # 使用第一个返回的有效结果
        for future, source in futures:
            try:
                result = future.result(timeout=2)  # 2秒超时
                if result and result.get('price'):
                    # 更新缓存
                    self._cache[cache_key] = (result, time.time())
                    result['source'] = source
                    return result
            except:
                continue

        return None