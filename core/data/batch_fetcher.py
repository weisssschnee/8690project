# core/data/batch_fetcher.py
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any
import pandas as pd
from core.data.manager import DataManager

logger = logging.getLogger(__name__)


class BatchFetcher:
    """
    一个专门用于批量、并发获取市场数据的辅助类。
    它作为 DataManager 的一个性能增强层，本身不包含API key或连接逻辑。
    """

    def __init__(self, data_manager: DataManager, max_workers: int = 10):
        """
        初始化批量获取器。

        Args:
            data_manager (DataManager): 一个已初始化的 DataManager 实例。
            max_workers (int): 用于并发请求的线程池的最大线程数。
        """
        if not isinstance(data_manager, DataManager):
            raise TypeError("data_manager must be an instance of DataManager")
        self.data_manager = data_manager
        self.max_workers = max_workers
        logger.info(f"BatchFetcher initialized with max_workers={self.max_workers}")

    def fetch_batch_historical_data(
            self,
            symbols: List[str],
            days: int,
            interval: str = "1d"
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        并发地为多个股票获取历史K线数据。

        Args:
            symbols (List[str]): 股票代码列表。
            days (int): 获取数据的天数。
            interval (str): K线周期 (e.g., '1d', '1h', '5m')。

        Returns:
            Dict[str, Optional[pd.DataFrame]]: 一个字典，键为股票代码，值为对应的DataFrame数据；
                                               如果某个股票获取失败，则值为None。
        """
        if not symbols:
            return {}

        logger.info(f"Starting batch historical data fetch for {len(symbols)} symbols...")
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 创建未来任务字典 {future: symbol}
            future_to_symbol = {
                executor.submit(self.data_manager.get_historical_data, symbol, days, interval): symbol
                for symbol in symbols
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        results[symbol] = data
                        logger.debug(f"Successfully fetched historical data for {symbol}.")
                    else:
                        results[symbol] = None
                        logger.warning(f"Failed to fetch historical data for {symbol} (returned empty).")
                except Exception as exc:
                    results[symbol] = None
                    logger.error(f"An exception occurred while fetching historical data for {symbol}: {exc}",
                                 exc_info=True)

        logger.info(
            f"Batch historical data fetch completed. Successfully retrieved data for {len([v for v in results.values() if v is not None])}/{len(symbols)} symbols.")
        return results

    def fetch_batch_news(
            self,
            symbols: List[str],
            num_articles: int = 20
    ) -> Dict[str, Optional[List[Dict]]]:
        """
        并发地为多个股票获取最新新闻。

        Args:
            symbols (List[str]): 股票代码列表。
            num_articles (int): 每个股票获取的新闻数量。

        Returns:
            Dict[str, Optional[List[Dict]]]: 一个字典，键为股票代码，值为新闻列表；获取失败则为None。
        """
        if not symbols:
            return {}

        logger.info(f"Starting batch news fetch for {len(symbols)} symbols...")
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.data_manager.get_news, symbol, num_articles): symbol
                for symbol in symbols
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    news_list = future.result()
                    if news_list:  # 检查列表是否为非空
                        results[symbol] = news_list
                        logger.debug(f"Successfully fetched {len(news_list)} news articles for {symbol}.")
                    else:
                        results[symbol] = None
                        logger.warning(f"Failed to fetch news for {symbol} (returned empty).")
                except Exception as exc:
                    results[symbol] = None
                    logger.error(f"An exception occurred while fetching news for {symbol}: {exc}", exc_info=True)

        logger.info(
            f"Batch news fetch completed. Successfully retrieved news for {len([v for v in results.values() if v is not None])}/{len(symbols)} symbols.")
        return results