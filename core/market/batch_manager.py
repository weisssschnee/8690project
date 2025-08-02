# core/market/batch_manager.py
from typing import List, Callable
import asyncio
import logging


class BatchManager:
    """批处理管理器 - 控制数据获取的速率和并发"""

    def __init__(self, batch_size: int = 10, delay: float = 1.0):
        self.batch_size = batch_size
        self.delay = delay
        self.logger = logging.getLogger(__name__)

    async def process_batches(
            self,
            items: List[str],
            process_func: Callable,
            max_retries: int = 3
    ):
        """批量处理项目"""
        results = {}
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            try:
                batch_results = await self._process_batch(batch, process_func, max_retries)
                results.update(batch_results)
                await asyncio.sleep(self.delay)  # 控制请求速率
            except Exception as e:
                self.logger.error(f"处理批次失败: {str(e)}")

        return results

    async def _process_batch(
            self,
            batch: List[str],
            process_func: Callable,
            max_retries: int
    ) -> dict:
        """处理单个批次"""
        results = {}
        tasks = []

        for item in batch:
            task = self._process_with_retry(item, process_func, max_retries)
            tasks.append(task)

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for item, result in zip(batch, batch_results):
            if not isinstance(result, Exception):
                results[item] = result

        return results

    async def _process_with_retry(
            self,
            item: str,
            process_func: Callable,
            max_retries: int
    ):
        """带重试的处理单个项目"""
        for attempt in range(max_retries):
            try:
                return await process_func(item)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1 * (attempt + 1))  # 指数退避