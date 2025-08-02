# 3. 创建 data_priority_manager.py
import time

class DataPriorityManager:
    """根据使用频率和重要性管理数据更新"""

    def __init__(self):
        self.update_tiers = {
            'realtime': {  # 1-5秒更新
                'symbols': [],  # 用户正在查看的股票
                'interval': 2
            },
            'frequent': {  # 30秒更新
                'symbols': [],  # 自选股
                'interval': 30
            },
            'normal': {  # 5分钟更新
                'symbols': [],  # 历史查看过的
                'interval': 300
            }
        }

    def get_update_priority(self, symbol: str) -> str:
        """获取股票的更新优先级"""
        for tier, config in self.update_tiers.items():
            if symbol in config['symbols']:
                return tier
        return 'normal'

    def should_update(self, symbol: str, last_update: float) -> bool:
        """判断是否需要更新"""
        tier = self.get_update_priority(symbol)
        interval = self.update_tiers[tier]['interval']
        return time.time() - last_update > interval