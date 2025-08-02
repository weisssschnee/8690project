import os
import sys
import pytest

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 回退到项目根目录
sys.path.append(project_root)

from app.main import TradingApp, main  # 使用完整的导入路径

def test_trading_app_init():
    """测试TradingApp初始化"""
    app = TradingApp()
    assert isinstance(app, TradingApp)

# 其他测试用例...

if __name__ == "__main__":
    pytest.main(["-v"])