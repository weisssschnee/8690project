# test_futu_integration.py
import logging
import sys
from pathlib import Path
import time
import pandas as pd

# --- 设置 Python 路径以导入 core 模块 ---
# 假设此脚本在项目根目录，core 是其子目录
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# --- 结束路径设置 ---

# --- 导入必要的类 ---
from core.config import Config
from core.data.manager import DataManager
from core.trading.executor import OrderExecutor
import futu as ft  # 导入 futu 用于常量比较

# --- 配置一个简单的日志记录器，以便在控制台看到输出 ---
logging.basicConfig(
    level=logging.INFO,  # 设置为 INFO 或 DEBUG 以查看详细信息
    format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s'
)
logger = logging.getLogger("FUTU_TEST")


def test_futu_integration():
    """执行一系列测试来验证 Futu API 的集成"""
    logger.info("=============================================")
    logger.info("===      FUTU API 集成测试开始           ===")
    logger.info("=============================================")

    # 1. 初始化 Config, DataManager, OrderExecutor
    logger.info("\n[步骤 1] 初始化系统组件...")
    try:
        config = Config()
        # DataManager 和 OrderExecutor 在初始化时会自动尝试连接 Futu
        # 我们需要给它们一点时间来完成后台连接
        data_manager = DataManager(config)
        order_executor = OrderExecutor(config, risk_manager=None)  # 测试时暂时不关心风控

        logger.info("组件已创建，等待后台连接 (5秒)...")
        time.sleep(5)  # 等待后台线程完成连接尝试
    except Exception as e:
        logger.error(f"组件初始化失败: {e}", exc_info=True)
        return

    # 2. 检查 Futu 连接状态
    logger.info("\n[步骤 2] 检查 Futu 连接状态...")
    dm_futu_connected = data_manager.futu_is_connected if hasattr(data_manager, 'futu_is_connected') else False
    oe_futu_unlocked = order_executor.futu_trade_is_unlocked if hasattr(order_executor,
                                                                        'futu_trade_is_unlocked') else False

    if dm_futu_connected:
        logger.info("✅ [成功] DataManager 的 Futu 行情上下文已连接！")
    else:
        logger.error("❌ [失败] DataManager 的 Futu 行情上下文未连接。请确保 FutuOpenD 正在运行。")

    if oe_futu_unlocked:
        logger.info("✅ [成功] OrderExecutor 的 Futu 交易上下文已解锁！")
    else:
        logger.warning("⚠️ [警告] OrderExecutor 的 Futu 交易上下文未解锁。请检查 FutuOpenD 中的交易密码和登录状态。")

    if not dm_futu_connected:
        logger.critical("行情上下文连接失败，后续测试可能无法进行。")
        # 即使连接失败，我们也可以继续测试，以验证回退逻辑是否正常工作

    # 3. 测试获取历史数据 (get_historical_data)
    logger.info("\n[步骤 3] 测试获取历史 K 线数据 (US.AAPL)...")
    try:
        aapl_hist = data_manager.get_historical_data("US.AAPL", days=10)
        if aapl_hist is not None and not aapl_hist.empty:
            logger.info(f"✅ [成功] 获取到 US.AAPL 的历史数据 ({len(aapl_hist)} 行)。")
            logger.info("数据预览 (最后3行):")
            print(aapl_hist.tail(3))
        else:
            logger.error("❌ [失败] 未能获取到 US.AAPL 的历史数据。")
    except Exception as e:
        logger.error(f"获取历史数据时发生异常: {e}", exc_info=True)

    # 4. 测试获取实时价格 (get_realtime_price)
    logger.info("\n[步骤 4] 测试获取实时价格 (HK.00700)...")
    # 首先订阅
    if dm_futu_connected:
        logger.info("订阅 HK.00700 的实时报价...")
        data_manager.futu_subscribe(['HK.00700'])
        logger.info("等待几秒钟以接收推送数据...")
        time.sleep(3)

    try:
        tencent_price_data = data_manager.get_realtime_price("HK.00700")
        if tencent_price_data and tencent_price_data.get('price'):
            logger.info(f"✅ [成功] 获取到 HK.00700 的实时价格: {tencent_price_data}")
        else:
            logger.error(f"❌ [失败] 未能获取到 HK.00700 的实时价格。")
    except Exception as e:
        logger.error(f"获取实时价格时发生异常: {e}", exc_info=True)

    # 5. 测试股票搜索 (search_stocks)
    logger.info("\n[步骤 5] 测试股票搜索 (搜索 '阿里巴巴')...")
    try:
        search_results = data_manager.search_stocks("阿里巴巴", limit=5)
        if search_results:
            logger.info(f"✅ [成功] 搜索到 {len(search_results)} 个相关结果。")
            df_search = pd.DataFrame(search_results)
            print(df_search)
        else:
            logger.error("❌ [失败] 未能搜索到 '阿里巴巴' 的相关股票。")
    except Exception as e:
        logger.error(f"搜索股票时发生异常: {e}", exc_info=True)

    # 6. 测试交易下单 (place_order) - 在模拟环境中
    logger.info("\n[步骤 6] 测试模拟交易下单 (买入 100 股 US.BABA)...")
    if oe_futu_unlocked:
        order_details = {
            "symbol": "US.BABA",
            "quantity": 100,
            "price": 80.0,  # 限价单价格
            "order_type": "Limit Order",  # 使用内部标识符
            "direction": "Buy"  # 使用内部标识符
        }
        try:
            trade_result = order_executor.execute_order(**order_details)
            if trade_result and trade_result.get('success'):
                logger.info(f"✅ [成功] Futu 模拟下单成功！结果: {trade_result}")
            else:
                logger.error(f"❌ [失败] Futu 模拟下单失败。结果: {trade_result}")
        except Exception as e:
            logger.error(f"执行交易时发生异常: {e}", exc_info=True)
    else:
        logger.warning("⚠️ [跳过] 因交易未解锁，跳过交易下单测试。")

    logger.info("\n=============================================")
    logger.info("===      FUTU API 集成测试结束           ===")
    logger.info("=============================================")

