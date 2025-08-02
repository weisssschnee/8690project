# autotrader_service.py
import logging
import sys
import time
from pathlib import Path
from datetime import datetime, time as dt_time
import pandas as pd

# --- 1. 初始化环境 ---
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv(project_root / ".env.txt")

# --- 2. 导入核心模块 ---
from core.system import TradingSystem

try:
    from apscheduler.schedulers.blocking import BlockingScheduler

    APS_AVAILABLE = True
except ImportError:
    APS_AVAILABLE = False
    # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
    print("CRITICAL: APScheduler not installed. Please run 'pip install apscheduler'. AutoTrader cannot start.")
    # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^

# --- 3. 配置日志 ---
logging.basicConfig(
    level="INFO",  # 确保级别是 INFO 或 DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - [AutoTrader Service] - %(message)s',
    handlers=[
        logging.FileHandler("autotrader_service.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
logger.info("==========================================================")
logger.info("=            AutoTrader Service Starting Up            =")
logger.info("==========================================================")


# ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^


# --- 4. 核心业务逻辑函数 ---

def is_trading_time(symbol: str, market_hours: dict) -> bool:
    now = datetime.now()
    if "US" in symbol.upper():
        market = market_hours.get('US', {})
        market_open = market.get('market_open', dt_time(9, 30))
        market_close = market.get('market_close', dt_time(16, 0))
        # 简化检查，假设服务与市场在同一时区
        is_trading = market_open <= now.time() < market_close and now.weekday() < 5
        # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
        logger.debug(
            f"Checking trading time for {symbol} (US Market). Now: {now.time()}. Open: {market_open}, Close: {market_close}. In trading hours: {is_trading}")
        # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^
        return is_trading
    # (可以为 'CN' 等其他市场添加类似逻辑)
    logger.debug(f"Trading time check for {symbol}: Market not supported for time check, assuming closed.")
    return False


def execute_ml_quant_strategy(system: TradingSystem, config: dict):
    strategy_id = config.get("strategy_id")
    symbol = config.get("symbol")
    user_id = config.get("user_id")
    model_name = config.get("ml_model_name")

    # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
    logger.info(
        f"--- [STRATEGY START] ID: {strategy_id} | User: '{user_id}' | Symbol: {symbol} | Model: '{model_name}' ---")
    # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^

    try:
        # 1. 获取信号
        quant_prediction = system.strategy_manager.get_signal_for_autotrader(config)
        if not quant_prediction or 'message' in quant_prediction:
            raise ValueError(f"Prediction failed: {quant_prediction.get('message', 'Unknown error')}")

        # 2. 决策
        quant_dir = quant_prediction.get('direction', -1)
        confidence = quant_prediction.get('confidence', 0.0)
        conf_threshold = config.get('ml_confidence_threshold', 0.65)

        final_decision = "Hold"
        if quant_dir == 1 and confidence >= conf_threshold:
            final_decision = "Buy"
        elif quant_dir == 0 and confidence >= conf_threshold:
            final_decision = "Sell"

        # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
        logger.info(
            f"[{strategy_id}] Signal Details: Direction={quant_dir}, Confidence={confidence:.3f}, Threshold={conf_threshold} -> Decision: {final_decision}")
        # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^

        # 3. 获取当前持仓
        portfolio = system.persistence_manager.load_portfolio(user_id)
        if not portfolio: raise ValueError(f"Could not load portfolio for user '{user_id}'.")
        current_position = portfolio.get('positions', {}).get(symbol, {})
        has_position = current_position and current_position.get('quantity', 0) > 0

        # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
        logger.info(
            f"[{strategy_id}] Portfolio Check: User '{user_id}' currently {'HAS' if has_position else 'DOES NOT HAVE'} a position in {symbol}.")
        # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^

        # 4. 准备订单
        order_to_execute = None
        if final_decision == "Buy" and not has_position:
            order_to_execute = {
                "symbol": symbol, "quantity": config.get("trade_quantity"),
                "direction": "Buy", "order_type": "Market Order", "price": None
            }
        elif final_decision == "Sell" and has_position:
            order_to_execute = {
                "symbol": symbol, "quantity": current_position.get('quantity'),
                "direction": "Sell", "order_type": "Market Order", "price": None
            }

        # 5. 执行交易
        if order_to_execute:
            logger.warning(f"[{strategy_id}] ACTION: Executing trade for User '{user_id}': {order_to_execute}")
            trade_result = system.execute_trade_for_user(order_to_execute, user_id=user_id)
            logger.info(f"[{strategy_id}] Trade execution result: {trade_result}")
        else:
            # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
            reason = ""
            if final_decision == "Buy" and has_position:
                reason = "Already holding a position."
            elif final_decision == "Sell" and not has_position:
                reason = "No position to sell."
            elif final_decision == "Hold":
                reason = "Signal is 'Hold'."
            else:
                reason = "Confidence below threshold."
            logger.info(f"[{strategy_id}] No trade action required. Reason: {reason}")
            # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^

    except Exception as e:
        logger.error(f"[{strategy_id}] FAILED to execute strategy: {e}", exc_info=True)
    finally:
        # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
        system.persistence_manager.update_strategy_last_executed(strategy_id)
        logger.info(f"--- [STRATEGY END] ID: {strategy_id} ---")
        # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^


def trading_tick(system: TradingSystem):
    # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
    logger.info("-------------------- AutoTrader Tick Start --------------------")
    # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^

    enabled_strategies = system.persistence_manager.load_enabled_auto_strategies()

    # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
    if not enabled_strategies:
        logger.info("No enabled auto-strategies found in database. Tick finished.")
        logger.info("--------------------  AutoTrader Tick End  --------------------")
        return

    logger.info(f"Found {len(enabled_strategies)} enabled strategies to process.")
    # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^

    for strategy_config in enabled_strategies:
        strategy_id = strategy_config.get('strategy_id', 'Unknown')
        symbol = strategy_config.get('symbol', 'Unknown')
        try:
            if not is_trading_time(symbol, system.config.MARKET_HOURS):
                # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
                logger.debug(f"Skipping strategy '{strategy_id}' for {symbol}: Outside of trading hours.")
                # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^
                continue

            if strategy_config.get("type") == "ml_quant":
                execute_ml_quant_strategy(system, strategy_config)
            # (未来可扩展其他策略类型)
        except Exception as e:
            logger.error(f"Error processing strategy '{strategy_id}': {e}", exc_info=True)

    # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
    logger.info("--------------------  AutoTrader Tick End  --------------------")
    # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^


def main():
    if not APS_AVAILABLE: return
    logger.info("Initializing TradingSystem for AutoTrader Service...")
    try:
        system = TradingSystem()
        logger.info("TradingSystem initialized successfully.")
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to initialize TradingSystem. Service cannot start. Error: {e}",
                        exc_info=True)
        return

    scheduler = BlockingScheduler(timezone="Asia/Shanghai")
    scheduler.add_job(trading_tick, 'cron', minute='*', second='5', args=[system])
    logger.info("APScheduler started. AutoTrader service is now running...")
    logger.info("Log file is located at: autotrader_service.log")
    logger.info("Press Ctrl+C to stop the service.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutdown signal received. Stopping AutoTrader Service.")
    finally:
        if scheduler.running:
            scheduler.shutdown()
        logger.info("==========================================================")
        logger.info("=            AutoTrader Service Shut Down            =")
        logger.info("==========================================================")


if __name__ == "__main__":
    main()