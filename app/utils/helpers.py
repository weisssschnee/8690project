# app/utils/helpers.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def calculate_returns(prices: pd.Series) -> pd.Series:
    """计算收益率"""
    return prices.pct_change()


def calculate_volatility(returns: pd.Series, window: int = 252) -> float:
    """计算波动率"""
    return returns.std() * np.sqrt(window)


def calculate_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float,
        window: int = 252
) -> float:
    """计算夏普比率"""
    excess_returns = returns - risk_free_rate / window
    return np.sqrt(window) * excess_returns.mean() / returns.std()


def calculate_drawdown(prices: pd.Series) -> pd.Series:
    """计算回撤"""
    rolling_max = prices.expanding().max()
    drawdown = (prices - rolling_max) / rolling_max
    return drawdown


def format_currency(value: float) -> str:
    """格式化货币数值"""
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """格式化百分比"""
    return f"{value:.2%}"


def validate_order(order: Dict) -> bool:
    """验证订单"""
    required_fields = ['symbol', 'quantity', 'type']
    return all(field in order for field in required_fields)


def calculate_position_value(
        quantity: int,
        price: float
) -> float:
    """计算持仓价值"""
    return quantity * price


def is_market_open(market: str) -> bool:
    """检查市场是否开放"""
    now = datetime.now()

    if market == "US":
        # 美股市场时间 (EST)
        market_open = now.replace(hour=9, minute=30, second=0)
        market_close = now.replace(hour=16, minute=0, second=0)
    elif market == "CN":
        # 中国市场时间
        morning_open = now.replace(hour=9, minute=30, second=0)
        morning_close = now.replace(hour=11, minute=30, second=0)
        afternoon_open = now.replace(hour=13, minute=0, second=0)
        afternoon_close = now.replace(hour=15, minute=0, second=0)

        return (morning_open <= now <= morning_close) or \
            (afternoon_open <= now <= afternoon_close)

    return market_open <= now <= market_close


def handle_api_error(error: Exception) -> Dict:
    """处理API错误"""
    logger.error(f"API Error: {str(error)}")
    return {
        'success': False,
        'message': f"API错误: {str(error)}",
        'timestamp': datetime.now().isoformat()
    }