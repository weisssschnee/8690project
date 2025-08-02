import yfinance as yf
import pandas as pd
import numpy as np


def get_vix():
    try:
        # 尝试直接从Yahoo获取
        vix = yf.download("^VIX", period="1d", progress=False)
        if not vix.empty:
            return vix['Close'].iloc[0]

        # 备用方案：使用标普500波动率
        spy = yf.download("SPY", period="30d", progress=False)
        spy_vol = spy['Close'].pct_change().std() * np.sqrt(252) * 100
        return spy_vol * 1.5  # 经验系数调整

    except Exception as e:
        print(f"获取数据失败: {e}")
        return 20.0  # 默认值


# 步骤3：使用VIX数据
current_vix = get_vix()
print(f"当前市场波动率指数(VIX): {current_vix:.2f}")

# 根据VIX值制定策略
if current_vix > 30:
    print("市场恐慌状态 - 建议保守策略")
elif current_vix < 15:
    print("市场自满状态 - 注意风险")
else:
    print("市场正常波动 - 中性策略")