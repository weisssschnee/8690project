# tests/test_performance.py
import time
import sys
from pathlib import Path
import os
import streamlit as st

# --- 路径设置 (保持不变) ---
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- 导入 (保持不变) ---
from core.data.manager import DataManager
from core.config import Config


# --- 测试函数 (修改了名称) ---
def test_benchmark_data_fetch():  # <--- 修改函数名
    """
    Tests DataManager performance, especially the caching effect.
    This is now a pytest test case.
    """
    print("--- Starting Performance Benchmark Test ---")

    # 1. Initialization (保持不变)
    try:
        env_path = project_root / '.env.txt'
        if env_path.exists():
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=env_path)
            print(f"Test: Loaded .env from {env_path}")
        else:
            print(f"Test: .env file not found at {env_path}")

        config_instance = Config()
        data_manager = DataManager(config_instance)
        print("DataManager initialized.")
    except Exception as e:
        print(f"Initialization failed: {e}")
        assert False, f"Setup failed: {e}"  # 在测试中，用 assert False 抛出失败

    symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', '600519.SH']

    # --- First Run (Cold Start) ---
    print("\n--- First Run (Cold Start, should call API)... ---")
    st.cache_data.clear()

    start_time_cold = time.time()
    for symbol in symbols:
        data = data_manager.get_historical_data(symbol, days=30)
        assert data is not None, f"Cold fetch failed for {symbol}"
        assert not data.empty, f"Cold fetch returned empty data for {symbol}"
    end_time_cold = time.time()

    duration_cold = end_time_cold - start_time_cold
    print(f"\nFirst Run (Cold Start) Total Time: {duration_cold:.2f} seconds")

    # --- Second Run (Warm Start / Cached) ---
    print("\n--- Second Run (Warm Start, should hit cache)... ---")

    start_time_hot = time.time()
    for symbol in symbols:
        data = data_manager.get_historical_data(symbol, days=30)
        assert data is not None, f"Warm fetch failed for {symbol}"
        assert not data.empty, f"Warm fetch returned empty data for {symbol}"
    end_time_hot = time.time()

    duration_hot = end_time_hot - start_time_hot
    print(f"\nSecond Run (Warm Start) Total Time: {duration_hot:.2f} seconds")

    # --- Performance Comparison and Assertion ---
    print("\n--- Performance Comparison ---")
    print(f"Cold Start Time: {duration_cold:.2f} s")
    print(f"Warm Start Time: {duration_hot:.2f} s")

    # 在测试中，我们用断言 (assert) 来验证结果
    assert duration_hot < duration_cold, "Cached run should be faster than non-cached run."

    # 例如，我们可以断言缓存后的速度至少快 10 倍
    if duration_hot > 0:
        performance_gain = duration_cold / duration_hot
        print(f"Performance Gain: {performance_gain:.2f}x")
        assert performance_gain > 10, "Expected at least 10x performance gain from caching."
    else:
        print("Warm start was too fast to measure gain, which is a good sign.")

# --- 移除 if __name__ == "__main__": 块 ---
# pytest 会自动找到并运行 test_benchmark_data_fetch 函数
# 如果你还想让它能作为脚本运行，可以保留 if 块并调用 test_benchmark_data_fetch()
# if __name__ == "__main__":
#     test_benchmark_data_fetch()
