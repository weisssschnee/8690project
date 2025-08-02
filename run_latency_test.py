# run_latency_test.py (Corrected Version)
import time
import pandas as pd
import numpy as np
from datetime import datetime
from core.system import TradingSystem
import streamlit as st
import logging
import random

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


def run_latency_test(num_orders=1000):
    """执行订单提交延迟测试，并模拟网络延迟"""
    print(f"\n--- Starting Latency Test for {num_orders} orders with Network Simulation ---")

    internal_latencies = []
    simulated_network_latencies = []
    total_simulated_latencies = []

    # 初始化系统
    system = TradingSystem()

    # 设置账户状态
    st.session_state.portfolio['cash'] = 1_000_000_000
    logger.info(f"Test setup: Initial cash set to {st.session_state.portfolio['cash']:,}")

    order_data = {
        "symbol": "AAPL",
        "quantity": 1,
        "price": 150.00,
        "direction": "Buy",
        "order_type": "Market Order"
    }

    print("Warming up the system with a pre-run...")
    system.execute_trade(order_data.copy())
    print("Warm-up complete. Starting main test loop...")

    for i in range(num_orders):
        if (i + 1) % 100 == 0:
            print(f"Executing order {i + 1}/{num_orders}...")

        # --- 步骤 1: 测量纯粹的内部处理延迟 ---
        start_time_internal = time.perf_counter()
        system.execute_trade(order_data.copy())
        end_time_internal = time.perf_counter()

        internal_latency_ms = (end_time_internal - start_time_internal) * 1000
        internal_latencies.append(internal_latency_ms)

        # --- 步骤 2: 独立地模拟网络延迟 ---
        # 模拟一个典型的网络往返时间，例如在20ms到100ms之间
        simulated_network_latency_ms = random.uniform(20.0, 100.0)
        simulated_network_latencies.append(simulated_network_latency_ms)

        # --- 步骤 3: 计算总模拟延迟 ---
        total_simulated_latency_ms = internal_latency_ms + simulated_network_latency_ms
        total_simulated_latencies.append(total_simulated_latency_ms)

        # 实际测试中不需要sleep，因为我们已经独立模拟了网络延迟
        # time.sleep(0.01) # 可以移除或保留一个极小值以防CPU过载

    print("--- Latency Test Finished ---")

    # --- 分析并打印结果 ---
    if total_simulated_latencies:
        internal_series = pd.Series(internal_latencies)
        network_series = pd.Series(simulated_network_latencies)
        total_series = pd.Series(total_simulated_latencies)

        print("\n--- Performance Test Results ---")
        print(f"Total Orders Tested: {len(total_series)}")

        print("\n--- Internal Processing Latency (System's own speed) ---")
        print(f"Average:    {internal_series.mean():.4f} ms")
        print(f"Median:     {internal_series.median():.4f} ms")
        print(f"Std Dev:    {internal_series.std():.4f} ms")

        print("\n--- Simulated Network Latency (External factor) ---")
        print(f"Average:    {network_series.mean():.4f} ms")

        print("\n--- Total Simulated End-to-End Latency ---")
        print(f"Average:    {total_series.mean():.4f} ms")
        print(f"Median:     {total_series.median():.4f} ms")
        print(f"Std Dev:    {total_series.std():.4f} ms")
        print(f"Minimum:    {total_series.min():.4f} ms")
        print(f"Maximum:    {total_series.max():.4f} ms")
        print(f"95th Percentile: {total_series.quantile(0.95):.4f} ms")
        print(f"99th Percentile: {total_series.quantile(0.99):.4f} ms")

        # 将结果保存到CSV
        results_df = pd.DataFrame({
            'internal_latency_ms': internal_latencies,
            'simulated_network_latency_ms': simulated_network_latencies,
            'total_simulated_latency_ms': total_simulated_latencies
        })
        results_df.to_csv("latency_test_results.csv", index_label="order_num")
        print("\nResults saved to latency_test_results.csv")
    else:
        print("No latency results were recorded.")


def run_data_processing_test(num_iterations=100):
    """执行数据处理延迟测试"""
    print(f"\n--- Starting Data Processing Test for {num_iterations} iterations ---")

    from core.analysis.technical import TechnicalAnalyzer
    from core.config import Config

    config = Config()
    tech_analyzer = TechnicalAnalyzer(config)

    data = pd.DataFrame({
        'open': np.random.rand(100) * 10 + 150,
        'high': np.random.rand(100) * 10 + 155,
        'low': np.random.rand(100) * 10 + 145,
        'close': np.random.rand(100) * 10 + 150,
        'volume': np.random.rand(100) * 100000 + 5000000
    })

    latencies = []
    for _ in range(num_iterations):
        start_time = time.perf_counter()
        _ = tech_analyzer.analyze(data.copy())
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

    avg_latency = np.mean(latencies)
    print(f"--- Data Processing Test Finished ---")
    print(f"Average processing latency for a 100-day dataframe: {avg_latency:.4f} ms")

    return avg_latency


if __name__ == "__main__":
    run_latency_test()
    run_data_processing_test()