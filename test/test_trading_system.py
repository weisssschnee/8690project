import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import time
import os
import sys
from datetime import datetime

# Ensure the project's root directory is in the Python path
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.system import TradingSystem
from core.config import Config
from core.strategy.ml_strategy import TextFeatureExtractor


# =============================================================================
# 1. SETUP FIXTURES
# =============================================================================

@pytest.fixture(scope="function")
def test_config(tmp_path_factory):
    """Creates a temporary, isolated configuration for the test run."""
    test_dir = tmp_path_factory.mktemp("test_run_data")

    class TestConfig(Config):
        def __init__(self):
            super().__init__()
            self.DATABASE_URL = f"sqlite:///{test_dir / 'test_trading.db'}"
            self.DATA_PATH = test_dir
            self.ML_MODELS_BASE_PATH = test_dir / "models"
            self.ML_MODELS_BASE_PATH.mkdir(exist_ok=True)
            self.FUTU_ENABLED = False
            # We don't need to set API keys here anymore, monkeypatch will handle it.

    return TestConfig()


# vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
# FIX: This is the definitive fix for the API key issue during testing.
@pytest.fixture(scope="function")
def trading_system(test_config, monkeypatch):
    """
    Initializes the TradingSystem in a controlled environment by
    first simulating the presence of necessary environment variables.
    """
    # --- STEP 1: Simulate Environment Variables BEFORE Config is loaded ---
    # monkeypatch.setenv is the correct way to set environment variables for a test.
    # This simulates what `load_dotenv` would do in a real run.
    print("\n--- Monkeypatching environment variables for test ---")
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key_for_testing")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test_deepseek_key_for_testing")
    monkeypatch.setenv("FINNHUB_KEY", "test_finnhub_key_for_testing")
    # Add any other keys your system strictly requires during initialization here.

    # --- STEP 2: Patch the Config class to use our temporary test_config ---
    # This ensures isolated paths (database, models, etc.).
    monkeypatch.setattr('core.system.Config', lambda: test_config)

    # --- STEP 3: Now, initialize the TradingSystem ---
    # When TradingSystem creates its Config instance, it will:
    # 1. Use our TestConfig class (due to setattr).
    # 2. Inside TestConfig, os.getenv("DEEPSEEK_API_KEY") will find the value
    #    we just set with monkeypatch.setenv.
    system = TradingSystem()

    # Give background threads a moment to initialize
    time.sleep(1)

    return system


# ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^


@pytest.fixture(scope="module")
def mock_historical_data():
    """Generates a realistic-looking but fake historical price DataFrame."""
    date_range = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=500, freq='D'))
    data = {
        'open': np.random.uniform(95, 105, size=500),
        'high': np.random.uniform(105, 110, size=500),
        'low': np.random.uniform(90, 95, size=500),
        'close': np.random.uniform(100, 105, size=500),
        'volume': np.random.randint(1_000_000, 5_000_000, size=500)
    }
    df = pd.DataFrame(data, index=date_range)
    df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, 2, size=500)
    df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, 2, size=500)
    df.index.name = 'date'
    return df


# =============================================================================
# 2. THE INTEGRATION TEST (Logic is unchanged)
# =============================================================================

def test_end_to_end_ml_backtest_workflow_with_latency_check(trading_system, mock_historical_data, monkeypatch):
    """
    This is the main integration test. It verifies the entire workflow and measures performance.
    """
    # --- STEP 1: Mock external dependencies ---
    monkeypatch.setattr(
        trading_system.data_manager,
        'get_historical_data',
        lambda *args, **kwargs: mock_historical_data
    )

    def mock_text_features(*args, **kwargs):
        dates = kwargs.get('dates', mock_historical_data.index)
        text_df = pd.DataFrame(index=dates)
        text_df['gemini_avg_sentiment'] = 0.0
        text_df['gemini_news_count'] = 3.0
        return text_df

    if trading_system.strategy_manager.text_feature_extractor:
        monkeypatch.setattr(
            trading_system.strategy_manager.text_feature_extractor,
            'get_and_extract_features_for_backtest',
            mock_text_features
        )

    # --- STEP 2: Train a model and measure latency ---
    print("\n--- Running Model Training ---")
    start_train_time = time.perf_counter()

    model_name_to_train = "默认随机森林"
    train_result = trading_system.strategy_manager.ml_strategy_instance.train(
        data=mock_historical_data,
        symbol="TEST_AAPL",
        model_display_name_to_save=model_name_to_train
    )

    end_train_time = time.perf_counter()
    training_latency = end_train_time - start_train_time
    print(f"--- Model Training Latency: {training_latency:.4f} seconds ---")

    assert train_result['success'] is True, f"Model training failed: {train_result.get('message')}"
    assert 'test_score' in train_result
    print(f"--- Model Training Successful (Test Score: {train_result['test_score']:.2f}) ---")

    # --- STEP 3: Run a backtest and measure latency ---
    print("\n--- Running Backtest ---")
    strategy_config = {
        "type": "ml_quant",
        "ml_model_name": model_name_to_train,
        "alpha_threshold": 0.1,
        "use_llm": False
    }

    start_backtest_time = time.perf_counter()

    backtest_result = trading_system.strategy_manager.backtest_strategy(
        symbol="TEST_AAPL",
        data=mock_historical_data,
        strategy_config=strategy_config,
        initial_capital=10000.0,
        commission_rate=0.001
    )

    end_backtest_time = time.perf_counter()
    backtest_latency = end_backtest_time - start_backtest_time
    print(f"--- Backtest Execution Latency: {backtest_latency:.4f} seconds ---")

    # --- STEP 4: Verify the Final Result ---
    print("\n--- Verifying Backtest Result ---")
    assert backtest_result is not None
    assert backtest_result['success'] is True, f"Backtest failed: {backtest_result.get('message')}"

    stats = backtest_result.get('stats')
    assert isinstance(stats, dict)

    expected_metrics = {
        'total_return': float, 'annual_return': float, 'sharpe_ratio': float,
        'max_drawdown': float, 'win_rate': float, 'trades': int
    }

    for metric, expected_type in expected_metrics.items():
        assert metric in stats, f"Metric '{metric}' is missing"
        assert isinstance(stats[metric], expected_type), f"Metric '{metric}' has wrong type"

    history_df = backtest_result.get('history_df')
    assert isinstance(history_df, pd.DataFrame)
    assert not history_df.empty

    # --- STEP 5: Latency and Performance Evaluation ---
    print("\n--- Evaluating Performance and Timestamps ---")

    MAX_TRAINING_LATENCY = 30.0
    MAX_BACKTEST_LATENCY = 15.0

    assert training_latency < MAX_TRAINING_LATENCY, f"Training took {training_latency:.2f}s, exceeding the {MAX_TRAINING_LATENCY}s threshold."
    assert backtest_latency < MAX_BACKTEST_LATENCY, f"Backtest took {backtest_latency:.2f}s, exceeding the {MAX_BACKTEST_LATENCY}s threshold."

    print("--- Latency Thresholds PASSED ---")

    assert isinstance(history_df.index, pd.DatetimeIndex), "history_df index should be a DatetimeIndex"

    print("--- Timestamp Integrity PASSED ---")
    print("\n✅ End-to-end integration and performance test PASSED.")