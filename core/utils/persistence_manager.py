# core/utils/persistence_manager.py
import sqlite3
import json
import pandas as pd
import hashlib
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import numpy as np
import streamlit as st

logger = logging.getLogger(__name__)


class NumpyJSONEncoder(json.JSONEncoder):
    """
    一个自定义的 JSON 编码器，用于处理 NumPy 的数据类型。
    """

    def default(self, obj):
        if isinstance(obj, (
                np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
                np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            if np.isinf(obj) or np.isnan(obj):
                return None  # 将 inf/nan 转换为 null
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime, pd.Timestamp)):  # 处理 datetime 和 Timestamp
            return obj.isoformat()
        return super(NumpyJSONEncoder, self).default(obj)


class PersistenceManager:
    """
    统一管理所有持久化数据，包括结果缓存和账户状态。
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._create_tables()
            logger.info(f"PersistenceManager initialized with database at: {self.db_path}")
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to initialize PersistenceManager database: {e}", exc_info=True)
            raise

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接"""
        return sqlite3.connect(self.db_path, timeout=10)

    def _create_tables(self):
        """创建所有需要的数据库表 (如果它们不存在)"""
        with self._get_conn() as conn:
            cursor = conn.cursor()

            # --- 版本控制表 ---
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS db_version (
                version_id INTEGER PRIMARY KEY,
                version_tag TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
            cursor.execute("INSERT OR IGNORE INTO db_version (version_id, version_tag) VALUES (1, ?)",
                           ("v3_multi_user",))

            # --- 投资组合状态表 (支持多用户) ---
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_state (
                user_id TEXT PRIMARY KEY,
                portfolio_json TEXT NOT NULL,
                last_update DATETIME NOT NULL
            )
            """)

            # --- 交易历史表 (支持多用户) ---
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_history (
                trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                total REAL NOT NULL,
                commission REAL,
                is_mock BOOLEAN,
                timestamp DATETIME NOT NULL
            )
            """)

            # --- 回测缓存表 ---
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_cache (
                config_hash TEXT PRIMARY KEY,
                config_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # --- 预测缓存表 ---
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_cache (
                request_hash TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                model_name TEXT NOT NULL,
                use_llm BOOLEAN NOT NULL,
                result_json TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
            cursor.execute("""
                        CREATE TABLE IF NOT EXISTS portfolio_history (
                            history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id TEXT NOT NULL,
                            timestamp DATETIME NOT NULL,
                            total_value REAL NOT NULL,
                            cash REAL NOT NULL,
                            positions_json TEXT -- 存储当时的持仓快照 (可选)
                        )
                        """)

            # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
            # --- 自动化策略配置表 ---
            cursor.execute("""
                                   CREATE TABLE IF NOT EXISTS automated_strategies (
                                       strategy_id TEXT PRIMARY KEY,
                                       user_id TEXT NOT NULL,
                                       config_json TEXT NOT NULL,
                                       is_enabled BOOLEAN DEFAULT TRUE,
                                       last_executed DATETIME
                                   )
                                   """)
            # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^
            conn.commit()

    # --- Backtest Cache Methods ---
    def _generate_backtest_hash(self, config: Dict) -> str:
        config_string = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_string.encode('utf-8')).hexdigest()

    def get_backtest_result(self, config: Dict) -> Optional[Dict]:
        try:
            config_hash = self._generate_backtest_hash(config)
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT result_json FROM backtest_cache WHERE config_hash = ?", (config_hash,))
                row = cursor.fetchone()
                if row and row[0]:
                    logger.info(f"Backtest cache HIT for config hash: {config_hash[:10]}...")
                    result = json.loads(row[0])
                    if 'history_df' in result and isinstance(result['history_df'], str):
                        try:
                            history_df = pd.read_json(result['history_df'], orient='split')
                            if not isinstance(history_df.index, pd.DatetimeIndex):
                                history_df.index = pd.to_datetime(history_df.index)
                            result['history_df'] = history_df
                        except Exception as e:
                            logger.error(f"Failed to reload history_df from cache: {e}");
                            result['history_df'] = pd.DataFrame()
                    return result
            logger.info(f"Backtest cache MISS for config hash: {config_hash[:10]}...")
            return None
        except Exception as e:
            logger.error(f"Error getting backtest result from cache: {e}", exc_info=True);
            return None

    def set_backtest_result(self, config: Dict, result: Dict):
        try:
            config_hash = self._generate_backtest_hash(config)
            result_to_store = result.copy()
            if 'history_df' in result_to_store and isinstance(result_to_store['history_df'], pd.DataFrame):
                result_to_store['history_df'] = result_to_store['history_df'].to_json(orient='split', date_format='iso')

            with self._get_conn() as conn:
                config_json_str = json.dumps(config, sort_keys=True, cls=NumpyJSONEncoder)
                result_json_str = json.dumps(result_to_store, cls=NumpyJSONEncoder)
                conn.execute(
                    "INSERT OR REPLACE INTO backtest_cache (config_hash, config_json, result_json) VALUES (?, ?, ?)",
                    (config_hash, config_json_str, result_json_str)
                )
                conn.commit()
                logger.info(f"Backtest result saved to cache for config hash: {config_hash[:10]}...")
        except Exception as e:
            logger.error(f"Error setting backtest result to cache: {e}", exc_info=True)

    # --- Prediction Cache Methods ---
    def _generate_prediction_hash(self, symbol: str, model_name: str, use_llm: bool) -> str:
        config_string = f"{symbol}-{model_name}-{use_llm}"
        return hashlib.sha256(config_string.encode('utf-8')).hexdigest()

    def get_prediction_result(self, symbol: str, model_name: str, use_llm: bool) -> Optional[Dict]:
        try:
            request_hash = self._generate_prediction_hash(symbol, model_name, use_llm)
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT result_json FROM prediction_cache WHERE request_hash = ?", (request_hash,))
                row = cursor.fetchone()
                if row and row[0]:
                    logger.info(f"Prediction cache HIT for request: {symbol}/{model_name}/LLM={use_llm}")
                    return json.loads(row[0])
            logger.info(f"Prediction cache MISS for request: {symbol}/{model_name}/LLM={use_llm}")
            return None
        except Exception as e:
            logger.error(f"Error getting prediction result from cache: {e}", exc_info=True);
            return None

    def set_prediction_result(self, symbol: str, model_name: str, use_llm: bool, result: Dict):
        try:
            request_hash = self._generate_prediction_hash(symbol, model_name, use_llm)
            with self._get_conn() as conn:
                result_json_str = json.dumps(result, cls=NumpyJSONEncoder)
                conn.execute(
                    "INSERT OR REPLACE INTO prediction_cache (request_hash, symbol, model_name, use_llm, result_json) VALUES (?, ?, ?, ?, ?)",
                    (request_hash, symbol, model_name, use_llm, result_json_str)
                )
                conn.commit()
                logger.info(f"Prediction result saved to cache for request: {symbol}/{model_name}/LLM={use_llm}")
        except Exception as e:
            logger.error(f"Error setting prediction result to cache: {e}", exc_info=True)

    # --- Portfolio & Trades Persistence Methods ---
    def save_portfolio(self, portfolio: Dict, user_id: str):
        if not user_id: logger.error("Cannot save portfolio: user_id is missing."); return
        try:
            portfolio_json_str = json.dumps(portfolio, cls=NumpyJSONEncoder)
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO portfolio_state (user_id, portfolio_json, last_update) VALUES (?, ?, ?)",
                    (user_id, portfolio_json_str, datetime.now())
                )
                conn.commit()
            logger.debug(f"Portfolio state for user '{user_id}' saved to database.")
        except Exception as e:
            logger.error(f"Error saving portfolio for user '{user_id}': {e}", exc_info=True)

    def load_portfolio(self, user_id: str) -> Optional[Dict]:
        if not user_id: logger.error("Cannot load portfolio: user_id is missing."); return None
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT portfolio_json FROM portfolio_state WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    logger.info(f"Portfolio state for user '{user_id}' loaded from database.")
                    return json.loads(row[0])
            logger.warning(f"No portfolio state found in database for user '{user_id}'.")
            return None
        except Exception as e:
            logger.error(f"Error loading portfolio for user '{user_id}': {e}", exc_info=True);
            return None

    def add_trade(self, trade: Dict, user_id: str):
        if not user_id: logger.error("Cannot add trade: user_id is missing."); return
        try:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO trade_history (user_id, symbol, direction, quantity, price, total, commission, is_mock, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        trade.get('symbol'), trade.get('direction'), trade.get('quantity'),
                        trade.get('price'), trade.get('total'), trade.get('commission'),
                        trade.get('is_mock', True),
                        # 确保 timestamp 是数据库兼容的格式
                        trade.get('timestamp', datetime.now()).isoformat() if isinstance(trade.get('timestamp'),
                                                                                         datetime) else trade.get(
                            'timestamp')
                    )
                )
                conn.commit()
            logger.info(f"Trade for user '{user_id}' symbol '{trade.get('symbol')}' added to database.")
        except Exception as e:
            logger.error(f"Error adding trade for user '{user_id}': {e}", exc_info=True)

    def load_trades(self, user_id: str, limit: int = 200) -> List[Dict]:
        if not user_id: logger.error("Cannot load trades: user_id is missing."); return []
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM trade_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                               (user_id, limit,))
                rows = cursor.fetchall()
                cols = [desc[0] for desc in cursor.description]
                trades = [dict(zip(cols, row)) for row in rows]
                for trade in trades:
                    if 'timestamp' in trade and isinstance(trade['timestamp'], str):
                        try:
                            trade['timestamp'] = datetime.fromisoformat(trade['timestamp'])
                        except ValueError:
                            pass  # Keep as string if format is wrong
                logger.info(f"Loaded {len(trades)} trades from database history for user '{user_id}'.")
                return list(reversed(trades))
        except Exception as e:
            logger.error(f"Error loading trades for user '{user_id}': {e}", exc_info=True);
            return []

    def clear_trades(self, user_id: str):
        """[新增] 清空指定用户的所有交易历史。"""
        if not user_id: logger.error("Cannot clear trades: user_id is missing."); return
        try:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM trade_history WHERE user_id = ?", (user_id,))
                conn.commit()
            logger.warning(f"Cleared all trade history for user '{user_id}'.")
        except Exception as e:
            logger.error(f"Error clearing trades for user '{user_id}': {e}", exc_info=True)

    def add_portfolio_history_entry(self, history_entry: Dict, user_id: str):
        """将一条投资组合历史记录添加到数据库。"""
        if not user_id: return
        try:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO portfolio_history (user_id, timestamp, total_value, cash, positions_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        history_entry.get('timestamp', datetime.now()),
                        history_entry.get('total_value'),
                        history_entry.get('cash'),
                        json.dumps(history_entry.get('positions', {}), cls=NumpyJSONEncoder)
                    )
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error adding portfolio history for user '{user_id}': {e}", exc_info=True)

    def load_portfolio_history(self, user_id: str, limit: int = 1000) -> List[Dict]:
        """从数据库加载指定用户的投资组合历史。"""
        if not user_id: return []
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT timestamp, total_value, cash, positions_json FROM portfolio_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (user_id, limit,))
                rows = cursor.fetchall()
                cols = [desc[0] for desc in cursor.description]
                history = [dict(zip(cols, row)) for row in rows]
                for entry in history:
                    if 'timestamp' in entry and isinstance(entry['timestamp'], str):
                        entry['timestamp'] = datetime.fromisoformat(entry['timestamp'])
                    if 'positions_json' in entry and isinstance(entry['positions_json'], str):
                        entry['positions'] = json.loads(entry['positions_json'])
                        del entry['positions_json']
                logger.info(f"Loaded {len(history)} portfolio history entries for user '{user_id}'.")
                return list(reversed(history))
        except Exception as e:
            logger.error(f"Error loading portfolio history for user '{user_id}': {e}", exc_info=True);
            return []

    def update_strategy_last_executed(self, strategy_id: str):
            """[新增] 更新策略的最后执行时间。"""
            if not strategy_id: return
            try:
                with self._get_conn() as conn:
                    conn.execute(
                        "UPDATE automated_strategies SET last_executed = ? WHERE strategy_id = ?",
                        (datetime.now(), strategy_id)
                    )
                    conn.commit()
                logger.debug(f"Updated last_executed timestamp for strategy '{strategy_id}'.")
            except Exception as e:
                logger.error(f"Error updating last_executed for strategy '{strategy_id}': {e}", exc_info=True)

        # 在 load_strategy_config 方法中，加载更多字段
    def load_strategy_config(self, strategy_id: str, user_id: str) -> Optional[Dict]:
            """[修改] 加载单个策略的配置，并包含状态信息。"""
            if not all([strategy_id, user_id]): return None
            with self._get_conn() as conn:
                cursor = conn.cursor()
                # 加载 config_json 和 last_executed
                cursor.execute(
                    "SELECT config_json, last_executed FROM automated_strategies WHERE strategy_id = ? AND user_id = ?",
                    (strategy_id, user_id))
                row = cursor.fetchone()
                if row:
                    config = json.loads(row[0])
                    config['last_executed'] = row[1]  # 添加 last_executed 字段
                    return config
            return None

    # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
    def save_strategy_config(self, strategy_config: Dict):
        """[最终版] 保存或更新一个自动化策略配置。"""
        strategy_id = strategy_config.get("strategy_id")
        user_id = strategy_config.get("user_id")
        is_enabled = strategy_config.get("enabled", True)
        config_json = json.dumps(strategy_config, cls=NumpyJSONEncoder)

        if not all([strategy_id, user_id]):
            logger.error(f"Cannot save strategy config due to missing IDs: {strategy_config}")
            return

        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO automated_strategies (strategy_id, user_id, config_json, is_enabled) VALUES (?, ?, ?, ?)",
                (strategy_id, user_id, config_json, is_enabled)
            )
            conn.commit()
        logger.info(f"Saved/Updated auto-strategy '{strategy_id}' for user '{user_id}'.")

    def delete_strategy_config(self, strategy_id: str, user_id: str):
        """[最终版] 删除一个自动化策略配置。"""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM automated_strategies WHERE strategy_id = ? AND user_id = ?",
                         (strategy_id, user_id))
            conn.commit()
        logger.info(f"Deleted auto-strategy '{strategy_id}' for user '{user_id}'.")

    def load_enabled_auto_strategies(self) -> List[Dict]:
        """[最终版] 从数据库加载所有已启用的自动化策略配置。"""
        strategies = []
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT config_json FROM automated_strategies WHERE is_enabled = TRUE")
                rows = cursor.fetchall()
                for row in rows:
                    strategies.append(json.loads(row[0]))
            logger.info(f"Loaded {len(strategies)} enabled auto-strategies from database.")
        except Exception as e:
            logger.error(f"Failed to load enabled auto-strategies: {e}", exc_info=True)
        return strategies

    def load_strategy_config(self, strategy_id: str, user_id: str) -> Optional[Dict]:
        """[新增] 加载单个策略的配置，用于检查其启用状态。"""
        if not all([strategy_id, user_id]):
            return None
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT config_json FROM automated_strategies WHERE strategy_id = ? AND user_id = ?",
                           (strategy_id, user_id))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None

    def load_all_strategies_for_user(self, user_id: str) -> List[Dict]:
        """[新增] 加载指定用户的所有策略配置（无论启用与否）。"""
        strategies = []
        if not user_id: return strategies
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT config_json FROM automated_strategies WHERE user_id = ?", (user_id,))
                rows = cursor.fetchall()
                for row in rows:
                    strategies.append(json.loads(row[0]))
            logger.info(f"Loaded {len(strategies)} strategies for user '{user_id}'.")
        except Exception as e:
            logger.error(f"Failed to load strategies for user '{user_id}': {e}", exc_info=True)
        return strategies

    def save_strategy_last_decision(self, strategy_id: str, decision: Dict):
        """[新增] 保存策略的最后一次决策到其配置中。"""
        # 注意：这是一种简化实现。在高并发系统中，最好将决策存储在独立的表中。
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT config_json, user_id FROM automated_strategies WHERE strategy_id = ?", (strategy_id,))
                row = cursor.fetchone()
                if row:
                    config = json.loads(row[0])
                    user_id = row[1]
                    config["last_decision"] = decision
                    self.save_strategy_config(config) # save_strategy_config 是 INSERT OR REPLACE，可以用于更新
        except Exception as e:
             logger.error(f"Failed to save last decision for strategy '{strategy_id}': {e}", exc_info=True)
    # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^