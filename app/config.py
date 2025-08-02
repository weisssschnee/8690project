# config.py

import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from typing import List, Dict  # 确保导入 Dict
from datetime import time

# 加载环境变量
load_dotenv()


class Config:
    """配置类"""

    def __init__(self):
        """初始化配置"""
        self._init_logging()
        self._init_trading_params()
        self._init_ml_model_config()  # <--- 新增调用
        self._validate_env()
        self.create_directories()

    def _init_logging(self):
        """初始化日志配置"""
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        self.LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        self.LOG_FILE = 'trading_system.log'

        # 配置日志
        logging.basicConfig(
            level=self.LOG_LEVEL,
            format=self.LOG_FORMAT,
            handlers=[
                # logging.FileHandler(self.LOG_FILE, encoding='utf-8'), # 暂时注释掉
                logging.StreamHandler()  # 只输出到控制台
            ]
        )
        logging.info("Simplified logging initialized (console only).")  # 添加一条测试日志

    def _init_ml_model_config(self):
        """初始化机器学习模型配置"""
        self.ML_MODELS_BASE_PATH = Path(os.getenv("ML_MODELS_BASE_PATH", "models"))
        self.ML_MODELS_BASE_PATH.mkdir(parents=True, exist_ok=True)  # 确保目录存在

        # 键是用户在UI上看到的名称，值是模型文件名（不含路径，但包含扩展名）
        self.AVAILABLE_ML_MODELS = {
            "默认随机森林": "default_random_forest.joblib",
            # 示例：您可以添加更多预定义的模型文件及其在UI上显示的名称
            # "股票A专用XGBoost": "stock_a_specific_xgboost.joblib",
        }
        # 设置默认选择的模型 (确保这个键存在于 AVAILABLE_ML_MODELS 中)
        self.DEFAULT_ML_MODEL_NAME = "默认随机森林"
        if self.DEFAULT_ML_MODEL_NAME not in self.AVAILABLE_ML_MODELS:
            if self.AVAILABLE_ML_MODELS:  # 如果列表不为空，选第一个
                self.DEFAULT_ML_MODEL_NAME = next(iter(self.AVAILABLE_ML_MODELS))
            else:  # 如果列表为空，设置一个占位符，并警告
                self.DEFAULT_ML_MODEL_NAME = "无可用模型"
                logging.warning("配置文件中 AVAILABLE_ML_MODELS 为空，且未找到默认模型。请添加模型配置。")

        # 机器学习超参数
        self.ML_HYPERPARAMETERS = {
            'RandomForestClassifier': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                # 'class_weight': 'balanced' # 类别不平衡时可以考虑
            },
            # 'XGBClassifier': { # 示例，如果您使用XGBoost
            #     'n_estimators': 150,
            #     'learning_rate': 0.05,
            #     'max_depth': 5,
            #     'random_state': 42
            # }
        }

    def _init_trading_params(self):
        """初始化交易参数"""
        # API Keys
        self.ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
        self.TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")
        self.NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
        self.FINNHUB_KEY = os.getenv("FINNHUB_KEY", "")
        self.POLYGON_KEY = os.getenv("POLYGON_KEY", "")
        self.NEWS_SOURCES = {
            'primary': {
                'type': 'newsapi',
                'key': self.NEWS_API_KEY
            },
            'fallback': {
                'type': 'rss',
                'feeds': [
                    'http://feeds.marketwatch.com/marketwatch/topstories/',
                    'http://feeds.finance.yahoo.com/rss/2.0/headline',
                    'https://seekingalpha.com/feed.xml',
                ]
            }
        }

        # 市场时间配置
        self.MARKET_HOURS = {
            'CN': {
                'morning_open': time(9, 30),
                'morning_close': time(11, 30),
                'afternoon_open': time(13, 0),
                'afternoon_close': time(15, 0)
            },
            'US': {
                'pre_market_start': time(4, 0),
                'market_open': time(9, 30),
                'market_close': time(16, 0),
                'post_market_end': time(20, 0)
            }
        }

        # 交易参数
        self.MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "0.1"))
        self.MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", "0.15"))
        self.RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.03"))
        self.COMMISSION_RATE = float(os.getenv("COMMISSION_RATE", "0.0003"))
        self.SLIPPAGE_RATE = float(os.getenv("SLIPPAGE_RATE", "0.0001"))
        self.MIN_TRADE_AMOUNT = float(os.getenv("MIN_TRADE_AMOUNT", "1000"))

        # 数据获取配置
        self.PRICE_CACHE_DURATION = int(os.getenv("PRICE_CACHE_DURATION", "5"))  # 秒
        self.RETRY_COUNT = int(os.getenv("RETRY_COUNT", "3"))
        self.RETRY_DELAY = int(os.getenv("RETRY_DELAY", "1"))  # 秒
        self.TIMEOUT = int(os.getenv("TIMEOUT", "10"))  # 秒

        # 风险控制参数
        self.VAR_CONFIDENCE = float(os.getenv("VAR_CONFIDENCE", "0.95"))
        self.VAR_LIMIT = float(os.getenv("VAR_LIMIT", "0.02"))
        self.MAX_LEVERAGE = float(os.getenv("MAX_LEVERAGE", "1.5"))
        self.VOLATILITY_LIMIT = float(os.getenv("VOLATILITY_LIMIT", "0.3"))

        # 数据库配置
        self.DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///trading.db")
        self.DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
        self.DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))

        # 机器学习配置 (旧的MODEL_PATH和DATA_PATH，新的ML_MODELS_BASE_PATH在_init_ml_model_config中)
        self.MODEL_PATH = Path(os.getenv("MODEL_PATH", "models"))  # 可以作为新模型保存时的默认目录，但MLStrategy会用ML_MODELS_BASE_PATH
        self.DATA_PATH = Path(os.getenv("DATA_PATH", "data"))
        self.FEATURE_COLUMNS = [
            'open', 'high', 'low', 'close', 'volume',
            'ma5', 'ma10', 'ma20', 'ma30', 'ma60',
            'rsi', 'macd', 'signal', 'hist',
            'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
            'atr', 'adx', 'cci'
        ]
        self.TRAIN_TEST_SPLIT = float(os.getenv("TRAIN_TEST_SPLIT", "0.8"))
        self.VALIDATION_SIZE = float(os.getenv("VALIDATION_SIZE", "0.2"))
        self.RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

        # 技术分析参数
        self.MA_PERIODS = [5, 10, 20, 30, 60]
        self.RSI_PERIOD = 14
        self.MACD_PARAMS = {
            'fast': 12,
            'slow': 26,
            'signal': 9
        }
        self.BOLLINGER_PARAMS = {
            'period': 20,
            'std_dev': 2
        }

        # 系统参数
        self.MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
        self.CACHE_TIMEOUT = int(os.getenv("CACHE_TIMEOUT", "3600"))
        self.UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL", "60"))  # 仅当不使用独立扫描服务时，这个可能用于Streamlit的刷新逻辑
        self.MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))  # 通用API重试次数

        # 情绪分析配置
        self.SENTIMENT_CONFIG = {
            'weights': {
                'news': float(os.getenv("SENTIMENT_NEWS_WEIGHT", "0.4")),
                'social': float(os.getenv("SENTIMENT_SOCIAL_WEIGHT", "0.3")),
                'technical': float(os.getenv("SENTIMENT_TECHNICAL_WEIGHT", "0.3"))
            },
            'cache_duration': int(os.getenv("SENTIMENT_CACHE_DURATION", "3600")),
            'news': {
                'max_age': int(os.getenv("NEWS_MAX_AGE", "48")),
                'min_articles': int(os.getenv("NEWS_MIN_ARTICLES", "5")),
                'source_weights': {
                    'premium': 1.0,
                    'standard': 0.7,
                    'social': 0.5
                }
            },
            'social': {
                'platforms': ['twitter', 'reddit'],
                'min_posts': int(os.getenv("SOCIAL_MIN_POSTS", "10")),
                'influence_factors': {
                    'followers': 0.4,
                    'engagement': 0.4,
                    'reputation': 0.2
                }
            },
            'technical': {
                'indicators': [
                    'rsi',
                    'macd',
                    'volume',
                    'price_momentum'
                ],
                'lookback_periods': {
                    'short': 14,
                    'medium': 30,
                    'long': 90
                }
            },
            'thresholds': {
                'extreme_positive': float(os.getenv("SENTIMENT_EXTREME_POSITIVE", "0.8")),
                'positive': float(os.getenv("SENTIMENT_POSITIVE", "0.3")),
                'neutral': float(os.getenv("SENTIMENT_NEUTRAL", "0.1")),  # 注意：原代码中neutral是0.1，通常中性是更接近0的区间
                'negative': float(os.getenv("SENTIMENT_NEGATIVE", "-0.3")),
                'extreme_negative': float(os.getenv("SENTIMENT_EXTREME_NEGATIVE", "-0.8"))
            }
        }

        # API请求限制
        self.API_RATE_LIMITS = {
            'news_api': {
                'requests_per_day': int(os.getenv("NEWS_API_DAILY_LIMIT", "1000")),
                'requests_per_second': int(os.getenv("NEWS_API_RATE_LIMIT", "1"))
            },
            'social_api': {
                'requests_per_day': int(os.getenv("SOCIAL_API_DAILY_LIMIT", "5000")),
                'requests_per_minute': int(os.getenv("SOCIAL_API_RATE_LIMIT", "30"))
            }
        }

        # 自定义策略配置 (旧的，可能需要与新的文本化策略配置方式整合或废弃)
        self.STRATEGY_CONFIG = {
            'volume_threshold': float(os.getenv("VOLUME_THRESHOLD", "2.0")),
            'price_change_threshold': float(os.getenv("PRICE_CHANGE_THRESHOLD", "0.02")),
            'min_volume_percentile': float(os.getenv("MIN_VOLUME_PERCENTILE", "80")),
            'scan_interval': int(os.getenv("SCAN_INTERVAL", "60")),  # 扫描间隔，可能被新的扫描服务取代
            'auto_trade': {  # 自动交易相关，与执行器和策略逻辑紧密相关
                'enabled': os.getenv("AUTO_TRADE_ENABLED", "False").lower() in ('true', '1', 't'),
                'max_positions': int(os.getenv("MAX_POSITIONS", "5")),
                'position_size': float(os.getenv("POSITION_SIZE", "0.1")),  # 单次交易占总资金的百分比
                'stop_loss': float(os.getenv("DEFAULT_STOP_LOSS", "0.05")),
                'take_profit': float(os.getenv("DEFAULT_TAKE_PROFIT", "0.1"))
            }
        }
        # 文本化策略配置目录
        self.TEXT_STRATEGIES_PATH = Path(os.getenv("TEXT_STRATEGIES_PATH", "strategies"))
        self.TEXT_STRATEGIES_PATH.mkdir(parents=True, exist_ok=True)

        # 情绪数据存储配置
        self.SENTIMENT_DATA_PATH = self.DATA_PATH / 'sentiment'
        self.SENTIMENT_LOG_FILE = self.DATA_PATH / 'sentiment_analysis.log'  # 路径修正

        # 市场扫描器配置
        self.SCREENER_CONFIG = {
            'enable_alpha_vantage': True,
            'enable_tushare': True,
            'default_markets': ['US', 'CN'],
            'volume_threshold': 1.5,
            'price_change_threshold': 0.05,
            'technical_filters': {
                'enable_ma': True,
                'enable_rsi': True,
                'enable_macd': False
            },
            'scan_interval_seconds': int(os.getenv("MARKET_SCAN_INTERVAL_SECONDS", "300"))  # 市场扫描服务的时间间隔
        }

    def _validate_env(self):
        """验证环境变量"""
        required_vars = {
            # 'TUSHARE_TOKEN': '用于获取A股数据', # 可选
            # 'NEWS_API_KEY': '用于获取新闻数据', # 可选
            # 'ALPHA_VANTAGE_KEY': '用于获取美股数据', # 可选
            # 'FINNHUB_KEY': '用于获取实时行情', # 可选
            # 'POLYGON_KEY': '用于获取实时行情(备用)' # 可选
        }
        # 将API Key设为可选，因为系统可能在没有它们的情况下运行（例如，仅回测或使用其他数据源）
        # 但如果配置了使用特定API的服务，而Key缺失，则应在该服务初始化时警告。

        missing_vars_messages = []
        for var, desc in required_vars.items():
            # 检查 os.getenv(VAR) 而不是 getattr(self, VAR.upper()) 因为我们还没完全赋值
            if not os.getenv(var.upper()):  # 环境变量通常大写
                missing_vars_messages.append(f"警告: 缺少推荐的环境变量 {var.upper()} ({desc}). 某些功能可能受限。")

        if missing_vars_messages:
            for msg in missing_vars_messages:
                logging.warning(msg)
            self._log_api_instructions()  # 仍然提供获取说明

    def _log_api_instructions(self):
        """输出API获取说明"""
        instructions = """
        提示：为了获得完整功能，建议配置以下API密钥（在.env文件中设置）：

        1. Tushare Token (A股数据):
           - 访问 https://tushare.pro , 注册并获取token.
           - .env 文件中: TUSHARE_TOKEN=你的token

        2. News API Key (新闻数据):
           - 访问 https://newsapi.org , 注册并获取API密钥.
           - .env 文件中: NEWS_API_KEY=你的密钥

        3. Alpha Vantage Key (美股数据):
           - 访问 https://www.alphavantage.co , 注册并获取API密钥.
           - .env 文件中: ALPHA_VANTAGE_KEY=你的密钥

        4. Finnhub Key (实时行情/备用):
           - 访问 https://finnhub.io , 注册并获取API密钥.
           - .env 文件中: FINNHUB_KEY=你的密钥

        5. Polygon Key (实时行情/备用):
           - 访问 https://polygon.io , 注册并获取API密钥.
           - .env 文件中: POLYGON_KEY=你的密钥

        如果缺少这些密钥，系统将尝试使用备用数据源或限制相关功能。
        """
        logging.info(instructions)

    def create_directories(self):
        """创建必要的目录"""
        directories = [
            self.MODEL_PATH,  # 旧的，可能仍用于某些通用模型保存
            self.ML_MODELS_BASE_PATH,  # 新的，用于MLStrategy管理模型
            self.DATA_PATH,
            self.SENTIMENT_DATA_PATH,
            self.TEXT_STRATEGIES_PATH,  # 用于存放文本化策略文件
            self.DATA_PATH / "logs"  # 日志文件目录
        ]
        try:
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)

            # 修正日志文件路径为绝对路径或相对于项目根目录的已知路径
            log_file_dir = self.DATA_PATH / "logs"
            self.LOG_FILE = log_file_dir / 'trading_system.log'
            self.SENTIMENT_LOG_FILE = log_file_dir / 'sentiment_analysis.log'

            # 重新配置日志处理器以使用新路径 (如果日志已初始化)
            # 这部分比较 tricky，因为basicConfig通常只应调用一次。
            # 更好的做法是在 __init__ 最开始确定好日志文件路径。
            # 这里假设 _init_logging 会在 create_directories 之后再次被某种机制刷新或日志句柄会更新
            # 或者，在_init_logging中就使用最终确定的路径
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:  # Iterate over a copy
                if isinstance(handler, logging.FileHandler):
                    root_logger.removeHandler(handler)

            logging.basicConfig(  # Reconfigure with correct path
                level=self.LOG_LEVEL,
                format=self.LOG_FORMAT,
                handlers=[
                    logging.FileHandler(self.LOG_FILE, encoding='utf-8'),
                    logging.StreamHandler()
                ]
            )
            logging.info(f"日志将输出到: {self.LOG_FILE}")

        except Exception as e:
            logging.error(f"创建目录时出错: {e}", exc_info=True)

    @property
    def is_development(self) -> bool:
        """是否为开发环境"""
        return os.getenv('ENVIRONMENT', 'development').lower() == 'development'

    @property
    def is_production(self) -> bool:
        """是否为生产环境"""
        return os.getenv('ENVIRONMENT', 'development').lower() == 'production'

    def validate_trading_params(self) -> bool:
        """验证交易参数"""
        try:
            assert 0 < self.MAX_POSITION_SIZE <= 1, "持仓比例必须在0-1之间"
            assert 0 < self.MAX_DRAWDOWN <= 1, "最大回撤必须在0-1之间"
            assert self.RISK_FREE_RATE >= 0, "无风险利率必须大于等于0"
            assert self.COMMISSION_RATE >= 0, "佣金率必须大于等于0"
            assert self.MIN_TRADE_AMOUNT > 0, "最小交易金额必须大于0"
            return True
        except AssertionError as e:
            logging.error(f"交易参数验证失败: {str(e)}")
            return False

    def validate_sentiment_params(self) -> bool:
        """验证情绪分析参数"""
        try:
            weights = self.SENTIMENT_CONFIG['weights']
            # 允许权重和不为1，如果它们代表不同维度的独立贡献因子
            # assert abs(sum(weights.values()) - 1.0) < 1e-6, "情绪权重总和必须接近1"
            assert all(0 <= w <= 1 for w in weights.values()), "情绪分析各项权重必须在0-1之间"

            thresholds = self.SENTIMENT_CONFIG['thresholds']
            # 确保阈值顺序正确
            sorted_threshold_keys = ['extreme_negative', 'negative', 'neutral', 'positive', 'extreme_positive']
            threshold_values = [thresholds[k] for k in sorted_threshold_keys]
            assert all(threshold_values[i] < threshold_values[i + 1] for i in range(len(threshold_values) - 1)), \
                "情绪阈值必须严格递增 (extreme_negative < negative < neutral < positive < extreme_positive)"

            return True
        except AssertionError as e:
            logging.error(f"情绪分析参数验证失败: {str(e)}")
            return False
        except KeyError as e:
            logging.error(f"情绪分析参数配置错误: 缺失键 {str(e)}")
            return False

    def __str__(self) -> str:
        """返回配置信息的字符串表示"""
        env_type = 'Production' if self.is_production else 'Development'
        return f"Trading System Config (Environment: {env_type}, Log: {self.LOG_FILE})"

    # update_strategy_settings 和 _save_strategy_settings 可能需要重构
    # 以适应新的文本化策略加载机制，或者仅用于UI上可调整的全局参数。
    # 当前的实现是基于旧的 self.STRATEGY_CONFIG 结构。
    def update_strategy_settings(self, strategy_name: str, settings: Dict):
        """
        更新指定策略的设置 (主要用于UI可调参数，或与文本策略文件关联的参数)
        """
        # 这个方法需要根据最终如何管理策略来决定其具体实现。
        # 如果策略主要通过文件加载，这个方法可能用于覆盖文件中的某些参数，
        # 或者管理一些不适合放在策略文件中的运行时参数。
        try:
            if strategy_name not in self.STRATEGY_CONFIG:  # 旧的 STRATEGY_CONFIG
                self.STRATEGY_CONFIG[strategy_name] = {}
            self.STRATEGY_CONFIG[strategy_name].update(settings)
            self._save_global_settings()  # 可能需要一个保存全局可调参数的方法
            logging.info(f"策略 '{strategy_name}' 的设置已更新。")
            return True
        except Exception as e:
            logging.error(f"更新策略 '{strategy_name}' 设置时出错: {e}")
            return False

    def _save_global_settings(self):
        """保存可调整的全局配置到文件 (例如 settings/global_runtime_settings.json)"""
        try:
            settings_dir = Path('settings')
            settings_dir.mkdir(exist_ok=True)
            import json
            # 将那些适合运行时调整的配置保存起来
            runtime_configurable_settings = {
                "MAX_POSITION_SIZE": self.MAX_POSITION_SIZE,
                "MAX_DRAWDOWN": self.MAX_DRAWDOWN,
                "AUTO_TRADE_ENABLED": self.STRATEGY_CONFIG.get('auto_trade', {}).get('enabled'),
                # 添加其他希望用户可以通过UI修改并持久化的参数
            }
            settings_file = settings_dir / 'global_runtime_settings.json'
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(runtime_configurable_settings, f, indent=4)
            logging.info(f"全局运行时设置已保存到 {settings_file}")
        except Exception as e:
            logging.error(f"保存全局运行时设置失败: {e}")

    def get_screener_config(self, market: str) -> Dict:
        """获取指定市场的扫描器配置"""
        base_config = self.SCREENER_CONFIG.copy()
        if market == 'CN':
            base_config.update({
                'symbol_suffix': '.SH',  # 或者 .SZ，需要更复杂的逻辑来区分
                'volume_field': 'vol',
                'price_adjust': 'hfq'
            })
        elif market == 'US':
            base_config.update({
                'symbol_suffix': '',
                'volume_field': 'volume',
                'price_adjust': None
            })
        else:  # 其他市场默认配置
            base_config.update({
                'symbol_suffix': '',
                'volume_field': 'volume',
                'price_adjust': None
            })
        return base_config

    def get(self, key: str, default=None):
        """
        获取配置项，类似字典的 get 方法。
        Args:
            key (str): 配置项的名称 (通常是实例属性名的大写形式)。
            default: 如果配置项不存在时返回的默认值。
        Returns:
            配置项的值或默认值。
        """
        # 尝试直接获取大写属性名
        value = getattr(self, key.upper(), None)  # 假设配置属性都是大写的
        if value is not None:
            return value

        # 如果大写属性不存在，尝试获取原始大小写的属性名
        value = getattr(self, key, None)
        if value is not None:
            return value

        # 如果都没有找到，则记录一个警告并返回默认值
        # logging.warning(f"配置项 '{key}' 未在Config对象中找到，将使用默认值: {default}")
        return default

    # render_settings_ui - 这个方法通常在UIManager中或者一个专门的SettingsUI模块中实现
    # Config类本身主要负责加载和提供配置，而不是渲染UI。
    # 但如果保持在这里，它需要 streamlit (st) 的导入。
    def render_settings_ui(self):
        """在Streamlit中渲染配置设置界面 (示例)"""
        try:
            import streamlit as st  # 局部导入，避免Config类强依赖streamlit
            st.header("系统配置")

            st.subheader("交易参数")
            self.MAX_POSITION_SIZE = st.slider("最大单笔仓位比例", 0.01, 0.5, self.MAX_POSITION_SIZE, 0.01)
            self.MAX_DRAWDOWN = st.slider("允许的最大回撤", 0.05, 0.5, self.MAX_DRAWDOWN, 0.01)
            # self.STRATEGY_CONFIG['auto_trade']['enabled'] = st.checkbox(
            #     "启用自动交易",
            #     self.STRATEGY_CONFIG.get('auto_trade', {}).get('enabled', False)
            # )

            # 确保 auto_trade 和 enabled 存在
            if 'auto_trade' not in self.STRATEGY_CONFIG:
                self.STRATEGY_CONFIG['auto_trade'] = {'enabled': False}
            elif 'enabled' not in self.STRATEGY_CONFIG['auto_trade']:
                self.STRATEGY_CONFIG['auto_trade']['enabled'] = False

            auto_trade_enabled_current = self.STRATEGY_CONFIG['auto_trade']['enabled']
            auto_trade_enabled_new = st.checkbox("启用自动交易", value=auto_trade_enabled_current)
            if auto_trade_enabled_new != auto_trade_enabled_current:
                self.STRATEGY_CONFIG['auto_trade']['enabled'] = auto_trade_enabled_new

            st.subheader("API密钥 (仅显示状态，请在.env文件中修改)")
            apis_to_check = {
                "TUSHARE_TOKEN": self.TUSHARE_TOKEN,
                "NEWS_API_KEY": self.NEWS_API_KEY,
                "ALPHA_VANTAGE_KEY": self.ALPHA_VANTAGE_KEY,
                "FINNHUB_KEY": self.FINNHUB_KEY,
                "POLYGON_KEY": self.POLYGON_KEY
            }
            for name, key_value in apis_to_check.items():
                status = "已配置" if key_value else "未配置"
                st.text(f"{name}: {status}")

            if st.button("保存配置更改"):
                self._save_global_settings()  # 保存可调整的参数
                st.success("配置已保存！部分设置可能需要重启生效。")
                # 注意：修改 .env 文件中的API密钥需要手动操作并重启应用。

        except ImportError:
            logging.error("Streamlit未安装，无法渲染设置UI。Config类应主要用于配置管理。")
        except Exception as e:
            logging.error(f"渲染设置UI时出错: {e}", exc_info=True)
            st.error(f"渲染设置UI时出错: {e}")