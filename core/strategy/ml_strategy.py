# core/strategy/ml_strategy.py
import pandas as pd
import numpy as np
import joblib
import logging
from typing import Optional, Dict, List, Any
from pathlib import Path
from datetime import datetime, timedelta
import streamlit as st

try:
    from .dl_models import (
        LSTMModelHandler, TransformerModelHandler, AlphaTransformerModelHandler,
        TENSORFLOW_AVAILABLE, PositionalEncoding, TransformerEncoderBlock
    )

    if TENSORFLOW_AVAILABLE:
        from tensorflow.keras.models import load_model

        # 更新自定义对象字典
        CUSTOM_OBJECTS = {
            'PositionalEncoding': PositionalEncoding,
            'TransformerEncoderBlock': TransformerEncoderBlock,
        }
except ImportError:
    CUSTOM_OBJECTS = {}
# --- 导入依赖，只进行一次 ---
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.utils.validation import check_is_fitted

    SKLEARN_AVAILABLE = True

except ImportError:
    class RandomForestClassifier:
        pass


    class StandardScaler:
        pass


    def train_test_split(*args, **kwargs):
        pass


    def accuracy_score(*args, **kwargs):
        return 0.0


    def check_is_fitted(estimator, attributes=None):
        raise ImportError("scikit-learn is not installed.")


    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn 未安装，传统机器学习功能将受限。")

try:
    from .dl_models import LSTMModelHandler, TransformerModelHandler, AlphaTransformerModelHandler, TENSORFLOW_AVAILABLE

    if TENSORFLOW_AVAILABLE:
        from tensorflow.keras.models import load_model
except ImportError:
    class LSTMModelHandler:
        pass


    class TransformerModelHandler:
        pass


    class AlphaTransformerModelHandler:
        pass


    TENSORFLOW_AVAILABLE = False


    def load_model(filepath):
        pass


    logging.warning("dl_models.py 或 TensorFlow 未找到，深度学习功能将不可用。")

try:
    from core.analysis.text_feature_extractor import TextFeatureExtractor

    TEXT_EXTRACTOR_AVAILABLE = True
except ImportError:
    class TextFeatureExtractor:
        pass


    TEXT_EXTRACTOR_AVAILABLE = False
    logging.warning("TextFeatureExtractor 未找到，文本特征功能将不可用。")

logger = logging.getLogger(__name__)


# --- 缓存函数 ---
# 临时修改：直接注释掉缓存装饰器
# @st.cache_resource(show_spinner="加载机器学习模型资源...", ttl=3600)
def load_ml_resource(file_path: Path) -> Optional[Any]:
    if not file_path.exists():
        logger.warning(f"资源文件未找到: {file_path}")
        return None

    if file_path.suffix in ['.h5', '.keras']:
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow/Keras 不可用。")
            return None
        try:
            if CUSTOM_OBJECTS:
                return load_model(file_path, custom_objects=CUSTOM_OBJECTS)
            else:
                return load_model(file_path)
        except Exception as e:
            logger.error(f"加载 Keras 模型失败 {file_path}: {e}", exc_info=True)
            return None

    elif file_path.suffix == '.joblib':
        try:
            import joblib
            obj = joblib.load(file_path)
            logger.info(f"成功加载 joblib 文件: {file_path}")
            return obj
        except Exception as e:
            logger.error(f"加载 joblib 文件失败 {file_path}: {e}", exc_info=True)
            return None

    else:
        logger.warning(f"不支持的文件类型: {file_path.suffix}")
        return None


def clear_load_ml_resource_cache():
    try:
        load_ml_resource.clear()
    except Exception as e:
        logger.warning(f"Failed to clear ml_resource cache: {e}")


class SklearnHandler:
    def __init__(self): self.model, self.scaler = None, None


class MLStrategy:
    def __init__(self, config: Dict, data_manager_ref: Optional[Any] = None):
        self.is_available = SKLEARN_AVAILABLE or TENSORFLOW_AVAILABLE
        if not self.is_available: logger.critical("CRITICAL: No ML framework available.")

        self.config = config
        self.data_manager = data_manager_ref
        self.active_model_handler: Optional[Any] = None
        self.current_model_name: Optional[str] = None
        self.current_model_filename: Optional[str] = None
        self.logger = logging.getLogger(__name__)
        self.models_base_path = getattr(config, 'ML_MODELS_BASE_PATH', Path("models"))

        self.base_feature_columns = list(getattr(config, 'BASE_FEATURE_COLUMNS', []))
        self.text_feature_columns = list(getattr(config, 'TEXT_FEATURE_COLUMNS', []))
        self.required_features = self.base_feature_columns[:]

        self.text_feature_extractor = None
        self.logger.info("Initializing TextFeatureExtractor within MLStrategy...")
        if TEXT_EXTRACTOR_AVAILABLE and self.data_manager:
            try:
                self.text_feature_extractor = TextFeatureExtractor(config, self.data_manager)
                if self.text_feature_extractor.is_available:
                    self.logger.info("TextFeatureExtractor initialized successfully and IS AVAILABLE.")
                else:
                    self.logger.warning(
                        "TextFeatureExtractor initialized, but IS NOT AVAILABLE. Check Gemini Key/Library.")
            except Exception as e:
                logger.error(f"Failed to initialize TextFeatureExtractor: {e}")
        else:
            # 添加更详细的日志
            self.logger.warning(
                f"Skipping TextFeatureExtractor init. TEXT_EXTRACTOR_AVAILABLE={TEXT_EXTRACTOR_AVAILABLE}, self.data_manager is not None={self.data_manager is not None}")

        if self.is_available:
            default_model_name = getattr(config, 'DEFAULT_ML_MODEL_NAME', None)
            if default_model_name and default_model_name != "无可用模型":
                self.set_active_model(default_model_name)



    def _get_model_type(self, model_filename: str) -> str:
        if model_filename.endswith(('.h5', '.keras')):
            if 'transformer' in model_filename.lower(): return 'AlphaTransformer'
            return 'LSTM'
        elif model_filename.endswith('.joblib'):
            return 'RandomForest'
        return 'Unknown'

    def set_active_model(self, model_display_name: str) -> bool:
        if not self.is_available: return False
        self.logger.info(f"Attempting to set active model to: {model_display_name}")
        self.active_model_handler, self.current_model_name, self.current_model_filename = None, None, None
        available_models = getattr(self.config, 'AVAILABLE_ML_MODELS', {})
        if model_display_name not in available_models:
            self.logger.error(f"Model '{model_display_name}' not defined in config.");
            return False

        model_filename = available_models[model_display_name]
        model_type = self._get_model_type(model_filename)
        model_path = self.models_base_path / model_filename
        self.current_model_name, self.current_model_filename = model_display_name, model_filename

        if model_type in ['LSTM', 'AlphaTransformer']:
            if not TENSORFLOW_AVAILABLE: logger.error("TensorFlow not installed."); return False
            HandlerClass = LSTMModelHandler if model_type == 'LSTM' else AlphaTransformerModelHandler
            self.active_model_handler = HandlerClass(self.config)
            self.active_model_handler.model = load_ml_resource(model_path)
            if self.active_model_handler.model:
                logger.info(f"{model_type} model '{model_display_name}' loaded.")
            else:
                logger.warning(f"{model_type} model file '{model_filename}' not found. Needs training.")
        elif model_type == 'RandomForest':
            if not SKLEARN_AVAILABLE: logger.error("scikit-learn not installed."); return False
            self.active_model_handler = SklearnHandler()
            scaler_path = self.models_base_path / f"{Path(model_filename).stem}_scaler.joblib"
            self.active_model_handler.model = load_ml_resource(model_path)
            self.active_model_handler.scaler = load_ml_resource(scaler_path)
            if not self.active_model_handler.scaler: self.active_model_handler.scaler = StandardScaler()
        else:
            logger.error(f"Unknown model type '{model_type}' for file '{model_filename}'");
            return False
        return True

    def train(self, data: pd.DataFrame, symbol: str, model_display_name_to_save: Optional[str] = None) -> Dict:
        """
        [数据流修复版] 训练模型，确保与预测/回测使用相同的数据处理流程。
        """
        if not self.active_model_handler:
            if not model_display_name_to_save or not self.set_active_model(model_display_name_to_save):
                return {'success': False, 'message': f"无法设置或初始化模型 '{model_display_name_to_save}'。"}

        model_type = self._get_model_type(self.current_model_filename)

        logger.info(f"开始训练 {model_type} 模型: {model_display_name_to_save}")
        logger.info(f"原始训练数据形状: {data.shape}")

        try:
            # 1. 使用相同的特征准备流程
            features_df = self.prepare_features(data, symbol=symbol)
            if features_df.empty:
                return {'success': False, 'message': "特征准备失败"}

            logger.info(f"特征准备后数据形状: {features_df.shape}")

            # 2. 调用对应的训练方法
            if model_type in ['LSTM', 'AlphaTransformer']:
                return self._train_dl(features_df, symbol, model_type)
            elif model_type == 'RandomForest':
                return self._train_sklearn(features_df, symbol)
            else:
                return {'success': False, 'message': f"未知的模型类型 '{model_type}'。"}

        except Exception as e:
            logger.error(f"训练过程失败: {e}", exc_info=True)
            return {'success': False, 'message': f"训练失败: {str(e)}"}

    def predict(self, data: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """
        [重构] 预测。调度到具体的预测方法。
        """
        if not self.active_model_handler or getattr(self.active_model_handler, 'model', None) is None:
            return {'message': f"模型 '{self.current_model_name or '未知'}' 未加载，无法预测。"}

        model_type = self._get_model_type(self.current_model_filename)

        if model_type in ['LSTM', 'AlphaTransformer']:
            return self._predict_dl(data, symbol, model_type)
        elif model_type == 'RandomForest':
            return self._predict_sklearn(data, symbol)
        else:
            return {'message': f"未知的模型类型 '{model_type}'。"}

    def _train_dl(self, data: pd.DataFrame, symbol: str, model_type: str) -> Dict:
        """[最终修复版] 训练深度学习模型，采用更健壮的 NaN 处理和数据流。"""
        handler = self.active_model_handler
        if not handler: return {'success': False, 'message': f"DL 处理器未准备好用于 {model_type}。"}
        params = self.config.ML_HYPERPARAMETERS.get(model_type, {})
        lookback = params.get('lookback', 60)
        self.logger.info(f"开始训练 {model_type} 模型: {self.current_model_name}...")
        try:
            # 1. 准备所有特征 (现在会返回一个带有大量 NaN 的 DataFrame)
            features_df_full = self.prepare_features(data, symbol=symbol)
            if features_df_full.empty: return {'success': False, 'message': "特征准备返回了空的数据框。"}

            # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
            # 2. 生成目标列
            if 'close' not in features_df_full.columns:
                return {'success': False, 'message': "prepare_features 未能返回 'close' 列。"}
            if model_type == 'AlphaTransformer':
                features_df_full['target'] = features_df_full['close'].pct_change(periods=5).shift(-5)
            else:
                features_df_full['target'] = (features_df_full['close'].shift(-1) > features_df_full['close']).astype(
                    int)

            # 3. 替换无穷大值
            features_df_full.replace([np.inf, -np.inf], np.nan, inplace=True)

            # 4. 执行一次性的、最终的 dropna
            # 这会移除所有因为特征计算（开头）和目标计算（末尾）产生的无效行
            features_df_clean = features_df_full.dropna()
            # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^

            self.logger.info(f"Data for training, after NaN cleanup: {features_df_clean.shape[0]} rows.")
            if len(features_df_clean) < lookback + 50:  # 确保有足够数据用于 lookback 和 train/val split
                return {'success': False, 'message': f"清理后数据严重不足({len(features_df_clean)})，无法训练。"}

                # 5. 从干净的 DataFrame 中准备 NumPy 数组
            training_feature_cols = [col for col in self.required_features if col in features_df_clean.columns]
            X_unscaled = features_df_clean[training_feature_cols].values.astype(np.float32)
            y = features_df_clean['target'].values.astype(np.float32)

            # 6. 训练/验证分割 (在 NumPy 数组上进行)
            split_idx = int(len(X_unscaled) * 0.8)
            X_train_unscaled, X_val_unscaled = X_unscaled[:split_idx], X_unscaled[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # 7. 训练并应用 Scaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_unscaled)
            X_val_scaled = scaler.transform(X_val_unscaled)

            # 8. 将归一化后的 NumPy 数组重塑为 DL 格式
            X_train_final, y_train_final = handler.prepare_data(X_train_scaled, y_train, lookback)
            X_val_final, y_val_final = handler.prepare_data(X_val_scaled, y_val, lookback)

            if len(X_train_final) == 0:
                return {'success': False, 'message': "重塑后训练样本为空，请检查 lookback 设置。"}

            # 9. 构建和训练模型
            if handler.model is None:
                handler.build_model(input_shape=(X_train_final.shape[1], X_train_final.shape[2]))

            history = handler.train(X_train_final, y_train_final, X_val_final, y_val_final)

            # 10. 保存模型和 Scaler
            model_path = self.models_base_path / self.current_model_filename
            scaler_path = self.models_base_path / f"{Path(self.current_model_filename).stem}_scaler.joblib"
            handler.model.save(model_path)
            if SKLEARN_AVAILABLE:  # 保存 scaler 的 feature names
                try:
                    scaler.feature_names_in_ = training_feature_cols
                except Exception:
                    pass
            joblib.dump(scaler, scaler_path)
            clear_load_ml_resource_cache()

            # 11. 返回结果
            metric_key = 'val_accuracy' if model_type != 'AlphaTransformer' else 'val_mean_absolute_error'
            final_val_metric = history.get(metric_key, [0])[-1]
            return {'success': True, 'message': f"{model_type} 模型 '{self.current_model_name}' 训练完成。",
                    'validation_metric': final_val_metric, 'n_samples': len(X_train_final), 'history': history}

        except Exception as e:
            self.logger.error(f"训练 {model_type} 模型时出错: {e}", exc_info=True)
            return {'success': False, 'message': f"训练 {model_type} 出错: {e}"}

    def _reshape_for_dl(self, data: np.ndarray, lookback: int) -> np.ndarray:
        """[新增] 将 2D NumPy 数组重塑为 DL 模型需要的 3D (samples, timesteps, features) 格式"""
        if len(data) < lookback:
            return np.array([])  # 返回空数组

        shape = (data.shape[0] - lookback + 1, lookback, data.shape[1])
        strides = (data.strides[0], data.strides[0], data.strides[1])
        return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    def _train_sklearn(self, data: pd.DataFrame, symbol: str) -> Dict:
        """[私有] 训练 RandomForest 模型 (修复了 NaN 处理)"""
        handler = self.active_model_handler
        self.logger.info(f"开始训练 RandomForest 模型: {self.current_model_name}...")
        try:
            if handler.model is None:
                rf_params = self.config.ML_HYPERPARAMETERS.get('RandomForestClassifier', {})
                handler.model = RandomForestClassifier(**rf_params)
            if handler.scaler is None: handler.scaler = StandardScaler()

            features_df_full = self.prepare_features(data, symbol=symbol)
            if features_df_full.empty:
                return {'success': False, 'message': "特征准备返回了空的数据框。"}

            # 2. 生成标签
            labels = (features_df_full['close'].shift(-1) > features_df_full['close']).astype(int)
            features_df_full['labels'] = labels

            # 3. 清理 NaN
            features_df_clean = features_df_full.dropna()
            # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^

            self.logger.info(f"Sklearn: Cleaned data for training: {features_df_clean.shape[0]} rows.")
            if features_df_clean.empty or len(features_df_clean) < 65:
                return {'success': False, 'message': f"特征准备失败或数据不足 ({len(features_df_clean)}行)。"}

                # 4. 准备 NumPy 数组
            features_values = features_df_clean[self.required_features].values
            labels_values = features_df_clean['labels'].values

            split_idx = int(len(features_values) * 0.8)
            X_train, X_test = features_values[:split_idx], features_values[split_idx:]
            y_train, y_test = labels_values[:split_idx], labels_values[split_idx:]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                return {'success': False, 'message': "训练集或测试集中只包含单一类别。"}

            handler.scaler.fit(X_train)
            X_train_scaled = handler.scaler.transform(X_train)
            X_test_scaled = handler.scaler.transform(X_test)
            handler.model.fit(X_train_scaled, y_train)

            train_score = accuracy_score(y_train, handler.model.predict(X_train_scaled))
            test_score = accuracy_score(y_test, handler.model.predict(X_test_scaled))

            self._save_model()

            return {'success': True, 'train_score': train_score, 'test_score': test_score,
                    'n_samples': len(features_values),
                    'message': f"RandomForest 模型 '{self.current_model_name}' 训练完成并保存。"}
        except Exception as e:
            self.logger.error(f"训练 RandomForest 模型时出错: {e}", exc_info=True)
            return {'success': False, 'message': f"训练出错: {e}"}


    def _predict_sklearn(self, data: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """[修复版] 为 Scikit-learn 模型执行预测。"""
        handler = self.active_model_handler
        if handler.scaler is None or not hasattr(handler.scaler, "mean_"):
            return {'message': "Scaler 未加载或未拟合。"}
        try:
            features_df = self.prepare_features(data, symbol=symbol)
            if features_df.empty: return {'message': "特征准备失败。"}

            # --- 关键修复：直接取最后一行，然后检查 ---
            latest_features_row = features_df.iloc[-1:]
            if latest_features_row[self.required_features].isnull().values.any():
                return {'message': "最新的特征数据包含无效值(NaN)，无法预测。"}

            latest_features = latest_features_row[self.required_features].values

            features_scaled = handler.scaler.transform(latest_features)
            prediction = handler.model.predict(features_scaled)[0]
            probabilities = handler.model.predict_proba(features_scaled)[0]

            feature_importance = {}
            if hasattr(handler.model, 'feature_importances_'):
                feature_importance = dict(zip(self.required_features, handler.model.feature_importances_))

            return {
                'direction': int(prediction), 'confidence': float(max(probabilities)),
                'probability_up': float(probabilities[1]), 'probability_down': float(probabilities[0]),
                'feature_importance': feature_importance, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_used': self.current_model_name
            }
        except Exception as e:
            self.logger.error(f"RandomForest 预测时出错: {e}", exc_info=True)
            return {'message': f"RandomForest 预测出错: {e}"}

    def _predict_dl(self, data: pd.DataFrame, symbol: str, model_type: str) -> Optional[Dict]:
        """[最终修复版] 预测深度学习模型。"""
        try:
            handler = self.active_model_handler
            # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
            # --- 在方法开头就获取所有需要的超参数 ---
            params = self.config.ML_HYPERPARAMETERS.get(model_type, {})
            lookback = params.get('lookback', 60)  # 60 是默认值

            # 1. 加载训练时使用的 Scaler
            scaler_path = self.models_base_path / f"{Path(self.current_model_filename).stem}_scaler.joblib"
            scaler = load_ml_resource(scaler_path)
            if not scaler or not hasattr(scaler, "n_features_in_"):
                return {'message': "找不到或未训练用于预测的 Scaler 文件。"}

            # 2. 准备预测所需的特征 (现在 prepare_features 总是返回固定结构的 df)
            features_df = self.prepare_features(data, symbol=symbol)
            if len(features_df) < lookback:
                return {'message': f"准备预测数据后行数不足({len(features_df)})"}

            # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
            # 3. 使用 scaler 中保存的特征名来确保一致性
            training_feature_cols = getattr(scaler, 'feature_names_in_', None)
            if training_feature_cols is None:
                # 如果旧的 scaler 没有保存特征名，我们进行回退
                self.logger.warning("Scaler does not contain feature names. Falling back to config.")
                training_feature_cols = self.base_feature_columns + self.text_feature_columns

            missing_cols = set(training_feature_cols) - set(features_df.columns)
            if missing_cols:
                return {'message': f"预测数据中缺少列: {', '.join(missing_cols)}"}

            # 4. 只选择 scaler 期望的列，并取最后 lookback 行
            latest_data_to_scale = features_df[training_feature_cols].iloc[-lookback:]
            # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^

            # 5. 应用 Scaler 并准备输入
            scaled_features_array = scaler.transform(latest_data_to_scale).astype(np.float32)
            X_pred = np.reshape(scaled_features_array, (1, lookback, scaled_features_array.shape[1]))

            if model_type == 'AlphaTransformer':
                predicted_alpha = handler.model.predict(X_pred, verbose=0)[0][0]
                # --- 关键修复：让模型总是给出方向 ---
                direction = 1 if predicted_alpha > 0 else 0
                # --- 结束修复 ---
                confidence = min(1.0, abs(predicted_alpha) / 0.05)
                return {'direction': direction, 'predicted_alpha': float(predicted_alpha),
                        'confidence': float(confidence), 'model_used': self.current_model_name}
            else:
                probability_up = handler.model.predict(X_pred, verbose=0)[0][0]
                prediction = 1 if probability_up > 0.5 else 0
                return {'direction': prediction, 'probability_up': float(probability_up),
                        'confidence': abs(probability_up - 0.5) * 2, 'model_used': self.current_model_name}
        except Exception as e:
            self.logger.error(f"{model_type} 预测时出错: {e}", exc_info=True)
            return {'message': f"{model_type} 预测出错: {e}"}

    def predict_for_backtest(self, features_df: pd.DataFrame, symbol: str) -> Optional[pd.Series]:
        """
        [最终修复版] 为回测创建独立的模型预测，确保数据流畅通。
        """
        if not self.active_model_handler or not getattr(self.active_model_handler, 'model', None):
            self.logger.error("模型未加载，无法进行批量预测")
            return None

        self.logger.info(f"开始批量预测 for {symbol} ({len(features_df)} 行数据)...")

        try:
            model_type = self._get_model_type(self.current_model_filename)
            self.logger.info(f"使用模型类型: {model_type}")

            if model_type in ['LSTM', 'AlphaTransformer']:
                return self._predict_dl_batch(features_df, symbol, model_type)
            elif model_type == 'RandomForest':
                return self._predict_sklearn_batch(features_df, symbol)
            else:
                self.logger.error(f"不支持的模型类型: {model_type}")
                return None

        except Exception as e:
            self.logger.error(f"批量预测失败: {e}", exc_info=True)
            return None

    def _predict_dl_batch(self, features_df: pd.DataFrame, symbol: str, model_type: str) -> Optional[pd.Series]:
        """深度学习模型的批量预测"""
        params = self.config.ML_HYPERPARAMETERS.get(model_type, {})
        lookback = params.get('lookback', 60)

        # 1. 加载 scaler
        scaler_path = self.models_base_path / f"{Path(self.current_model_filename).stem}_scaler.joblib"
        scaler = load_ml_resource(scaler_path)
        if not scaler:
            self.logger.error(f"无法加载 scaler: {scaler_path}")
            return None

        # 2. 获取训练时的特征列
        training_feature_cols = getattr(scaler, 'feature_names_in_', None)
        if training_feature_cols is None:
            # 修复：定义完整的特征列表
            technical_features = [
                'ma5', 'ma10', 'ma20', 'ma30', 'ma60',
                'vol_20', 'vol_60', 'vol_chg_5',
                'mom_20', 'mom_60',
                'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
                'signal', 'hist'
            ]
            training_feature_cols = (
                    self.base_feature_columns +
                    technical_features +
                    self.text_feature_columns
            )
            self.logger.warning("Scaler中没有feature_names_in_，使用完整特征列配置")

        self.logger.info(f"📋 需要的特征列数量: {len(training_feature_cols)}")

        # 3. 检查并计算缺失的技术指标
        missing_tech_features = []
        for col in training_feature_cols:
            if col not in features_df.columns and col in [
                'ma5', 'ma10', 'ma20', 'ma30', 'ma60',
                'vol_20', 'vol_60', 'vol_chg_5',
                'mom_20', 'mom_60',
                'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
                'signal', 'hist'
            ]:
                missing_tech_features.append(col)

        if missing_tech_features:
            self.logger.info(f"🔧 需要计算缺失的技术指标: {missing_tech_features}")
            try:
                features_df = self._add_missing_technical_features(features_df, missing_tech_features)
            except Exception as e:
                self.logger.error(f"计算技术指标失败: {e}")
                return None

        # 4. 验证特征列
        missing_cols = set(training_feature_cols) - set(features_df.columns)
        if missing_cols:
            self.logger.error(f"❌ 仍然缺少特征列: {missing_cols}")
            return None

        # 5. 准备数据
        features_clean = features_df[training_feature_cols].copy()

        # 处理无效值
        features_clean = features_clean.replace([np.inf, -np.inf], np.nan)
        features_clean = features_clean.ffill().bfill().fillna(0.0)

        X_unscaled = features_clean.values.astype(np.float32)

        if len(X_unscaled) < lookback:
            self.logger.error(f"数据长度不足: {len(X_unscaled)} < {lookback}")
            return None

        # 6. 数据缩放
        X_scaled = scaler.transform(X_unscaled)

        # 7. 创建序列数据
        X_sequences = []
        valid_indices = []

        for i in range(lookback - 1, len(X_scaled)):
            X_sequences.append(X_scaled[i - lookback + 1:i + 1])
            valid_indices.append(features_df.index[i])

        if not X_sequences:
            self.logger.error("无法创建有效的序列数据")
            return None

        X_sequences = np.array(X_sequences)
        self.logger.info(f"序列数据形状: {X_sequences.shape}")

        # 8. 批量预测
        try:
            batch_size = min(32, len(X_sequences))
            predictions = []

            num_batches = (len(X_sequences) + batch_size - 1) // batch_size

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_sequences))
                batch_data = X_sequences[start_idx:end_idx]

                batch_pred = self.active_model_handler.model.predict(
                    batch_data, batch_size=len(batch_data), verbose=0
                )
                predictions.extend(batch_pred.flatten())

            predictions = np.array(predictions)

        except Exception as e:
            self.logger.error(f"批量预测执行失败: {e}")
            return None

        # 9. 转换为 alpha 分数
        if model_type == 'AlphaTransformer':
            alpha_scores = predictions
        else:
            alpha_scores = predictions - 0.5

        # 10. 创建结果 Series
        result_series = pd.Series(alpha_scores, index=valid_indices, name='alpha_score')

        self.logger.info(f"批量预测完成，返回 {len(result_series)} 个预测值")
        return result_series

    def _add_missing_technical_features(self, df: pd.DataFrame, missing_features: list) -> pd.DataFrame:
        """为缺失的技术指标特征添加计算"""

        result_df = df.copy()

        # 需要价格数据列
        required_cols = ['close', 'high', 'low', 'volume']
        missing_price_cols = [col for col in required_cols if col not in df.columns]
        if missing_price_cols:
            raise ValueError(f"缺少计算技术指标所需的价格数据: {missing_price_cols}")

        for feature in missing_features:
            try:
                if feature.startswith('ma'):
                    # 移动平均线
                    period = int(feature[2:])  # ma20 -> 20
                    result_df[feature] = df['close'].rolling(window=period).mean()

                elif feature.startswith('vol_'):
                    # 波动率
                    if feature == 'vol_chg_5':
                        result_df[feature] = df['volume'].pct_change(5)
                    else:
                        period = int(feature[4:])  # vol_60 -> 60
                        result_df[feature] = df['close'].rolling(window=period).std()

                elif feature.startswith('mom_'):
                    # 动量指标
                    period = int(feature[4:])  # mom_60 -> 60
                    result_df[feature] = df['close'].pct_change(period)

                elif feature.startswith('bollinger_'):
                    # 布林带
                    ma20 = df['close'].rolling(window=20).mean()
                    std20 = df['close'].rolling(window=20).std()
                    if feature == 'bollinger_upper':
                        result_df[feature] = ma20 + 2 * std20
                    elif feature == 'bollinger_lower':
                        result_df[feature] = ma20 - 2 * std20
                    elif feature == 'bollinger_middle':
                        result_df[feature] = ma20

                elif feature == 'signal':
                    # 信号特征（简单实现）
                    result_df[feature] = (df['close'] > df['close'].rolling(5).mean()).astype(int)

                elif feature == 'hist':
                    # 历史特征（可以是任何历史统计）
                    result_df[feature] = df['close'].rolling(window=20).apply(lambda x: len(x))

            except Exception as e:
                self.logger.warning(f"计算特征 {feature} 失败: {e}")
                # 填充默认值
                result_df[feature] = 0.0

        # 填充 NaN 值
        result_df = result_df.ffill().bfill().fillna(0.0)

        self.logger.info(f"✅ 成功添加 {len(missing_features)} 个技术指标特征")
        return result_df

    def _predict_sklearn_batch(self, features_df: pd.DataFrame, symbol: str) -> Optional[pd.Series]:
        """Sklearn模型的批量预测"""
        handler = self.active_model_handler

        # 1. 验证特征列
        missing_cols = set(self.required_features) - set(features_df.columns)
        if missing_cols:
            self.logger.error(f"缺少特征列: {missing_cols}")
            return None

        # 2. 准备数据
        features_clean = features_df[self.required_features].copy()
        features_clean = features_clean.replace([np.inf, -np.inf], np.nan)
        features_clean = features_clean.ffill().bfill().fillna(0.0)

        X_unscaled = features_clean.values
        X_scaled = handler.scaler.transform(X_unscaled)

        # 3. 批量预测
        probabilities = handler.model.predict_proba(X_scaled)
        alpha_scores = probabilities[:, 1] - 0.5

        # 4. 返回结果
        result_series = pd.Series(alpha_scores, index=features_df.index, name='alpha_score')
        return result_series

    def get_feature_importance(self) -> Dict[str, float]:
        if isinstance(self.active_model_handler,
                      SklearnHandler) and self.active_model_handler.model is not None and hasattr(
                self.active_model_handler.model, 'feature_importances_'):
            try:
                check_is_fitted(self.active_model_handler.model)
                importances = self.active_model_handler.model.feature_importances_
                return dict(sorted(dict(zip(self.required_features, importances)).items(), key=lambda item: item[1],
                                   reverse=True))
            except Exception as e:
                self.logger.error(f"获取特征重要性失败: {e}");
                return {}
        return {}

    def _save_model(self) -> bool:
        """保存当前活动的 Sklearn 模型和 Scaler"""
        handler = self.active_model_handler
        if not (isinstance(handler,
                           SklearnHandler) and self.current_model_filename and handler.model and handler.scaler):
            self.logger.error("无法保存 Sklearn 模型：处理器、文件名、模型或 Scaler 缺失。")
            return False
        try:
            model_path = self.models_base_path / self.current_model_filename
            scaler_path = self.models_base_path / f"{Path(self.current_model_filename).stem}_scaler.joblib"
            joblib.dump(handler.model, model_path)
            joblib.dump(handler.scaler, scaler_path)
            self.logger.info(f"Sklearn 模型 '{self.current_model_name}' 和 Scaler 已保存。")
            clear_load_ml_resource_cache()
            return True
        except Exception as e:
            self.logger.error(f"保存 Sklearn 模型失败: {e}", exc_info=True)
            return False

    #     def _calculate_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [新增] 专门用于计算所有基础技术指标和增强特征的私有方法。
        """
        #         self.logger.debug(f"Calculating base features for {len(df)} rows...")
        #         min_periods_req = 1

        # --- 基础技术指标 ---
        for period in [5, 10, 20, 30, 60]:
            df[f'ma{period}'] = df['close'].rolling(window=period, min_periods=min_periods_req).mean()

        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=14, min_periods=min_periods_req).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14, min_periods=min_periods_req).mean()
        rs = gain / loss.replace(0, 1e-9)
        df['rsi'] = 100.0 - (100.0 / (1.0 + rs))
        df['rsi'] = df['rsi'].fillna(50)

        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['hist'] = df['macd'] - df['signal']

        df['bollinger_middle'] = df['close'].rolling(window=20, min_periods=min_periods_req).mean()
        std = df['close'].rolling(window=20, min_periods=min_periods_req).std().fillna(0)
        df['bollinger_upper'] = df['bollinger_middle'] + (2 * std)
        df['bollinger_lower'] = df['bollinger_middle'] - (2 * std)

        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1).fillna(0)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14, min_periods=min_periods_req).mean()

        plus_dm = (df['high'] - df['high'].shift(1)).fillna(0)
        minus_dm = (df['low'].shift(1) - df['low']).fillna(0)
        plus_dm[(plus_dm < 0) | (plus_dm < minus_dm)] = 0.0
        minus_dm[(minus_dm < 0) | (minus_dm < plus_dm)] = 0.0
        atr14 = df['atr']
        plus_di = 100 * (plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr14.replace(0, 1e-9))
        minus_di = 100 * (minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr14.replace(0, 1e-9))
        dx_numerator = abs(plus_di - minus_di)
        dx_denominator = (plus_di + minus_di).replace(0, 1e-9)
        dx = 100 * (dx_numerator / dx_denominator)
        df['adx'] = dx.ewm(alpha=1 / 14, adjust=False).mean()

        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(20, min_periods=min_periods_req).mean()
        try:
            mean_dev = tp.rolling(20, min_periods=min_periods_req).apply(lambda x: pd.Series(x).mad(), raw=True)
        except AttributeError:
            mean_dev = abs(tp - sma_tp).rolling(20, min_periods=min_periods_req).mean()
        df['cci'] = (tp - sma_tp) / (0.015 * mean_dev.replace(0, 1e-9).fillna(1e-9))

        # --- 增强特征 ---
        df['mom_20'] = df['close'] / df['close'].shift(20).replace(0, 1e-9)
        df['mom_60'] = df['close'] / df['close'].shift(60).replace(0, 1e-9)

        daily_returns = df['close'].pct_change()
        df['vol_20'] = daily_returns.rolling(window=20, min_periods=min_periods_req).std() * np.sqrt(252)
        df['vol_60'] = daily_returns.rolling(window=60, min_periods=min_periods_req).std() * np.sqrt(252)

        if 'volume' in df.columns and not df['volume'].empty:
            vol_ma_5 = df['volume'].rolling(window=5, min_periods=min_periods_req).mean()
            vol_ma_60 = df['volume'].rolling(window=60, min_periods=min_periods_req).mean()
            df['vol_chg_5'] = vol_ma_5 / vol_ma_60.replace(0, 1e-9)
        else:
            df['vol_chg_5'] = 1.0
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        return df

    def prepare_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        # 在方法开头添加开关状态检查
        self.logger.info(f"=== 特征准备开始 for {symbol} ===")
        self.logger.info(f"TextFeatureExtractor 可用状态: {self.text_feature_extractor is not None}")
        if self.text_feature_extractor:
            self.logger.info(f"TextFeatureExtractor.is_available: {self.text_feature_extractor.is_available}")
        """
        [终极修复版] 准备所有特征，包含详细调试信息。
        """
        if data is None or data.empty:
            self.logger.error(f"prepare_features received empty or None data for {symbol}.")
            return pd.DataFrame()

        self.logger.info(f"=== 开始特征准备 for {symbol} ===")
        self.logger.info(f"输入数据形状: {data.shape}")
        self.logger.info(f"输入列名: {list(data.columns)}")

        try:
            # 步骤1: 复制数据并标准化列名
            df = data.copy()
            original_columns = list(df.columns)
            df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]
            self.logger.info(f"标准化后列名: {list(df.columns)}")

            # 步骤2: 验证基础数据列
            required_raw_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_raw_cols = set(required_raw_cols) - set(df.columns)
            if missing_raw_cols:
                self.logger.error(f"缺少基础数据列: {missing_raw_cols}")
                return pd.DataFrame()

            # 步骤3: 检查数据质量
            self.logger.info(f"数据日期范围: {df.index.min()} 到 {df.index.max()}")
            self.logger.info(
                f"Close列统计: min={df['close'].min():.2f}, max={df['close'].max():.2f}, mean={df['close'].mean():.2f}")

            # 步骤4: 检查是否已有技术指标
            expected_technical_cols = ['ma_5', 'ma_10', 'ma_20', 'rsi', 'macd']
            existing_technical_cols = [col for col in expected_technical_cols if col in df.columns]

            if len(existing_technical_cols) >= 3:
                self.logger.info(f"检测到现有技术指标: {existing_technical_cols}")
                has_technical = True
            else:
                self.logger.info("未检测到技术指标，需要计算")
                has_technical = False

            # 步骤5: 计算或验证技术指标
            if not has_technical:
                self.logger.info("开始计算技术指标...")
                try:
                    from core.analysis.technical import TechnicalAnalyzer
                    technical_analyzer = TechnicalAnalyzer(self.config)
                    df_with_tech = technical_analyzer.analyze(df)

                    if df_with_tech.empty:
                        self.logger.error("TechnicalAnalyzer 返回空 DataFrame")
                        return pd.DataFrame()

                    df = df_with_tech
                    self.logger.info(f"技术指标计算完成，新的列数: {len(df.columns)}")
                    self.logger.info(f"新增的列: {set(df.columns) - set(original_columns)}")

                except Exception as e:
                    self.logger.error(f"技术指标计算失败: {e}", exc_info=True)
                    return pd.DataFrame()

            # 步骤6: 映射列名到配置期望的格式
            self.logger.info("开始列名映射...")
            column_mapping = {
                # TechnicalAnalyzer 输出 -> config.py 期望
                'ma_5': 'ma5', 'ma_10': 'ma10', 'ma_20': 'ma20', 'ma_30': 'ma30', 'ma_60': 'ma60',
                'macd_signal': 'signal', 'macd_diff': 'hist',
                'bb_upper': 'bollinger_upper', 'bb_middle': 'bollinger_middle', 'bb_lower': 'bollinger_lower'
            }

            for tech_name, config_name in column_mapping.items():
                if tech_name in df.columns and config_name not in df.columns:
                    df[config_name] = df[tech_name]
                    self.logger.debug(f"映射: {tech_name} -> {config_name}")

            # 步骤7: 计算缺失的增强特征
            self.logger.info("计算增强特征...")

            # 动量特征
            if 'mom_20' not in df.columns:
                df['mom_20'] = df['close'] / df['close'].shift(20).replace(0, 1e-9)
                self.logger.debug("计算 mom_20")

            if 'mom_60' not in df.columns:
                df['mom_60'] = df['close'] / df['close'].shift(60).replace(0, 1e-9)
                self.logger.debug("计算 mom_60")

            # 波动率特征
            if 'vol_20' not in df.columns:
                daily_returns = df['close'].pct_change()
                df['vol_20'] = daily_returns.rolling(window=20, min_periods=1).std() * np.sqrt(252)
                self.logger.debug("计算 vol_20")

            if 'vol_60' not in df.columns:
                daily_returns = df['close'].pct_change()
                df['vol_60'] = daily_returns.rolling(window=60, min_periods=1).std() * np.sqrt(252)
                self.logger.debug("计算 vol_60")

            # 成交量特征
            if 'vol_chg_5' not in df.columns:
                if 'volume' in df.columns and not df['volume'].isna().all():
                    vol_ma_5 = df['volume'].rolling(window=5, min_periods=1).mean()
                    vol_ma_60 = df['volume'].rolling(window=60, min_periods=1).mean()
                    df['vol_chg_5'] = vol_ma_5 / vol_ma_60.replace(0, 1e-9)
                    self.logger.debug("计算 vol_chg_5")
                else:
                    df['vol_chg_5'] = 1.0
                    self.logger.debug("volume列无效，vol_chg_5设为1.0")

            # 步骤8: 初始化文本特征
            self.logger.info("初始化文本特征列...")
            for col in self.text_feature_columns:
                if col not in df.columns:
                    df[col] = 0.0
                    self.logger.debug(f"初始化文本特征: {col} = 0.0")

            # 步骤9: 智能获取文本特征（支持开关控制）
            text_features_obtained = False

            if self.text_feature_extractor and self.text_feature_extractor.is_available:
                self.logger.info(f"TextFeatureExtractor 可用，尝试为 {symbol} 获取文本特征...")

                try:
                    # 调用修复后的批量文本特征获取方法
                    text_features_df = self.text_feature_extractor.get_and_extract_features_for_backtest(symbol,
                                                                                                         df.index)

                    if text_features_df is not None and not text_features_df.empty:
                        self.logger.info(f"成功获取文本特征，形状: {text_features_df.shape}")

                        # 确保索引名称一致
                        if df.index.name != text_features_df.index.name:
                            text_features_df.index.name = df.index.name

                        # 只合并我们需要的文本特征列
                        text_cols_to_merge = [col for col in self.text_feature_columns if
                                              col in text_features_df.columns]

                        if text_cols_to_merge:
                            # 使用 left join 确保保持原有的数据行数
                            df = df.join(text_features_df[text_cols_to_merge], how='left')

                            # 填充可能的 NaN 值
                            for col in text_cols_to_merge:
                                df[col] = df[col].fillna(0.0)

                            self.logger.info(f"成功合并 {len(text_cols_to_merge)} 个文本特征列: {text_cols_to_merge}")
                            text_features_obtained = True
                        else:
                            self.logger.warning("文本特征DataFrame中没有找到期望的列")
                    else:
                        self.logger.warning("文本特征获取返回了空结果")

                except Exception as e:
                    self.logger.error(f"文本特征获取过程中出错: {e}")
                    self.logger.info("将继续使用默认的文本特征值")
            else:
                self.logger.info("TextFeatureExtractor 不可用，跳过文本特征获取")

            # 确保所有文本特征列都存在且有有效值
            for col in self.text_feature_columns:
                if col not in df.columns:
                    df[col] = 0.0
                    self.logger.debug(f"补充缺失的文本特征列: {col}")
                elif df[col].isna().any():
                    df[col] = df[col].fillna(0.0)
                    self.logger.debug(f"填充文本特征列中的NaN: {col}")

            if text_features_obtained:
                self.logger.info("✅ 文本特征获取成功")
            else:
                self.logger.info("⚠️ 使用默认文本特征值（全为0）")

            # 步骤10: 数据清理
            self.logger.info("开始数据清理...")
            initial_shape = df.shape

            # 替换无穷值
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            inf_count = df.isnull().sum().sum() - initial_shape[0] * initial_shape[1] + df.count().sum()
            if inf_count > 0:
                self.logger.info(f"替换了 {inf_count} 个无穷值")

            # 填充缺失值
            df = df.ffill().bfill().fillna(0)

            # 步骤11: 最终列选择和验证
            self.logger.info("最终列选择...")
            required_base_features = [col for col in self.base_feature_columns if col != 'close']  # 暂时排除close进行检查
            required_text_features = self.text_feature_columns[:]
            all_required_features = required_base_features + required_text_features + ['close']

            self.logger.info(f"需要的特征列: {all_required_features}")

            available_features = [col for col in all_required_features if col in df.columns]
            missing_features = set(all_required_features) - set(available_features)

            if missing_features:
                self.logger.warning(f"缺失的特征: {missing_features}")
                # 对于缺失的非关键特征，用默认值填充
                for missing_col in missing_features:
                    if missing_col != 'close':  # close是必需的
                        df[missing_col] = 0.0
                        self.logger.info(f"用默认值填充缺失特征: {missing_col}")
                        available_features.append(missing_col)

            # 检查close列
            if 'close' not in df.columns:
                self.logger.error("关键列 'close' 缺失，无法继续")
                return pd.DataFrame()

            # 最终数据框
            final_df = df[available_features].copy()

            # 步骤12: 最终验证
            self.logger.info(f"最终数据形状: {final_df.shape}")
            self.logger.info(f"最终列名: {list(final_df.columns)}")

            # 检查是否有空数据
            if final_df.empty:
                self.logger.error("最终数据框为空!")
                return pd.DataFrame()

            # 检查数据质量
            null_counts = final_df.isnull().sum()
            if null_counts.sum() > 0:
                self.logger.warning(f"最终数据中仍有空值: {null_counts[null_counts > 0].to_dict()}")

            # 检查数值范围
            numeric_cols = final_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                col_data = final_df[col]
                if col_data.isna().all():
                    self.logger.warning(f"列 {col} 全为NaN")
                elif (col_data == 0).all():
                    self.logger.warning(f"列 {col} 全为0")

            self.logger.info("=== 特征准备完成 ===")
            return final_df

        except Exception as e:
            self.logger.error(f"特征准备过程中发生异常: {e}", exc_info=True)
            return pd.DataFrame()

    def validate_features(self, data: pd.DataFrame) -> bool:
        """Validates if the DataFrame contains the required features and no NaNs."""
        if not self.is_available: return False
        if data is None or data.empty: return False
        if not self.required_features: return False  # Need features defined

        try:
            missing = set(self.required_features) - set(data.columns)
            if missing:
                self.logger.error(f"Feature validation failed: Missing columns: {missing}")
                return False
            if data[self.required_features].isnull().values.any():
                nan_cols = data.columns[data[self.required_features].isnull().any()].tolist()
                self.logger.error(f"Feature validation failed: NaN values found in columns: {nan_cols}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error during feature validation: {e}", exc_info=True)
            return False