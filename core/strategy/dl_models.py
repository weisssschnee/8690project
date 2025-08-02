# core/strategy/dl_models.py
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict,Optional

# 尝试导入 TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model, Sequential
    from tensorflow.keras.layers import (
        Input, Dense, Dropout, LayerNormalization,
        MultiHeadAttention, GlobalAveragePooling1D,
        Embedding, Layer, LSTM
    )
    from tensorflow.keras.callbacks import EarlyStopping

    # 尝试导入 tensorflow_addons
    try:
        import tensorflow_addons as tfa

        TENSORFLOW_ADDONS_AVAILABLE = True
    except ImportError:
        tfa = None
        TENSORFLOW_ADDONS_AVAILABLE = False
        logging.warning("tensorflow-addons not found. Some advanced model features might be unavailable.")

    TENSORFLOW_AVAILABLE = True
except ImportError:
    # 定义占位符，以防导入失败
    Model, load_model, Sequential, Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Embedding, Layer, LSTM, EarlyStopping = (
                                                                                                                                                                  object,) * 13
    tfa = None
    TENSORFLOW_AVAILABLE = False
    TENSORFLOW_ADDONS_AVAILABLE = False
    logging.warning("TensorFlow/Keras not found. Deep learning models will be unavailable.")

logger = logging.getLogger(__name__)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model

        # 预计算位置编码
        angle_rads = self._get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )

        # 对偶数索引应用sin，对奇数索引应用cos
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        self.pos_encoding = tf.cast(angle_rads, dtype=tf.float32)
        self.pos_encoding = self.pos_encoding[np.newaxis, ...]

    def _get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'position': self.position,
            'd_model': self.d_model,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # 移除可能存在的不兼容参数
        valid_config = {
            'position': config['position'],
            'd_model': config['d_model']
        }
        # 只保留 Layer 基类能接受的参数
        base_config = {}
        if 'name' in config:
            base_config['name'] = config['name']
        if 'dtype' in config:
            # 处理 dtype 配置
            dtype_config = config['dtype']
            if isinstance(dtype_config, dict) and 'name' in dtype_config:
                base_config['dtype'] = dtype_config['name']
            elif isinstance(dtype_config, str):
                base_config['dtype'] = dtype_config

        valid_config.update(base_config)
        return cls(**valid_config)


class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(d_model),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerEncoderBlock, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # 移除可能存在的不兼容参数
        valid_config = {
            'd_model': config['d_model'],
            'num_heads': config['num_heads'],
            'ff_dim': config['ff_dim'],
            'dropout': config.get('dropout', 0.1)
        }

        # 只保留 Layer 基类能接受的参数
        base_config = {}
        if 'name' in config:
            base_config['name'] = config['name']
        if 'dtype' in config:
            dtype_config = config['dtype']
            if isinstance(dtype_config, dict) and 'name' in dtype_config:
                base_config['dtype'] = dtype_config['name']
            elif isinstance(dtype_config, str):
                base_config['dtype'] = dtype_config

        valid_config.update(base_config)
        return cls(**valid_config)

class BaseDLModelHandler:
    """所有深度学习模型处理器的基类，提供统一的数据准备方法"""

    def __init__(self, config: Dict):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(f"Cannot create {self.__class__.__name__}: TensorFlow/Keras is not installed.")
        self.config = config
        self.model: Optional[Model] = None

    def prepare_data(self, X_data: np.ndarray, y_data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        [最终版] 将 2D 特征数组和 1D 目标数组转换为 DL 模型所需的 3D 输入和对齐后的目标。
        只进行 NumPy 操作。
        """
        if len(X_data) < lookback: return np.array([]), np.array([])

        shape = (X_data.shape[0] - lookback + 1, lookback, X_data.shape[1])
        strides = (X_data.strides[0], X_data.strides[0], X_data.strides[1])
        X_final = np.lib.stride_tricks.as_strided(X_data, shape=shape, strides=strides)
        y_final = y_data[lookback - 1:]
        return X_final, y_final


class TransformerModelHandler(BaseDLModelHandler):
    """负责处理 Transformer 分类模型的构建、训练和预测。"""

    def build_model(self, input_shape: Tuple[int, int]):
        """构建 Transformer 分类模型架构。"""
        params = self.config.get('ML_HYPERPARAMETERS', {}).get('Transformer', {})
        lookback, n_features = input_shape
        embed_dim = params.get('embed_dim', 32)
        num_heads = params.get('num_heads', 2)
        ff_dim = params.get('ff_dim', 32)
        dropout_rate = params.get('dropout', 0.1)

        inputs = Input(shape=input_shape)
        x = Dense(embed_dim)(inputs)
        x = PositionalEncoding(lookback, embed_dim)(x)
        x = TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)
        x = GlobalAveragePooling1D()(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(1, activation="sigmoid")(x)  # Sigmoid for binary classification

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=params.get('optimizer', 'adam'),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        logger.info("Transformer classification model built successfully.")
        self.model.summary(print_fn=logger.info)
        return self.model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """训练 Transformer 模型"""
        params = self.config.get('ML_HYPERPARAMETERS', {}).get('Transformer', {})
        early_stopping = EarlyStopping(monitor='val_loss', patience=params.get('patience', 10),
                                       restore_best_weights=True)

        # vvvvvv START OF FIX vvvvvv
        # 确保 y 是正确的形状 (num_samples, 1) 并且是 float32
        y_train = np.asarray(y_train).astype('float32').reshape(-1, 1)
        y_val = np.asarray(y_val).astype('float32').reshape(-1, 1)
        # ^^^^^^ END OF FIX ^^^^^^

        history = self.model.fit(
            X_train, y_train,
            epochs=params.get('epochs', 50),
            batch_size=params.get('batch_size', 32),
            validation_data=(X_val, y_val),
            callbacks=[early_stopping], verbose=1
        )
        return history.history


class AlphaTransformerModelHandler(BaseDLModelHandler):
    """负责处理 Alpha-Transformer 回归模型的构建、训练和预测。"""

    def build_model(self, input_shape: Tuple[int, int]):
        """[最终修复版] 构建 Alpha-Transformer 回归模型架构。"""
        # --- 1. 从 config 中获取超参数字典 ---
        params = self.config.get('ML_HYPERPARAMETERS', {}).get('AlphaTransformer', {})

        # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
        # --- 2. 从字典中提取超参数到本地变量 ---
        lookback, n_features = input_shape
        d_model = params.get('d_model', 64)
        num_heads = params.get('num_heads', 4)
        ff_dim = params.get('ff_dim', 128)
        num_blocks = params.get('num_blocks', 2)
        dropout_rate = params.get('dropout', 0.1)
        # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^

        # --- 3. 构建模型 (现在所有变量都已定义) ---
        inputs = Input(shape=input_shape)
        x = Dense(d_model)(inputs)
        x = PositionalEncoding(lookback, d_model)(x)
        for _ in range(num_blocks):
            x = TransformerEncoderBlock(d_model, num_heads, ff_dim, dropout_rate)(x)

        x = GlobalAveragePooling1D()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(d_model // 2, activation='relu')(x)
        outputs = Dense(1, activation='linear')(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=params.get('optimizer', 'adam'),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        logger.info("Alpha-Transformer regression model built successfully.")
        self.model.summary(print_fn=logger.info)
        return self.model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """训练 Alpha-Transformer 模型"""
        params = self.config.get('ML_HYPERPARAMETERS', {}).get('AlphaTransformer', {})
        early_stopping = EarlyStopping(monitor='val_loss', patience=params.get('patience', 10),
                                       restore_best_weights=True)

        # vvvvvv START OF FIX vvvvvv
        # 确保 y 是正确的形状 (num_samples, 1) 并且是 float32
        y_train = np.asarray(y_train).astype('float32').reshape(-1, 1)
        y_val = np.asarray(y_val).astype('float32').reshape(-1, 1)
        # ^^^^^^ END OF FIX ^^^^^^

        history = self.model.fit(
            X_train, y_train,
            epochs=params.get('epochs', 50),
            batch_size=params.get('batch_size', 32),
            validation_data=(X_val, y_val),
            callbacks=[early_stopping], verbose=1
        )
        return history.history


class LSTMModelHandler(BaseDLModelHandler):
    """负责处理 LSTM 分类模型的构建、训练和预测。"""

    def build_model(self, input_shape: Tuple[int, int]):
        """构建 LSTM 分类模型架构。"""
        params = self.config.get('ML_HYPERPARAMETERS', {}).get('LSTM', {})

        self.model = Sequential([
            LSTM(units=params.get('units', 50), return_sequences=True, input_shape=input_shape),
            Dropout(params.get('dropout', 0.2)),
            LSTM(units=params.get('units', 50), return_sequences=False),
            Dropout(params.get('dropout', 0.2)),
            Dense(units=params.get('dense_units', 25), activation='relu'),
            Dense(units=1, activation='sigmoid')  # Sigmoid for binary classification
        ])
        self.model.compile(
            optimizer=params.get('optimizer', 'adam'),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        logger.info("LSTM classification model built successfully.")
        self.model.summary(print_fn=logger.info)
        return self.model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """训练 LSTM 模型"""
        params = self.config.get('ML_HYPERPARAMETERS', {}).get('LSTM', {})
        early_stopping = EarlyStopping(monitor='val_loss', patience=params.get('patience', 10),
                                       restore_best_weights=True, verbose=1)

        # vvvvvv START OF FIX vvvvvv
        # 确保 y 是正确的形状 (num_samples, 1) 并且是 float32
        y_train = np.asarray(y_train).astype('float32').reshape(-1, 1)
        y_val = np.asarray(y_val).astype('float32').reshape(-1, 1)
        # ^^^^^^ END OF FIX ^^^^^^

        history = self.model.fit(
            X_train, y_train,
            epochs=params.get('epochs', 50),
            batch_size=params.get('batch_size', 32),
            validation_data=(X_val, y_val),
            callbacks=[early_stopping], verbose=1
        )
        return history.history

    __all__ = [
        'PositionalEncoding',
        'TransformerEncoderBlock',
        'LSTMModelHandler',
        'TransformerModelHandler',
        'AlphaTransformerModelHandler',
        'TENSORFLOW_AVAILABLE'
    ]
