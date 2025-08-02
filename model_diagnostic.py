# 创建文件：model_diagnostic.py
import tensorflow as tf
import joblib
import numpy as np
from pathlib import Path


def diagnose_model(model_path, scaler_path):
    """诊断模型和scaler文件的完整性"""

    print(f"=== 模型诊断开始 ===")

    # 1. 检查文件存在性
    print(f"模型文件存在: {Path(model_path).exists()}")
    print(f"Scaler文件存在: {Path(scaler_path).exists()}")

    if not Path(model_path).exists() or not Path(scaler_path).exists():
        print("❌ 关键文件缺失")
        return False

    try:
        # 2. 尝试加载模型
        print("正在加载模型...")
        model = tf.keras.models.load_model(model_path)
        print(f"✅ 模型加载成功: {model.name}")
        print(f"模型输入形状: {model.input_shape}")
        print(f"模型输出形状: {model.output_shape}")

        # 3. 尝试加载scaler
        print("正在加载scaler...")
        scaler = joblib.load(scaler_path)
        print(f"✅ Scaler加载成功: {type(scaler)}")
        print(f"Scaler特征数: {scaler.n_features_in_}")

        # 4. 创建测试数据
        print("创建测试数据...")
        n_features = scaler.n_features_in_
        lookback = 60  # 假设的lookback

        # 创建随机测试数据
        test_data = np.random.randn(lookback + 10, n_features)

        # 5. 测试scaler
        print("测试数据缩放...")
        scaled_data = scaler.transform(test_data)
        print(f"✅ 数据缩放成功")

        # 6. 准备模型输入
        print("准备模型输入...")
        X_test = []
        for i in range(lookback, len(scaled_data)):
            X_test.append(scaled_data[i - lookback:i])
        X_test = np.array(X_test)
        print(f"测试输入形状: {X_test.shape}")

        # 7. 测试模型预测
        print("测试模型预测...")
        predictions = model.predict(X_test[:5], verbose=0)  # 只测试前5个样本
        print(f"✅ 预测成功: {predictions.shape}")
        print(f"预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")

        print("=== 模型诊断完成：所有测试通过 ===")
        return True

    except Exception as e:
        print(f"❌ 诊断失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# 使用示例
if __name__ == "__main__":
    # 替换为您的实际路径
    model_path = "8690project/models/Alpha-Transformer_model.h5"
    scaler_path = "8690project/models/Alpha-Transformer_model_scaler.joblib"

    diagnose_model(model_path, scaler_path)