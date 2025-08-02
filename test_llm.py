# test_llm.py
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv('.env.txt')


def test_gemini():
    """测试Gemini是否可用"""
    try:
        import google.generativeai as genai
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("❌ GEMINI_API_KEY 未配置")
            return False

        print(f"✅ Gemini库可用，API密钥: {api_key[:10]}...")

        # 尝试初始化
        from core.strategy.llm_trader_adapters import GeminiTraderAdapter
        adapter = GeminiTraderAdapter(api_key=api_key, model_name='gemini-2.5-flash')
        print("✅ GeminiTraderAdapter 创建成功")
        return True

    except ImportError as e:
        print(f"❌ Gemini库导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ Gemini适配器创建失败: {e}")
        return False


def test_deepseek():
    """测试DeepSeek是否可用"""
    try:
        from deepseek.client import DeepSeekClient
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            print("❌ DEEPSEEK_API_KEY 未配置")
            return False

        print(f"✅ DeepSeek库可用，API密钥: {api_key[:10]}...")

        # 尝试初始化
        from core.strategy.llm_trader_adapters import DeepSeekTraderAdapter
        adapter = DeepSeekTraderAdapter(api_key=api_key, model_name='deepseek-reasoner')
        print("✅ DeepSeekTraderAdapter 创建成功")
        return True

    except ImportError as e:
        print(f"❌ DeepSeek库导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ DeepSeek适配器创建失败: {e}")
        return False


if __name__ == "__main__":
    print("🧪 测试LLM适配器...")
    print("-" * 50)

    gemini_ok = test_gemini()
    print()
    deepseek_ok = test_deepseek()

    print("-" * 50)
    if gemini_ok or deepseek_ok:
        print("✅ 至少有一个LLM适配器可用")
    else:
        print("❌ 没有可用的LLM适配器")