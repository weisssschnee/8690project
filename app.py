import streamlit as st
import logging
import traceback
import sys
from pathlib import Path
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 加载 .env 文件，以便补丁函数可以读取代理设置
from dotenv import load_dotenv
load_dotenv(project_root / ".env.txt")


# 导入并应用补丁
try:
    from core.utils.network_patch import apply_monkey_patch
    apply_monkey_patch()
except Exception as e:
    # 如果补丁失败，打印一个关键错误
    print(f"CRITICAL ERROR: Failed to apply network patch at startup: {e}")
# ^^^^^^^^^^^^^^^^^^^^ END OF GLOBAL PATCH ^^^^^^^^^^^^^^^^^^^^
import os
print("--- Streamlit App sys.path ---")
for p_idx, p_val in enumerate(sys.path):
    print(f"{p_idx}: {p_val}")
print(f"--- Streamlit App CWD: {os.getcwd()} ---")
print("------")
import logging

logger = logging.getLogger(__name__)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(f"修正后项目根目录: {project_root}")
project_root = Path(__file__).parent
sys.path.append(str(project_root))
# 新增导入翻译模块
from core.translate import translator  # <--- 新增导入
try:
    from core.system import TradingSystem  # 确保模块路径正确
except ImportError as e:
    logger.error(f"导入核心组件失败: {e}")
    st.error(f"系统初始化失败，请检查安装: {e}")
    st.code(traceback.format_exc())
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def handle_streamlit_exception():
    """处理Streamlit应用的未捕获异常"""
    try:
        # 您的主应用逻辑
        pass
    except Exception as e:
        st.error("🚨 **应用遇到错误**")
        st.error(f"错误信息: {str(e)}")

        with st.expander("详细错误信息（用于调试）"):
            st.code(traceback.format_exc())

        # 记录错误
        logger.error(f"Streamlit application error: {e}", exc_info=True)

        # 提供恢复选项
        if st.button("🔄 重新加载应用"):
            st.rerun()


def main():
    """Application entry point"""
    try:
        # 初始化语言状态 (如果 translate.py 中处理了，这里可能不需要)
        # if 'lang' not in st.session_state:
        #     st.session_state.lang = 'zh-cn' # 使用 'zh-cn'

        # 获取翻译后的标题
        page_title_key = 'app_title'
        page_title_text = translator.t(page_title_key, fallback="智能交易系统") # <--- 修改

        st.set_page_config(
            page_title=page_title_text, # <--- 使用翻译结果
            page_icon="📈",
            layout="wide"
        )

        # 不再需要预加载这个方法
        # translator._precache_common_terms()

        # Maintain system instance through session_state
        if 'system' not in st.session_state:
            logger.info("Initializing trading system...")
            st.session_state.system = TradingSystem() # TradingSystem 内部不再调用 _precache...

        # Run the system
        st.session_state.system.run()

    except Exception as e:
        logger.error(f"System runtime error: {e}", exc_info=True) # 添加 exc_info=True 获取更详细日志

        # 使用翻译键显示错误
        error_message = translator.t('runtime_error', fallback="系统运行时发生错误") # <--- 修改
        st.error(error_message)

        # Display error details and reset option
        expander_title = translator.t('error_details', fallback="错误详情") # <--- 修改
        with st.expander(expander_title, expanded=False):
            st.code(traceback.format_exc())

        reset_button_label = translator.t('reset_system_button', fallback="重置系统") # <--- 修改
        if st.button(reset_button_label):
            logger.info("重置系统按钮被点击。")
            # (重置 session state 的逻辑)
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()

