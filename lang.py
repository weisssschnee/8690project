import streamlit as st


class LangPlugin:
    def __init__(self):
        self.translations = {
            "en": {
                "仪表盘": "Dashboard",
                "市场": "Market",
                "交易": "Trading",
                # 添加更多翻译...
            },
            "zh": {
                "Dashboard": "仪表盘",
                "Market": "市场",
                "Trading": "交易",
                # 中文无需翻译，自动回退
            }
        }

    def t(self, text):
        """智能翻译函数"""
        lang = st.session_state.get("lang", "zh")
        return self.translations[lang].get(text, text)

    def switcher(self):
        """在任意位置添加切换按钮"""
        if st.button("中/EN"):
            current = st.session_state.get("lang", "zh")
            st.session_state.lang = "en" if current == "zh" else "zh"
            st.rerun()


# 初始化插件实例
i18n = LangPlugin()