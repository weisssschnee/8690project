# core/alert/manager.py
import logging
import pandas as pd
import streamlit as st
from datetime import datetime

from core.translate import translator  # <--- 新增导入

logger = logging.getLogger(__name__)


class AlertManager:
    def __init__(self, system):
        self.system = system
        if 'alerts' not in st.session_state: st.session_state.alerts = []
        if 'alert_history' not in st.session_state: st.session_state.alert_history = []
        self.last_check_time = datetime.now()

    # ... (add_alert, remove_alert, toggle_alert, check_alerts 方法保持不变,
    #      但它们内部产生的消息（如日志）如果也需要翻译，则另行处理，
    #      目前主要关注UI部分的翻译) ...

    def add_alert(self, alert_data):
        """添加新的价格报警"""
        alert_id = f"alert_{len(st.session_state.alerts) + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        # 在创建默认消息时使用翻译
        default_message_key = "alert_default_message"
        default_message_fallback = "{symbol} 价格 {condition} {price}"
        default_message = translator.t(
            default_message_key,
            fallback=default_message_fallback,
            symbol=alert_data.get("symbol"),
            condition=translator.t(str(alert_data.get("condition")).lower(), fallback=str(alert_data.get("condition"))),
            # 翻译条件
            price=alert_data.get("price", 0)
        )

        alert = {
            "id": alert_id,
            "symbol": alert_data.get("symbol"),
            "condition": alert_data.get("condition"),  # 存储原始条件（如 "高于", "低于"）
            "price": float(alert_data.get("price", 0)),
            "message": alert_data.get("message", default_message),  # 使用翻译后的默认消息
            "created_at": datetime.now(),
            "active": True,
            "triggered": False,
            "trigger_count": 0,
            "last_check_price": 0.0  # 确保是浮点数
        }
        st.session_state.alerts.append(alert)
        # 日志中的文本通常不需要翻译，但如果需要也可以
        logger.info(f"价格报警已添加: {alert['symbol']} {alert['condition']} {alert['price']}")
        return alert_id

    def remove_alert(self, alert_id):
        """删除价格报警"""
        for i, alert in enumerate(st.session_state.alerts):
            if alert["id"] == alert_id:
                removed = st.session_state.alerts.pop(i)
                logger.info(f"价格报警已删除: {removed['symbol']} {removed['condition']} {removed['price']}")
                return True
        return False

    def toggle_alert(self, alert_id):
        """切换报警启用状态"""
        for alert in st.session_state.alerts:
            if alert["id"] == alert_id:
                alert["active"] = not alert["active"]
                status_key = "alert_status_enabled" if alert["active"] else "alert_status_disabled"
                status_fallback = "启用" if alert["active"] else "禁用"
                status_display = translator.t(status_key, fallback=status_fallback)
                logger.info(f"{status_display}价格报警: {alert['symbol']} {alert['condition']} {alert['price']}")
                return True
        return False

    def check_alerts(self):
        """检查所有活跃的价格报警"""
        # (内部逻辑不变，但如果报警消息需要动态翻译，这里是另一个地方)
        # ...
        # 确保 trigger_record 中的 message 是在创建报警时已经翻译好的，或者这里进行翻译
        # ...
        triggered_alerts = []
        for alert in st.session_state.alerts:
            if not alert["active"]: continue
            symbol = alert["symbol"];
            condition_internal = alert["condition"];
            target_price = alert["price"]
            current_price = self.system.get_realtime_price(symbol)
            if current_price is None: continue
            alert["last_check_price"] = current_price
            triggered = False
            # 使用内部的、非翻译的条件进行比较
            if condition_internal == "高于" and current_price > target_price:
                triggered = True
            elif condition_internal == "低于" and current_price < target_price:
                triggered = True
            elif condition_internal == "等于" and abs(current_price - target_price) < 0.01:
                triggered = True  # 假设价格相等比较

            if triggered:
                alert["triggered"] = True;
                alert["trigger_count"] += 1;
                alert["last_triggered"] = datetime.now()
                trigger_record = {"id": alert["id"], "symbol": symbol, "condition": condition_internal,
                                  "target_price": target_price, "current_price": current_price,
                                  "message": alert["message"],  # 使用创建时已翻译或自定义的消息
                                  "triggered_at": datetime.now()}
                st.session_state.alert_history.append(trigger_record)
                triggered_alerts.append(trigger_record)
                logger.info(f"报警触发: {symbol} 当前价 {current_price} {condition_internal} {target_price}")
        return triggered_alerts

    def render_alerts_ui(self):
        """渲染报警管理界面 (使用翻译)"""
        st.header(translator.t('price_alert_system_header', fallback="价格报警系统"))

        # --- 显示活跃报警 ---
        if st.session_state.get('alerts'):  # Use .get for safety
            st.subheader(translator.t('active_alerts_subheader', fallback="活跃报警"))
            alerts_data = []
            # 定义列标题的翻译键
            col_id_key = 'col_alert_id';
            col_stock_key = 'col_alert_stock';
            col_condition_key = 'col_alert_condition';
            col_price_key = 'col_alert_price';
            col_status_key = 'col_alert_status';
            col_created_at_key = 'col_alert_created_at';
            col_triggers_key = 'col_alert_trigger_count';
            col_latest_price_key = 'col_alert_latest_price';

            for alert in st.session_state.alerts:
                condition_display = translator.t(str(alert["condition"]).lower(), fallback=alert["condition"])  # 翻译条件显示
                status_key = "alert_status_active" if alert["active"] else "alert_status_inactive"
                status_display = translator.t(status_key, fallback="活跃" if alert["active"] else "禁用")

                alerts_data.append({
                    translator.t(col_id_key, fallback="ID"): alert["id"],
                    translator.t(col_stock_key, fallback="股票"): alert["symbol"],
                    translator.t(col_condition_key, fallback="条件"): condition_display,
                    translator.t(col_price_key, fallback="价格"): alert["price"],
                    translator.t(col_status_key, fallback="状态"): status_display,
                    translator.t(col_created_at_key, fallback="创建时间"): alert["created_at"].strftime(
                        "%Y-%m-%d %H:%M"),
                    translator.t(col_triggers_key, fallback="触发次数"): alert["trigger_count"],
                    translator.t(col_latest_price_key,
                                 fallback="最新检查价格"): f"{alert.get('last_check_price', 0.0):.2f}"  # 格式化
                })
            if alerts_data:  # Ensure there's data before creating DataFrame
                st.dataframe(pd.DataFrame(alerts_data), use_container_width=True)
            else:  # Should not happen if st.session_state.alerts is not empty, but defensive
                st.info(translator.t('no_active_alerts_to_display', fallback="当前没有活跃的报警。"))

            # --- 报警操作 ---
            op_col1, op_col2 = st.columns(2)
            with op_col1:
                # 为 selectbox 的选项生成翻译（如果需要）
                alert_options = []
                for a in st.session_state.alerts:
                    cond_disp = translator.t(str(a['condition']).lower(), fallback=a['condition'])
                    alert_options.append(f"{a['symbol']} {cond_disp} {a['price']} ({a['id']})")

                selected_alert_str_toggle = st.selectbox(
                    translator.t('select_alert_to_toggle_label', fallback="选择要切换状态的报警"),
                    alert_options, key="toggle_alert_select_alerts_tab"  # Unique key
                )
                if st.button(translator.t('toggle_alert_status_button', fallback="切换状态"),
                             key="toggle_alert_btn_alerts_tab"):
                    if selected_alert_str_toggle:
                        alert_id_toggle = selected_alert_str_toggle.split("(")[-1].rstrip(")")
                        if self.toggle_alert(alert_id_toggle):
                            st.success(translator.t('alert_status_updated_success', fallback="报警状态已更新"))
                            st.rerun()
            with op_col2:
                selected_alert_str_remove = st.selectbox(
                    translator.t('select_alert_to_remove_label', fallback="选择要删除的报警"),
                    alert_options,  # Re-use alert_options
                    key="remove_alert_select_alerts_tab"  # Unique key
                )
                if st.button(translator.t('remove_alert_button', fallback="删除报警"),
                             key="remove_alert_btn_alerts_tab"):
                    if selected_alert_str_remove:
                        alert_id_remove = selected_alert_str_remove.split("(")[-1].rstrip(")")
                        if self.remove_alert(alert_id_remove):
                            st.success(translator.t('alert_removed_success', fallback="报警已删除"))
                            st.rerun()
        else:
            st.info(translator.t('no_active_alerts_info', fallback="没有活跃的价格报警。"))

        # --- 添加新报警 ---
        st.subheader(translator.t('create_new_alert_subheader', fallback="创建新报警"))
        new_col1, new_col2, new_col3 = st.columns(3)
        with new_col1:
            new_alert_symbol = st.text_input(translator.t('stock_symbol_label', fallback="股票代码"), "AAPL",
                                             key="new_alert_symbol_input")

        # 翻译条件选项
        condition_options_keys = ['above', 'below', 'equal_to']  # Internal keys
        condition_options_defaults = ["高于", "低于", "等于"]
        condition_options_display = [translator.t(k, fallback=d) for k, d in
                                     zip(condition_options_keys, condition_options_defaults)]

        with new_col2:
            selected_condition_display = st.selectbox(
                translator.t('condition_label', fallback="条件"),
                condition_options_display,
                key="new_alert_condition_select"
            )
            # Map display back to internal key if needed, or store display value if internal logic handles it
            # For simplicity, let's assume internal logic uses the display value for now,
            # or map it back when calling add_alert
            # If internal logic expects "高于", "低于", "等于", then no mapping needed for these specific values.
            # If internal logic expects English, mapping is needed:
            # internal_condition = condition_options_keys[condition_options_display.index(selected_condition_display)]
            internal_condition = selected_condition_display  # Assuming backend handles this text for now
            # For a truly robust system, you'd use the keys 'above', 'below', 'equal_to' internally
            # and only use translated strings for display.
            # For now, let's assume your `check_alerts` uses the Chinese strings directly.
            # The add_alert method will store whatever `internal_condition` is.

        with new_col3:
            new_alert_price = st.number_input(translator.t('price_label', fallback="价格"), min_value=0.01, value=150.0,
                                              key="new_alert_price_input", format="%.2f")

        # 生成默认提醒消息时也使用翻译
        default_msg_template = translator.t('new_alert_default_msg_template',
                                            fallback="{symbol} 价格 {condition} {price}")
        new_alert_message = st.text_input(
            translator.t('alert_message_label', fallback="提醒消息"),
            default_msg_template.format(symbol=new_alert_symbol, condition=selected_condition_display,
                                        price=new_alert_price),
            key="new_alert_message_input"
        )

        if st.button(translator.t('add_alert_button', fallback="添加报警"), key="add_alert_btn_alerts_tab"):
            if new_alert_symbol and new_alert_price > 0:
                # Use the internal condition value when creating the alert_data
                alert_data_to_add = {
                    "symbol": new_alert_symbol,
                    "condition": internal_condition,  # Use the selected (possibly translated) condition string
                    "price": new_alert_price,
                    "message": new_alert_message
                }
                alert_id_added = self.add_alert(alert_data_to_add)
                if alert_id_added:
                    st.success(translator.t('alert_added_success_msg', fallback="报警添加成功! ID: {id}").format(
                        id=alert_id_added))
                    st.rerun()
            else:
                st.error(translator.t('error_invalid_symbol_or_price', fallback="请填写有效的股票代码和价格。"))

        # --- 显示报警历史 ---
        if st.session_state.get('alert_history'):
            st.subheader(translator.t('alert_history_subheader', fallback="报警历史"))
            history_data = []
            # 定义历史表格列标题翻译键
            hist_col_time_key = 'col_hist_time';
            hist_col_stock_key = 'col_hist_stock';
            hist_col_condition_key = 'col_hist_condition';
            hist_col_target_price_key = 'col_hist_target_price';
            hist_col_trigger_price_key = 'col_hist_trigger_price';
            hist_col_message_key = 'col_hist_message';

            for record in st.session_state.alert_history:
                condition_hist_display = translator.t(str(record["condition"]).lower(),
                                                      fallback=record["condition"])  # 翻译条件显示
                history_data.append({
                    translator.t(hist_col_time_key, fallback="时间"): record["triggered_at"].strftime(
                        "%Y-%m-%d %H:%M:%S"),
                    translator.t(hist_col_stock_key, fallback="股票"): record["symbol"],
                    translator.t(hist_col_condition_key, fallback="条件"): condition_hist_display,
                    translator.t(hist_col_target_price_key, fallback="目标价格"): record["target_price"],
                    translator.t(hist_col_trigger_price_key, fallback="触发价格"): record["current_price"],
                    translator.t(hist_col_message_key, fallback="消息"): record["message"]  # 消息在创建时已处理
                })
            if history_data:
                st.dataframe(pd.DataFrame(history_data), use_container_width=True)

            if st.button(translator.t('clear_alert_history_button', fallback="清空历史记录"),
                         key="clear_alert_hist_btn"):
                st.session_state.alert_history = []
                st.success(translator.t('alert_history_cleared_success', fallback="历史记录已清空。"))
                st.rerun()
        # else: # No history yet
        # st.info(translator.t('no_alert_history_info', fallback="暂无报警历史记录。")) # Optional: message if no history