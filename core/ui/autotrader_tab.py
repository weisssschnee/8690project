# core/ui/autotrader_tab.py
import streamlit as st
from core.translate import translator
from datetime import datetime


class AutoTraderTab:
    """
    Renders the "Automation" tab UI, with corrected logic for creating and managing strategies.
    """

    def __init__(self, system):
        self.system = system
        self.persistence_manager = system.persistence_manager

    def render(self):
        st.header(translator.t('autotrader_center_header', fallback="自动化策略中心"))
        st.info(translator.t('autotrader_center_desc',
                             fallback="在这里创建、监控和管理您的所有自动化交易策略。策略由侧边栏的自动交易引擎总开关统一调度执行。"))

        self._render_create_strategy_form()

        st.markdown("---")

        st.subheader(translator.t('saved_strategies_subheader', fallback="已部署的策略"))
        self._render_saved_strategies()

    def _render_create_strategy_form(self):
        """
        [FULLY TRANSLATED] Renders the form for creating a new strategy with proper LLM availability check.
        """
        with st.expander(translator.t('create_new_strategy_expander'), expanded=True):
            with st.form("complete_strategy_form", clear_on_submit=True):
                # --- Section 1: Basic inputs ---
                c1, c2 = st.columns(2)
                with c1:
                    strategy_name = st.text_input(
                        translator.t('strategy_name_label'),
                        placeholder=translator.t('strategy_name_placeholder'),
                        key="strat_name_input_form"
                    )
                with c2:
                    symbols_input = st.text_input(
                        translator.t('stock_symbol_label'),
                        placeholder=translator.t('stock_symbols_placeholder'),
                        help=translator.t('stock_symbols_help'),
                        key="strat_symbols_input_form"
                    )

                # --- Section 2: Core Type selection ---
                st.markdown(f"**{translator.t('decision_core_label')}**")

                available_core_types = []
                available_models = list(getattr(self.system.config, 'AVAILABLE_ML_MODELS', {}).keys())
                if available_models:
                    available_core_types.append("ml_model")

                available_llms = []
                if (hasattr(self.system, 'strategy_manager') and
                        hasattr(self.system.strategy_manager, 'llm_traders') and
                        isinstance(self.system.strategy_manager.llm_traders, dict)):
                    available_llms = list(self.system.strategy_manager.llm_traders.keys())
                    if available_llms:
                        available_core_types.append("llm_trader")

                # --- Debug Info Section (Now Translated) ---
                with st.popover(translator.t('debug_info_header')):
                    st.markdown(f"**{translator.t('debug_available_ml_models')}**")
                    st.write(available_models or "[]")
                    st.markdown(f"**{translator.t('debug_available_llm_traders')}**")
                    st.write(available_llms or "[]")
                    st.markdown(f"**{translator.t('debug_strategy_manager_type')}**")
                    st.write(type(self.system.strategy_manager).__name__)

                if not available_core_types:
                    st.error(translator.t('error_no_core_types'))
                    # Use st.form_submit_button to prevent the form from submitting
                    st.form_submit_button("💾 " + translator.t('save_strategy_button'), disabled=True)
                    return  # Stop rendering the rest of the form

                core_type_options = {}
                if "ml_model" in available_core_types:
                    core_type_options["ml_model"] = translator.t('core_type_ml')
                if "llm_trader" in available_core_types:
                    core_type_options["llm_trader"] = translator.t('core_type_llm')

                core_type = st.selectbox(
                    translator.t('select_core_type_label'),
                    options=list(core_type_options.keys()),
                    format_func=lambda x: core_type_options[x],
                    key="strat_core_type_form"
                )

                # --- Section 3: Dynamic parameters based on core type ---
                params = {}
                if core_type == "ml_model":
                    st.markdown(f"**{translator.t('ml_model_config_header')}**")
                    params['ml_model_name'] = st.selectbox(translator.t('select_ml_model'), options=available_models,
                                                           key="strat_ml_model_form")
                    params['confidence_threshold'] = st.slider(translator.t('ml_confidence_threshold_trade'), 0.50,
                                                               0.99, 0.65, 0.01, key="strat_ml_thresh_form")
                elif core_type == "llm_trader":
                    st.markdown(f"**{translator.t('llm_trader_config_header')}**")
                    params['llm_name'] = st.selectbox(translator.t('select_llm_trader_label'), options=available_llms,
                                                      key="strat_llm_name_form")
                    params['confidence_threshold'] = st.slider(translator.t('llm_confidence_threshold_trade'), 0.50,
                                                               0.99, 0.70, 0.01, key="strat_llm_thresh_form")

                # --- Section 4: Trade parameters ---
                st.markdown(f"**{translator.t('trade_risk_params_label')}**")
                params['trade_quantity'] = st.number_input(translator.t('trade_quantity_ml'), min_value=1, value=10,
                                                           step=1, key="strat_quantity_form")

                # --- Submit button ---
                submitted = st.form_submit_button("💾 " + translator.t('save_strategy_button'))

                if submitted:
                    self._handle_form_submission(
                        strategy_name=strategy_name,
                        symbols_input=symbols_input,
                        core_type=core_type,
                        params=params
                    )

    def _handle_form_submission(self, strategy_name, symbols_input, core_type, params):
        """
        Centralized logic for form submission with enhanced debugging.
        """
        symbols_list = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
        if not strategy_name or not symbols_list:
            st.error(translator.t('error_strategy_name_symbol_required'))
            return

        if core_type == "llm_trader":
            if not params.get('llm_name'):
                st.error(translator.t('error_llm_not_selected'))
                return
        if core_type == "ml_model" and not params.get('ml_model_name'):
            # This is a new key that was missing before
            st.error(translator.t('error_ml_model_not_selected', fallback="Please select a Machine Learning model."))
            return

        user_id = st.session_state.get('username', 'Guest')
        symbols_id_part = "_".join(symbols_list).replace('.', '_')
        strategy_id = f"auto_{user_id}_{symbols_id_part}_{strategy_name.replace(' ', '_')}"

        config = {
            "strategy_id": strategy_id, "user_id": user_id, "name": strategy_name,
            "symbols": symbols_list, "core_type": core_type, "enabled": True, **params
        }
        try:
            self.persistence_manager.save_strategy_config(config)
            st.success(translator.t('strategy_saved_success').format(name=strategy_name))
            st.rerun()
            # No st.rerun() needed here because of `clear_on_submit=True`
        except Exception as e:
            st.error(translator.t('save_strategy_error_generic', fallback="Failed to save strategy: {e}").format(e=e))

    def _render_saved_strategies(self):
        """
        [FIXED] Renders the list of saved strategies with improved feedback for errors.
        """
        user_id = st.session_state.get('username', 'Guest')
        all_strategies = self.persistence_manager.load_all_strategies_for_user(user_id)

        if not all_strategies:
            st.info(translator.t('no_saved_strategies_info'))
            return

        cols = st.columns(2)
        for i, strat in enumerate(all_strategies):
            with cols[i % 2]:
                with st.container(border=True):
                    strat_id = strat['strategy_id']
                    symbols_display = ", ".join(strat.get('symbols', [strat.get('symbol', 'N/A')]))
                    st.markdown(f"**{strat.get('name')}** (`{symbols_display}`)")

                    core_type = strat.get('core_type', 'N/A')
                    if core_type == 'ml_model':
                        st.caption(f"核心: ML - {strat.get('ml_model_name', 'N/A')}")
                    elif core_type == 'llm_trader':
                        st.caption(f"核心: LLM - {strat.get('llm_name', 'N/A')}")

                    c1, c2 = st.columns([3, 2])
                    with c1:
                        is_enabled = strat.get('enabled', False)
                        label = "🟢 " + translator.t('running') if is_enabled else "⏸️ " + translator.t('paused')
                        # The key is already unique, which is good.
                        new_enabled = st.toggle(label, value=is_enabled, key=f"toggle_{strat_id}")
                        if new_enabled != is_enabled:
                            strat['enabled'] = new_enabled
                            self.persistence_manager.save_strategy_config(strat)
                            st.rerun()
                    with c2:
                        # Give the button a similarly unique key.
                        if st.button("🗑️ " + translator.t('delete'), key=f"delete_btn_{strat_id}", type="secondary",
                                     use_container_width=True):
                            self.persistence_manager.delete_strategy_config(strat_id, user_id)
                            # IMPORTANT: No need to pop from session state, the rerun will handle it.
                            st.rerun()

                    with st.container(border=True):
                        last_exec_str = strat.get('last_executed')
                        last_exec_dt = datetime.fromisoformat(last_exec_str.split('.')[0]) if last_exec_str else None
                        if last_exec_dt:
                            time_ago = datetime.now() - last_exec_dt
                            st.markdown(f"**上次心跳:** {int(time_ago.total_seconds())} 秒前")
                        else:
                            st.markdown(f"**上次心跳:** {translator.t('pending_execution')}")

                        # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
                        last_decision = strat.get("last_decision")
                        if last_decision and "error" in last_decision:
                            st.error(f"**最近决策:** 失败 - {last_decision['error'][:100]}...")  # Show error message
                        elif last_decision and "decision" in last_decision:
                            # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^
                            decision = last_decision.get('decision', 'N/A')
                            confidence = last_decision.get('confidence', 0)
                            color = "green" if decision == "BUY" else "red" if decision == "SELL" else "gray"
                            st.markdown(
                                f"**最近决策:** <span style='color:{color}; font-weight:bold;'>{decision}</span> (信度: {confidence:.2f})",
                                unsafe_allow_html=True)
                            if "reasoning" in last_decision and last_decision["reasoning"]:
                                with st.popover("🧠 " + translator.t('view_reasoning_popover')):
                                    time_str = last_exec_dt.strftime('%H:%M:%S') if last_exec_dt else "N/A"
                                    st.markdown(f"##### 决策理由 - {time_str}")
                                    st.markdown(last_decision["reasoning"])
                        else:
                            st.markdown(f"**最近决策:** {translator.t('awaiting_first_decision')}")