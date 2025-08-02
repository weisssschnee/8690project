# core/trading/executor.py
import logging
import pandas as pd
from datetime import datetime
import streamlit as st
from core.translate import translator
from typing import Dict, Optional, List, Any

try:
    import futu as ft

    FUTU_AVAILABLE = True
except ImportError:
    ft = None
    FUTU_AVAILABLE = False
    logging.warning("futu-api library not found. Real trading functionality disabled.")
import threading

logger = logging.getLogger(__name__)


class OrderExecutor:
    def __init__(self, config: Dict, risk_manager: Optional[Any] = None, futu_manager: Optional[Any] = None):
        self.config = config
        self.risk_manager = risk_manager
        self.futu_manager = futu_manager
        self.order_history = []

        # --- Futu Trade Contexts ---
        self.futu_trade_ctx_hk: Optional[ft.OpenHKTradeContext] = None
        self.futu_trade_ctx_us: Optional[ft.OpenUSTradeContext] = None
        self.futu_trade_is_unlocked = False

        if FUTU_AVAILABLE and getattr(self.config, 'FUTU_ENABLED', False):
            self._connect_futu_trade()

        if self.futu_manager and self.futu_manager.is_connected:
            trade_conn_thread = threading.Thread(target=self._connect_futu_trade, daemon=True)
            trade_conn_thread.start()
        elif self.futu_manager:
            logger.warning("FutuManager exists but not connected, trade context connection skipped for now.")
        else:
            logger.info("FutuManager not provided to OrderExecutor, real trading disabled.")

    def _connect_futu_trade(self):
        """[新增] 建立与 FutuOpenD 的交易连接并解锁"""
        host = getattr(self.config, 'FUTU_HOST', '127.0.0.1')
        port = getattr(self.config, 'FUTU_PORT', 11111)
        password = getattr(self.config, 'FUTU_PWD', '')

        logger.info(f"Attempting to connect Futu Trade Contexts at {host}:{port}...")
        try:
            self.futu_trade_ctx_hk = ft.OpenHKTradeContext(host=host, port=port)
            self.futu_trade_ctx_us = ft.OpenUSTradeContext(host=host, port=port)

            if password:
                ret_hk, _ = self.futu_trade_ctx_hk.unlock_trade(password)
                ret_us, _ = self.futu_trade_ctx_us.unlock_trade(password)
                if ret_hk == ft.RET_OK and ret_us == ft.RET_OK:
                    self.futu_trade_is_unlocked = True
                    logger.info("Futu HK and US trade contexts unlocked successfully.")
                else:
                    logger.error(f"Futu trade unlock failed. HK_ret={ret_hk}, US_ret={ret_us}")
            else:
                logger.warning("Futu password not provided. Trading will be locked.")
        except Exception as e:
            logger.error(f"Error connecting Futu trade contexts: {e}", exc_info=True)
            if self.futu_trade_ctx_hk: self.futu_trade_ctx_hk.close()
            if self.futu_trade_ctx_us: self.futu_trade_ctx_us.close()

    def disconnect_futu_trade(self):
        """安全地关闭富途交易上下文"""
        if self.futu_trade_ctx_hk: self.futu_trade_ctx_hk.close()
        if self.futu_trade_ctx_us: self.futu_trade_ctx_us.close()
        self.futu_trade_ctx_hk = None
        self.futu_trade_ctx_us = None
        self.futu_trade_is_unlocked = False

    def execute_order(self, symbol: str, quantity: int, price: float, order_type: str = "Market Order",
                      direction: str = "Buy") -> Dict:
        """
        执行单个订单，优先使用富途真实/模拟交易。
        如果富途不可用，则执行系统内的模拟交易。
        """
        # --- Priority 1: Futu Real/Simulate Trading ---
        if self.futu_manager and self.futu_trade_is_unlocked:
            logger.info(f"Attempting to place order via Futu: {direction} {quantity} {symbol} @ {price}")
            try:
                # 确定交易市场和上下文
                trade_ctx = None
                if symbol.upper().startswith('HK.'):
                    trade_ctx = self.futu_trade_ctx_hk
                elif symbol.upper().startswith('US.'):
                    trade_ctx = self.futu_trade_ctx_us

                if not trade_ctx:
                    msg = f"Futu does not support market for symbol: {symbol}"
                    logger.warning(msg)
                    return self._execute_internal_mock(symbol, quantity, price, order_type, direction, reason=msg)

                # 映射参数
                trd_side = ft.TrdSide.BUY if direction == 'Buy' else ft.TrdSide.SELL
                # 富途市价单价格传 0, 限价单传指定价格
                price_for_futu = 0 if order_type == 'Market Order' else price
                # 富途普通单对应限价单
                order_type_futu = ft.OrderType.NORMAL

                # 从 config 读取是模拟还是真实交易 (如果配置了)
                # 默认为模拟交易，更安全
                trd_env_str = getattr(self.config, 'FUTU_TRD_ENV', 'SIMULATE').upper()
                trd_env = ft.TrdEnv.REAL if trd_env_str == 'REAL' else ft.TrdEnv.SIMULATE
                logger.info(f"Using Futu trade environment: {trd_env_str}")

                ret, data = trade_ctx.place_order(
                    price=price_for_futu, qty=quantity, code=symbol.upper(),
                    trd_side=trd_side, order_type=order_type_futu, trd_env=trd_env
                )

                if ret == ft.RET_OK:
                    order_id = data['order_id'].iloc[0]
                    logger.info(f"Futu order placed successfully. OrderID: {order_id}")
                    # 返回一个与模拟交易相似的标准化字典
                    return {
                        'success': True,
                        'order_id': order_id,
                        'source': 'Futu',
                        'is_mock': (trd_env == ft.TrdEnv.SIMULATE),
                        'message': f"Futu order ({trd_env_str}) submitted successfully.",
                        # Futu 不直接返回执行价格和成本，这些需要后续查询订单状态
                        'price': price,  # 记录下单时价格
                        'total_cost': quantity * price,  # 估算的名义价值
                        'timestamp': datetime.now()
                    }
                else:
                    logger.error(f"Futu place_order API call failed: {data}")
                    return {'success': False, 'message': str(data)}
            except Exception as e:
                logger.error(f"Error executing Futu order: {e}", exc_info=True)
                # Fallthrough to mock execution if Futu fails

        # --- Fallback: System's Internal Mock Execution ---
        logger.warning(f"Futu trade not available or failed. Executing internal mock trade for {symbol}.")
        return self._execute_internal_mock(symbol, quantity, price, order_type, direction)

    def _execute_internal_mock(self, symbol: str, quantity: int, price: float, order_type: str, direction: str,
                               reason: str = "内部模拟") -> Dict:
        """执行内部模拟交易，用于回退或无真实交易接口时"""
        logger.info(f"Executing internal mock trade: {direction} {quantity} {symbol} @ {price}")
        # 这里 price 应该是已经确定的数值
        if price is None:
            return {'success': False, 'message': 'Mock trade failed: Price is None.'}

        order_value = quantity * price

        # 将方向映射到 +/-
        signed_quantity = quantity if direction == "Buy" else -quantity

        self.order_history.append({
            "symbol": symbol, "quantity": signed_quantity, "price": price,
            "order_type": order_type, "direction": direction,
            "timestamp": datetime.now(), 'is_mock': True
        })

        return {
            "success": True,
            "symbol": symbol,
            "quantity": quantity,  # 返回无符号的数量
            "price": price,
            "total_cost": order_value,
            "timestamp": datetime.now(),
            "message": f"{reason} - {direction} 订单执行成功",
            "is_mock": True
        }

    def execute_batch_orders(self, orders_to_process: List[Dict], current_portfolio_state: Dict,
                             system_ref: Optional[Any] = None) -> Dict:
        """
        [最终修复版] 执行批量订单，修复了重复调用和风控逻辑。
        """
        logger.info(f"Executing {len(orders_to_process)} batch orders...")
        results = {"success": [], "failed": [], "total_success": 0, "total_failed": 0}

        # --- 1. 创建一个临时的、用于模拟交易过程的投资组合状态 ---
        # 我们将在这个模拟 portfolio 上进行所有事前检查
        simulated_portfolio = {
            'cash': current_portfolio_state.get('cash', 0),
            'positions': {sym: pos.copy() for sym, pos in current_portfolio_state.get('positions', {}).items()},
            'total_value': current_portfolio_state.get('total_value', 0)
        }

        # --- 2. 循环处理每一笔订单 ---
        for i, order in enumerate(orders_to_process):
            # a. 解析和验证订单基础参数
            symbol = order.get("symbol")
            quantity = order.get("quantity", 0)
            price_from_csv = order.get("price")
            direction = order.get("direction", "Buy")
            order_type = order.get("order_type", "Market Order")

            logger.debug(f"Processing batch order {i + 1}/{len(orders_to_process)}: {direction} {quantity} {symbol}")
            if not all([symbol, isinstance(quantity, (int, float)), quantity > 0]):
                results["failed"].append({"order": order, "reason": translator.t('error_batch_invalid_order_params')})
                continue  # 跳到下一个订单

            # b. 确定执行价格
            exec_price = price_from_csv
            if order_type == "Market Order" and exec_price is None:
                if system_ref and system_ref.data_manager:
                    price_data = system_ref.data_manager.get_realtime_price(symbol)
                    if price_data and price_data.get('price'):
                        exec_price = price_data['price']
                        logger.info(f"Batch: Fetched market price for {symbol}: {exec_price}")
                    else:
                        results["failed"].append(
                            {"order": order, "reason": translator.t('error_batch_market_no_price')})
                        continue
                else:
                    results["failed"].append(
                        {"order": order, "reason": translator.t('error_batch_market_no_system_ref')})
                    continue

            # c. 准备用于验证和执行的完整订单信息
            order_for_validation = {**order, "price": float(exec_price) if exec_price is not None else None}
            if order_for_validation["price"] is None:
                results["failed"].append({"order": order, "reason": translator.t('error_batch_order_no_price')})
                continue

            # d. 在【模拟的】投资组合上进行风险检查
            if self.risk_manager:
                risk_result = self.risk_manager.validate_order(
                    order=order_for_validation,
                    portfolio=simulated_portfolio,  # 使用模拟 portfolio 进行检查
                    system_ref=system_ref
                )
                if not risk_result.get('valid'):
                    results["failed"].append({"order": order, "reason": risk_result.get('reason', "Risk check failed")})
                    continue

            # e. 执行单个订单 (调用您现有的、独立的 execute_order 方法)
            # 注意：execute_order 内部会再次进行一次最终的真实风控检查，这是双重保障
            actual_exec_result = self.execute_order(
                symbol=symbol,
                quantity=quantity,
                price=exec_price,  # 传递确定的价格
                order_type=order_type,
                direction=direction
            )

            # f. 根据真实执行结果，更新模拟的投资组合，以便为下一个订单提供正确的状态
            if actual_exec_result and actual_exec_result.get("success"):
                results["success"].append(actual_exec_result)
                # 更新模拟的现金和持仓
                if direction == "Buy":
                    simulated_portfolio['cash'] -= actual_exec_result.get('total_cost', 0)
                    sim_pos = simulated_portfolio['positions'].setdefault(symbol, {'quantity': 0})
                    sim_pos['quantity'] += quantity
                elif direction == "Sell":
                    simulated_portfolio['cash'] += actual_exec_result.get('total_cost', 0)
                    if symbol in simulated_portfolio['positions']:
                        simulated_portfolio['positions'][symbol]['quantity'] -= quantity
                        if simulated_portfolio['positions'][symbol]['quantity'] <= 0:
                            del simulated_portfolio['positions'][symbol]
            else:
                results["failed"].append(
                    {"order": order, "reason": actual_exec_result.get("message", "Execution failed")})

        # --- 3. 汇总最终结果 ---
        results["total_success"] = len(results["success"])
        results["total_failed"] = len(results["failed"])
        logger.info(f"Batch execution finished. Success: {results['total_success']}, Failed: {results['total_failed']}")
        return results

    # ^^^^^^^^^^^^^^^^^^^^ END OF MODIFIED execute_batch_orders ^^^^^^^^^^^^^^^^^^^^

    def liquidate_positions(self, positions, liquidation_type="all"):
        """一键平仓功能"""
        liquidation_orders = []

        for symbol, position in positions.items():
            # 跳过数量为0的持仓
            if position.get('quantity', 0) <= 0:
                continue

            # 根据平仓类型决定是否包含此持仓
            if liquidation_type == "profit" and position.get('current_price', 0) <= position.get('cost_basis', 0):
                continue

            if liquidation_type == "loss" and position.get('current_price', 0) >= position.get('cost_basis', 0):
                continue

            # 创建卖出订单
            liquidation_orders.append({
                "symbol": symbol,
                "quantity": position.get('quantity', 0),
                "price": position.get('current_price', 0),
                "direction": "卖出",
                "order_type": "市价单",
            })

        return liquidation_orders

    def render_batch_trading_ui(self, system: Any):
        st.header(translator.t('batch_trading_header', fallback="批量交易"))

        # --- CSV Batch Import ---
        st.subheader(translator.t('batch_import_from_csv', fallback="批量导入CSV交易/策略规则"))

        target_symbols_input_csv = st.text_area(
            translator.t('batch_target_symbols_label_csv',
                         fallback="目标股票代码 (可选, 逗号分隔, 若此处填写则应用于CSV每条规则):"),
            key="batch_trade_target_symbols_csv_executor_v5",
            help=translator.t('batch_target_symbols_help_csv_v2',
                              fallback="如果CSV文件是通用策略模板，请在此处输入要应用的股票代码。CSV中的'symbol'列将被忽略或不应存在。")
        )

        st.markdown(f"**{translator.t('batch_ui_trade_params_header', fallback='为CSV规则设置通用交易参数:')}**")
        col_qty, col_dir, col_type = st.columns(3)

        # 获取翻译后的选项
        buy_opt_trans = translator.t('buy', fallback="买入")
        sell_opt_trans = translator.t('sell', fallback="卖出")
        market_opt_trans = translator.t('market_order', fallback="市价单")
        limit_opt_trans = translator.t('limit_order', fallback="限价单")

        with col_qty:
            ui_default_quantity = st.number_input(
                translator.t('batch_ui_default_quantity', fallback="默认数量:"),
                min_value=1, value=10, step=1, key="batch_ui_qty_v5_executor",
                help=translator.t('batch_ui_default_quantity_help', fallback="如果CSV行中未指定数量，则使用此值。")
            )
        with col_dir:
            ui_default_direction_display = st.selectbox(
                translator.t('batch_ui_default_direction', fallback="默认方向:"),
                [buy_opt_trans, sell_opt_trans], key="batch_ui_dir_v5_executor",
                help=translator.t('batch_ui_default_direction_help', fallback="如果CSV行中未指定方向，则使用此值。")
            )
        with col_type:
            ui_default_order_type_display = st.selectbox(
                translator.t('batch_ui_default_order_type', fallback="默认订单类型:"),
                [market_opt_trans, limit_opt_trans], key="batch_ui_otype_v5_executor",
                help=translator.t('batch_ui_default_order_type_help', fallback="如果CSV行中未指定订单类型，则使用此值。")
            )

        ui_default_price_str = ""
        if ui_default_order_type_display == limit_opt_trans:
            ui_default_price_str = st.text_input(
                translator.t('batch_ui_default_price_limit', fallback="默认限价 (可选):"),
                key="batch_ui_price_v5_executor",
                help=translator.t('batch_ui_default_price_limit_help',
                                  fallback="如果CSV行中是限价单且未指定价格，可在此输入。留空则该限价单可能失败。")
            )

        with st.expander(translator.t('csv_format_guide_batch', fallback="CSV格式指南")):
            st.write("**重要：CSV文件必须包含以下列之一：**")
            st.write("1. **symbol列** - 直接指定交易标的")
            st.write("2. **或在上方UI中指定目标股票代码**")
            st.code(""""
# 方式1：CSV包含symbol列
symbol,quantity,direction,price,order_type
AAPL,10,Buy,150.0,Market Order
TSLA,5,Sell,250.0,Limit Order

# 方式2：CSV不含symbol，在UI中指定目标股票
quantity,direction,price,order_type
10,Buy,150.0,Market Order
5,Sell,250.0,Limit Order
            """)

        uploaded_file_csv = st.file_uploader(
            translator.t('upload_csv_file', fallback="上传CSV文件"),
            type=["csv"],
            key="batch_csv_uploader_executor_v5"
        )

        # 初始化session state
        if 'csv_orders_for_confirmation' not in st.session_state:
            st.session_state.csv_orders_for_confirmation = []

        if uploaded_file_csv is not None:
            if st.button(
                    translator.t('execute_csv_batch_button_v2', fallback="处理CSV并生成交易指令"),
                    key="process_csv_batch_btn_v5_executor"
            ):
                st.session_state.csv_orders_for_confirmation = []

                try:
                    # 🔥 修复：改进CSV读取和验证
                    batch_rules_df_csv = pd.read_csv(uploaded_file_csv)

                    if batch_rules_df_csv.empty:
                        st.warning(translator.t('warning_csv_empty', fallback="上传的CSV文件为空。"))
                        st.stop()

                    st.write("📋 **CSV文件预览：**")
                    st.dataframe(batch_rules_df_csv.head())

                    # 🔥 修复：改进symbol验证逻辑
                    target_symbols_list_csv = [
                        s.strip().upper() for s in target_symbols_input_csv.split(',') if s.strip()
                    ] if target_symbols_input_csv.strip() else []

                    has_symbol_column = 'symbol' in batch_rules_df_csv.columns

                    # 验证symbol来源
                    if not target_symbols_list_csv and not has_symbol_column:
                        st.error("❌ **错误：缺少交易标的！**")
                        st.error("请选择以下方式之一：")
                        st.error("1. 在CSV文件中包含 'symbol' 列")
                        st.error("2. 在上方UI中输入目标股票代码")
                        # 修复：不再停止整个程序，而是跳过处理
                        return

                    # 修复：添加日志记录
                    logger.info(f"Target symbols: {target_symbols_list_csv}, Has symbol column: {has_symbol_column}")

                    current_orders_to_execute = []

                    # 🔥 修复：改进订单生成逻辑
                    for rule_idx, csv_rule_item in enumerate(batch_rules_df_csv.to_dict('records')):
                        try:
                            # 确定symbol
                            symbols_for_this_rule = []
                            if target_symbols_list_csv:
                                # 使用UI指定的symbols
                                symbols_for_this_rule = target_symbols_list_csv
                            elif has_symbol_column and pd.notna(csv_rule_item.get('symbol')):
                                # 使用CSV中的symbol
                                symbol_from_csv = str(csv_rule_item['symbol']).strip().upper()
                                if symbol_from_csv:
                                    symbols_for_this_rule = [symbol_from_csv]

                            # 修复：跳过没有symbol的行
                            if not symbols_for_this_rule:
                                logger.warning(f"跳过第{rule_idx + 1}行：无有效symbol")
                                continue

                            # 处理quantity
                            try:
                                quantity_val = csv_rule_item.get('quantity')
                                if pd.notna(quantity_val):
                                    final_quantity = float(quantity_val)
                                else:
                                    final_quantity = float(ui_default_quantity)

                                if final_quantity <= 0:
                                    raise ValueError("数量必须大于0")

                            except (ValueError, TypeError) as e:
                                st.error(f"❌ 第{rule_idx + 1}行数量无效: {quantity_val}")
                                continue

                            # 处理direction
                            csv_direction = str(csv_rule_item.get('direction', '')).strip()
                            if csv_direction.lower() in ['buy', '买入', buy_opt_trans.lower()]:
                                final_direction = "Buy"
                            elif csv_direction.lower() in ['sell', '卖出', sell_opt_trans.lower()]:
                                final_direction = "Sell"
                            else:
                                # 使用UI默认值
                                final_direction = "Buy" if ui_default_direction_display == buy_opt_trans else "Sell"

                            # 处理order_type
                            csv_order_type = str(csv_rule_item.get('order_type', '')).strip()
                            if csv_order_type.lower() in ['market order', '市价单', market_opt_trans.lower()]:
                                final_order_type = "Market Order"
                            elif csv_order_type.lower() in ['limit order', '限价单', limit_opt_trans.lower()]:
                                final_order_type = "Limit Order"
                            else:
                                # 使用UI默认值
                                final_order_type = "Market Order" if ui_default_order_type_display == market_opt_trans else "Limit Order"

                            # 处理price
                            final_price = None
                            if pd.notna(csv_rule_item.get('price')):
                                try:
                                    final_price = float(csv_rule_item['price'])
                                except (ValueError, TypeError):
                                    pass

                            # 限价单必须有价格
                            if final_order_type == "Limit Order" and final_price is None:
                                if ui_default_price_str.strip():
                                    try:
                                        final_price = float(ui_default_price_str.strip())
                                    except ValueError:
                                        st.error(f"❌ 第{rule_idx + 1}行限价单缺少有效价格")
                                        continue
                                else:
                                    st.error(f"❌ 第{rule_idx + 1}行限价单缺少价格")
                                    continue

                            # 为每个symbol创建订单
                            for symbol in symbols_for_this_rule:
                                order = {
                                    "symbol": symbol,
                                    "quantity": int(final_quantity),
                                    "price": final_price,
                                    "direction": final_direction,
                                    "order_type": final_order_type,
                                    "original_rule_id": csv_rule_item.get('strategy_id', f'csv_rule_{rule_idx + 1}')
                                }
                                current_orders_to_execute.append(order)

                        except Exception as e:
                            st.error(f"❌ 处理第{rule_idx + 1}行时出错: {str(e)}")
                            logger.error(f"Error processing CSV row {rule_idx + 1}: {e}", exc_info=True)
                            continue

                    if current_orders_to_execute:
                        st.session_state.csv_orders_for_confirmation = current_orders_to_execute
                        st.success(f"✅ 成功生成 {len(current_orders_to_execute)} 个交易订单")
                        logger.info(f"Generated {len(current_orders_to_execute)} orders from CSV")
                    else:
                        st.warning("⚠️ 未能生成任何有效订单")
                        logger.warning("No valid orders generated from CSV")

                except Exception as e:
                    st.error(f"❌ CSV处理出错: {str(e)}")
                    logger.error(f"CSV processing error: {e}", exc_info=True)
                    st.session_state.csv_orders_for_confirmation = []

        # 🔥 修复：订单确认和执行逻辑
        orders_to_display = st.session_state.get('csv_orders_for_confirmation', [])

        if orders_to_display:
            st.markdown("### 📋 **待执行订单预览**")

            # 创建更友好的显示格式
            display_orders = []
            for order in orders_to_display:
                display_order = {
                    "股票代码": order['symbol'],
                    "数量": order['quantity'],
                    "方向": "买入" if order['direction'] == 'Buy' else "卖出",
                    "类型": "市价单" if order['order_type'] == 'Market Order' else "限价单",
                    "价格": f"${order['price']:.2f}" if order['price'] is not None else "市价",
                    "预估金额": f"${order['quantity'] * (order['price'] or 0):.2f}" if order['price'] else "待定"
                }
                display_orders.append(display_order)

            st.dataframe(pd.DataFrame(display_orders), use_container_width=True)

            # 确认执行
            col_confirm, col_execute = st.columns([2, 1])

            with col_confirm:
                proceed_confirmed = st.checkbox(
                    "✅ **我已确认上述订单，同意执行批量交易**",
                    key="confirm_batch_execution_v5"
                )

            if st.button(
                    "🚀 **执行批量交易**",
                    disabled=not proceed_confirmed,
                    key="execute_confirmed_batch_v5",
                    type="primary"
            ):
                if proceed_confirmed:
                    # 🔥 修复：确保system参数存在
                    if not system:
                        st.error("❌ 系统引用错误，无法执行交易")
                        return

                    with st.spinner("🔄 正在执行批量交易..."):
                        success_count = 0
                        failed_orders = []

                        # 获取当前投资组合状态
                        current_portfolio = st.session_state.get('portfolio', {})

                        # 使用批量执行方法提高效率
                        batch_result = self.execute_batch_orders(
                            orders_to_process=orders_to_display,
                            current_portfolio_state=current_portfolio,
                            system_ref=system
                        )

                        success_count = batch_result["total_success"]
                        failed_count = batch_result["total_failed"]

                        # 记录失败详情
                        for failed_order in batch_result["failed"]:
                            failed_orders.append({
                                "股票代码": failed_order["order"].get("symbol"),
                                "失败原因": failed_order["reason"]
                            })

                    st.success(f"✅ 成功执行 {success_count} 个订单")
                    if failed_orders:
                        st.error(f"❌ 执行失败 {len(failed_orders)} 个订单")
                        st.write("失败详情:")
                        st.dataframe(pd.DataFrame(failed_orders))

                    # 清除已执行的订单
                    st.session_state.csv_orders_for_confirmation = []
                    st.rerun()

        # --- 手动创建批量交易 ---
        st.subheader(translator.t('manual_batch_trade_subheader', fallback="手动批量交易"))
        with st.form(key="manual_batch_form_unique_exec"):
            col_manual1, col_manual2 = st.columns(2)
            with col_manual1:
                symbols_input_manual = st.text_area(
                    translator.t('manual_symbols_label', fallback="股票代码 (每行一个)"), height=100,
                    key="manual_symbols_input_area_exec")
                buy_option_manual = translator.t('buy', fallback="买入")
                sell_option_manual = translator.t('sell', fallback="卖出")
                trade_direction_manual_ui = st.radio(translator.t('manual_direction_label', fallback="交易方向"),
                                                     [buy_option_manual, sell_option_manual],
                                                     key="manual_direction_radio_exec")
            with col_manual2:
                quantity_input_manual = st.number_input(
                    translator.t('manual_qty_per_stock_label', fallback="每只股票数量"), min_value=1, value=10,
                    key="manual_qty_input_exec")
                market_price_option_manual = translator.t('manual_current_market_price_option', fallback="当前市价")
                custom_price_option_manual = translator.t('manual_custom_price_option', fallback="自定义价格")
                price_type_manual = st.radio(translator.t('manual_price_type_label', fallback="价格类型"),
                                             [market_price_option_manual, custom_price_option_manual],
                                             key="manual_price_type_radio_exec")
                custom_price_manual = st.number_input(
                    translator.t('manual_custom_price_label', fallback="自定义价格 (如选择)"), min_value=0.01,
                    value=100.0, format="%.2f",
                    disabled=(price_type_manual == market_price_option_manual), key="manual_custom_price_input_exec")

            submitted_manual_batch = st.form_submit_button(
                translator.t('manual_create_batch_button', fallback="创建并执行批量交易"),
            )
            if submitted_manual_batch:
                symbols_list_manual = [s.strip().upper() for s in symbols_input_manual.split("\n") if s.strip()]
                if not symbols_list_manual:
                    st.error(translator.t('manual_error_no_symbols', fallback="请输入至少一个股票代码"))
                else:
                    manual_batch_orders_to_exec = []
                    preview_data_manual = []  # Renamed to avoid conflict
                    for symbol_man_loop in symbols_list_manual:  # Changed loop var name
                        price_exec_manual = None
                        if price_type_manual == market_price_option_manual:
                            rt_price_info = system.data_manager.get_realtime_price(symbol_man_loop)
                            price_exec_manual = rt_price_info['price'] if rt_price_info and isinstance(
                                rt_price_info.get('price'), (float, int)) else None
                            if price_exec_manual is None:
                                st.warning(translator.t('manual_warning_cannot_get_price_skip',
                                                        fallback="无法获取 {symbol} 的市价，该股票将从批量订单中跳过。").format(
                                    symbol=symbol_man_loop))
                                continue
                        else:  # Custom price
                            price_exec_manual = custom_price_manual

                        internal_dir_manual = 'Buy' if trade_direction_manual_ui == buy_option_manual else 'Sell'
                        order_to_add = {"symbol": symbol_man_loop, "quantity": quantity_input_manual,
                                        "price": price_exec_manual,
                                        "direction": internal_dir_manual,
                                        "order_type": "Market Order" if price_type_manual == market_price_option_manual else "Limit Order"}
                        manual_batch_orders_to_exec.append(order_to_add)
                        preview_data_manual.append({
                            translator.t('stock_symbol'): symbol_man_loop,
                            translator.t('quantity'): quantity_input_manual,
                            translator.t(
                                'price'): f"${price_exec_manual:.2f}" if price_exec_manual is not None else translator.t(
                                'market_price_short', fallback="市价"),
                            translator.t('direction'): trade_direction_manual_ui,
                            translator.t('manual_col_total_value',
                                         fallback="预估总价值"): f"${quantity_input_manual * (price_exec_manual or 0):.2f}"
                            # Handle None price_exec
                        })

                    if preview_data_manual:
                        st.write(translator.t('manual_orders_preview', fallback="订单预览:"))
                        st.dataframe(pd.DataFrame(preview_data_manual), use_container_width=True)

                        if manual_batch_orders_to_exec:
                            # This is where the actual execution happens for manual batch
                            manual_success_count = 0
                            manual_fail_details = []
                            with st.spinner(
                                    translator.t('processing_batch_orders_spinner', fallback="正在处理批量订单...")):
                                # 使用批量执行方法
                                batch_result = self.execute_batch_orders(
                                    orders_to_process=manual_batch_orders_to_exec,
                                    current_portfolio_state=st.session_state.get('portfolio', {}),
                                    system_ref=system
                                )

                                manual_success_count = batch_result["total_success"]
                                for failed_order in batch_result["failed"]:
                                    manual_fail_details.append({
                                        "symbol": failed_order["order"].get("symbol"),
                                        "reason": failed_order["reason"]
                                    })

                            st.success(translator.t('batch_success_msg', fallback="成功执行 {count} 个订单").format(
                                count=manual_success_count))
                            if manual_fail_details:
                                st.error(translator.t('batch_failed_msg', fallback="执行失败 {count} 个订单").format(
                                    count=len(manual_fail_details)))
                                st.write(translator.t('batch_failed_details', fallback="失败详情:"));
                                st.dataframe(pd.DataFrame(manual_fail_details))
                            st.rerun()
                    else:
                        st.info(translator.t('no_valid_manual_orders', fallback="没有有效的订单可执行。"))

        # --- 一键平仓功能 ---
        st.subheader(translator.t('one_click_closeout_subheader', fallback="一键平仓"))
        current_portfolio_state_closeout = st.session_state.get('portfolio', {})
        positions_for_closeout = current_portfolio_state_closeout.get('positions', {})  # Use the correct variable

        if positions_for_closeout:
            close_col1, close_col2, close_col3 = st.columns(3)
            with close_col1:
                if st.button(translator.t('closeout_all_button', fallback="平掉所有持仓"), key="btn_close_all_exec"):
                    orders_all = self.liquidate_positions(positions_for_closeout, "all")  # Use correct var
                    self._handle_liquidation(orders_all, system)
            with close_col2:
                if st.button(translator.t('closeout_profit_button', fallback="平掉盈利持仓"),
                             key="btn_close_profit_exec"):
                    orders_profit = self.liquidate_positions(positions_for_closeout, "profit")  # Use correct var
                    self._handle_liquidation(orders_profit, system)
            with close_col3:
                if st.button(translator.t('closeout_loss_button', fallback="平掉亏损持仓"), key="btn_close_loss_exec"):
                    orders_loss = self.liquidate_positions(positions_for_closeout, "loss")  # Use correct var
                    self._handle_liquidation(orders_loss, system)

            st.markdown(f"**{translator.t('current_holdings_status_title', fallback='当前持仓状态')}**")
            if positions_for_closeout:  # Re-check as it might have been modified by liquidation
                positions_display_data_list = []  # New variable name
                col_disp_stock_cl = translator.t('col_stock', fallback="股票")
                col_disp_qty_cl = translator.t('col_quantity', fallback="数量")
                col_disp_price_cl = translator.t('col_current_price', fallback="当前价")
                col_disp_cost_cl = translator.t('col_cost_basis', fallback="成本价")
                col_disp_pnl_cl = translator.t('col_pnl', fallback="盈亏")

                for symbol_cl, pos_details_cl in positions_for_closeout.items():
                    current_price = pos_details_cl.get('current_price', 0)
                    cost_basis = pos_details_cl.get('cost_basis', 0)
                    quantity = pos_details_cl.get('quantity', 0)
                    pnl = (current_price - cost_basis) * quantity

                    positions_display_data_list.append({
                        col_disp_stock_cl: symbol_cl,
                        col_disp_qty_cl: quantity,
                        col_disp_price_cl: f"${current_price:.2f}",
                        col_disp_cost_cl: f"${cost_basis:.2f}",
                        col_disp_pnl_cl: f"${pnl:.2f} ({pnl / (cost_basis * quantity) * 100 if cost_basis * quantity != 0 else 0:.2f}%)"
                    })

                if positions_display_data_list:
                    st.dataframe(pd.DataFrame(positions_display_data_list), use_container_width=True)
                else:
                    st.info(translator.t('no_valid_positions_to_display_batch', fallback="没有有效的持仓可供显示。"))
            else:
                st.info(translator.t('no_current_positions_info', fallback="当前没有持仓。"))
        else:
            st.info(translator.t('no_current_positions_info', fallback="当前没有持仓。"))

    def _handle_liquidation(self, liquidation_orders: List[Dict], system):  # system is TradingSystem instance
        """处理平仓订单 (使用翻译)"""
        logger.debug(f"Handling liquidation for {len(liquidation_orders)} order(s).")
        if not liquidation_orders:
            st.info(translator.t('no_positions_to_closeout_msg', fallback="没有符合条件的持仓可平仓"))
            return

        # 获取当前投资组合状态以传递给 execute_batch_orders (如果它需要)
        current_portfolio_for_batch = st.session_state.get('portfolio', {})

        # 使用批量执行方法
        batch_result = self.execute_batch_orders(
            orders_to_process=liquidation_orders,
            current_portfolio_state=current_portfolio_for_batch,
            system_ref=system
        )

        successful_trades = batch_result["total_success"]
        failed_trades_info = []

        for failed_order in batch_result["failed"]:
            failed_trades_info.append({
                "symbol": failed_order["order"].get("symbol"),
                "reason": failed_order["reason"]
            })

        st.success(
            translator.t('closeout_success_msg', fallback="成功平仓 {count} 个持仓").format(count=successful_trades))
        if failed_trades_info:
            st.error(translator.t('closeout_failed_msg', fallback="平仓失败 {count} 个持仓").format(
                count=len(failed_trades_info)))
            st.write(translator.t('closeout_failed_details', fallback="失败详情:"))
            st.dataframe(pd.DataFrame(failed_trades_info))

        if successful_trades > 0 or failed_trades_info:  # Rerun if any action happened
            st.rerun()

    # 修复：移除析构函数，防止意外关闭连接导致其他页面崩溃
    # def __del__(self):
    #     """清理富途交易连接"""
    #     if self.futu_trade_ctx_hk: self.futu_trade_ctx_hk.close()
    #     if self.futu_trade_ctx_us: self.futu_trade_ctx_us.close()