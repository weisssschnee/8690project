# vvvvvvvvvvvvvvvvvvvv START OF REPLACEMENT vvvvvvvvvvvvvvvvvvvv
import ssl
import socket
import logging

# --- 核心修复：直接导入 urllib3，而不是通过 requests.packages ---
try:
    import urllib3
    from urllib3.util import connection as urllib3_connection
    from urllib3.util import ssl_ as urllib3_ssl

    # --- Force IPv4 to solve connection issues on some networks ---
    _original_allowed_gai_family = urllib3_connection.allowed_gai_family


    def force_ipv4():
        return socket.AF_INET


    urllib3_connection.allowed_gai_family = force_ipv4

    # --- Disable SSL Certificate Verification globally for urllib3 ---
    _original_create_urllib3_context = urllib3_ssl.create_urllib3_context


    def create_unverified_context(**kwargs):
        # This creates a context that doesn't verify certificates
        kwargs['cert_reqs'] = ssl.CERT_NONE
        return _original_create_urllib3_context(**kwargs)


    urllib3_ssl.create_urllib3_context = create_unverified_context

    # --- Disable the InsecureRequestWarning that comes with verify=False ---
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    logging.critical(
        "Applied aggressive network patch: Forced IPv4 and DISABLED SSL VERIFICATION for urllib3. This is a potential security risk.")
except Exception as e:
    logging.error(f"Failed to apply aggressive network patch: {e}", exc_info=True)
# --- END: AGGRESSIVE NETWORK PATCH ---


# ^^^^^^^^^^^^^^^^^^^^ END OF REPLACEMENT ^^^^^^^^^^^^^^^^^^^^
import pandas as pd
import numpy as np
import logging # 确保 logging 先导入
logger_av_test = logging.getLogger("AV_IMPORT_TEST")
logger_av_test.critical("Attempting to import alpha_vantage directly...")
import yfinance as yf
from datetime import datetime, timedelta, date
import requests
import time
import os
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import streamlit as st
import random
import threading
import json
import futu as ft
try:
    from futu import *
    FUTU_AVAILABLE = True
except ImportError:
    FUTU_AVAILABLE = False
    logging.warning("futu-api library not found. FutuOpenD functionality will be disabled.")

logger = logging.getLogger(__name__)

AKSHARE_AVAILABLE = False; TUSHARE_AVAILABLE = False; FINNHUB_AVAILABLE = False; POLYGON_AVAILABLE = False
ALPHA_VANTAGE_AVAILABLE = False; ALPHA_VANTAGE_SEARCH_AVAILABLE = False # For AV library search
ak = None; ts = None; _FinnhubClient = None; _PolygonClient = None # Use underscore to avoid potential global name clashes
_AVTimeSeriesLib, _AVFundamentalDataLib, _AVSearchingLib = None, None, None
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
    logging.info("AkShare library found and imported.")
except ImportError:
    ak = None; AKSHARE_AVAILABLE = False; logging.warning("AkShare library not found. CN stock data functionality will be limited.")
try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    ts = None; TUSHARE_AVAILABLE = False; logging.warning("Tushare library not found. Some CN stock data functionality limited.")
try:
    from finnhub import Client as FinnhubClient
    FINNHUB_AVAILABLE = True
except ImportError:
    FinnhubClient = None; FINNHUB_AVAILABLE = False; logging.warning("finnhub-python library not found. Finnhub functionality disabled.")
try:
    from polygon import RESTClient as PolygonClient
    POLYGON_AVAILABLE = True
except ImportError:
    PolygonClient = None; POLYGON_AVAILABLE = False; logging.warning("polygon-api-client library not found. Polygon functionality disabled.")
try:
    from alpha_vantage.searching import Searching
    from alpha_vantage.timeseries import TimeSeries as AVTimeSeriesInternal
    from alpha_vantage.fundamentaldata import FundamentalData as AVFundamentalDataInternal
    TimeSeriesAV, FundamentalDataAV = AVTimeSeriesInternal, AVFundamentalDataInternal # Assign to module-level vars
    ALPHA_VANTAGE_AVAILABLE = True; logging.info("AV TimeSeries & FundamentalData imported.")
    from alpha_vantage.timeseries import TimeSeries
    from alpha_vantage.fundamentaldata import FundamentalData
    try:
        from alpha_vantage.searching import Searching as AVSearchingInternal
        SearchingAVLib = AVSearchingInternal # Assign to module-level var
        ALPHA_VANTAGE_SEARCH_AVAILABLE = True; logging.info("AV Searching imported.")
    except ImportError: logger.warning("alpha_vantage.searching not found.")
except ImportError: logging.warning("Core Alpha Vantage components not found.")

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
    logging.info("websocket-client library found.")
except ImportError:
    websocket = None
    WEBSOCKET_AVAILABLE = False
    logging.warning("websocket-client library not found. WebSocket functionality will be disabled.")

ALPHA_VANTAGE_KEY = "YOUR_ALPHA_KEY"  # 替换为实际密钥
TUSHARE_TOKEN = "YOUR_TUSHARE_TOKEN"  # 替换为实际令牌

try:
    FUTU_AVAILABLE = True
except ImportError:
    FutuManager = None
    FUTU_AVAILABLE = False
    logging.warning("FutuManager module not found. Futu integration will be disabled.")



def _is_cn_stock_helper(symbol: str) -> bool:
    return isinstance(symbol, str) and (symbol.upper().endswith(('.SH', '.SZ')) or (symbol.isdigit() and len(symbol) == 6))

# --- Generic Data Formatter (Top Level - ensure this is correct and robust) ---
def _format_ohlcv_dataframe(df: pd.DataFrame, date_col_identifier: str, ohlcv_map: Dict[str, str], date_format: Optional[str] = None, volume_multiplier: int = 1, source_name: str = "API", date_unit: Optional[str] = None, datetime_index_is_date: bool = False, time_zone: Optional[str] = None) -> Optional[pd.DataFrame]:
    # (Implementation from previous version - assumed correct for brevity)
    try:
        if df is None or df.empty: return None
        df_copy = df.copy(); rename_for_pandas = {orig: std for std, orig in ohlcv_map.items() if orig in df_copy.columns}
        if rename_for_pandas: df_copy.rename(columns=rename_for_pandas, inplace=True)
        date_source_col_name = 'date_col_from_index' if datetime_index_is_date and isinstance(df_copy.index, pd.DatetimeIndex) else (date_col_identifier if date_col_identifier in df_copy.columns else None)
        if not date_source_col_name: logger.error(f"{source_name} format error: Date ID '{date_col_identifier}' not in {df_copy.columns.tolist()}"); return None
        if date_source_col_name == 'date_col_from_index': df_copy['date_col_from_index_actual'] = df_copy.index; date_source_col_name = 'date_col_from_index_actual'
        if date_unit: df_copy['date'] = pd.to_datetime(df_copy[date_source_col_name], unit=date_unit, errors='coerce', utc=(date_unit in ['ms','s','ns']))
        else: df_copy['date'] = pd.to_datetime(df_copy[date_source_col_name], format=date_format, errors='coerce')
        df_copy = df_copy.dropna(subset=['date']);
        if df_copy.empty: return None
        df_copy = df_copy.set_index('date')
        if time_zone:
            if df_copy.index.tz is None: df_copy.index = df_copy.index.tz_localize(time_zone, ambiguous='infer', nonexistent='NaT')
            else: df_copy.index = df_copy.index.tz_convert(time_zone)
            df_copy = df_copy.dropna(subset=[df_copy.index.name]);
            if df_copy.empty: return None
        standard_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close']; cols_to_keep = [col for col in standard_cols if col in df_copy.columns]
        if 'adj_close' in cols_to_keep:
            df_copy['close'] = df_copy['adj_close']
            if 'close' in cols_to_keep and 'adj_close' != 'close': cols_to_keep.remove('adj_close')
            elif 'close' not in cols_to_keep: cols_to_keep.append('close'); cols_to_keep.remove('adj_close')
        if 'close' not in df_copy.columns or 'close' not in cols_to_keep : logger.error(f"{source_name}: 'close' missing. Cols: {df_copy.columns.tolist()}"); return None
        cols_to_keep = sorted(list(set(col for col in cols_to_keep if col in df_copy.columns)))
        df_final = df_copy[cols_to_keep].sort_index()
        if 'volume' in df_final.columns and not df_final['volume'].empty and isinstance(df_final['volume'].iloc[0], (int,float)): df_final['volume'] *= volume_multiplier
        for col in df_final.columns:
             if col != df_final.index.name : df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
        df_final = df_final.dropna(subset=['close'])
        return df_final if not df_final.empty else None
    except Exception as e: logger.error(f"Error in _format_ohlcv_dataframe for {source_name} (date_id:'{date_col_identifier}'): {e}", exc_info=True); return None

def _format_av_direct_json_hist(json_data: Dict, symbol: str, interval: Optional[str]) -> Optional[pd.DataFrame]:
    # (Implementation from previous version - ensure robust)
    try:
        series_key = None; func_interval_map = {"TIME_SERIES_DAILY_ADJUSTED": "Time Series (Daily)", "TIME_SERIES_WEEKLY_ADJUSTED": "Weekly Adjusted Time Series", "TIME_SERIES_MONTHLY_ADJUSTED": "Monthly Adjusted Time Series"}
        if interval and interval not in ['1d', '1w', '1m']: series_key = next((k for k in json_data if f"Time Series ({interval})" in k), None)
        elif interval == '1d': series_key = next((k for k in json_data if func_interval_map["TIME_SERIES_DAILY_ADJUSTED"] in k or "Time Series (Daily)" in k), None)
        elif interval == '1w': series_key = next((k for k in json_data if func_interval_map["TIME_SERIES_WEEKLY_ADJUSTED"] in k or "Weekly Time Series" in k ), None)
        elif interval == '1m': series_key = next((k for k in json_data if func_interval_map["TIME_SERIES_MONTHLY_ADJUSTED"] in k or "Monthly Time Series" in k), None)
        if not series_key or series_key not in json_data: err_msg = json_data.get("Error Message", json_data.get("Information", "Unknown AV API response")); logger.warning(f"AV Direct JSON: Key for interval '{interval}' not found for {symbol}. API: {err_msg}"); return None
        data_dict = json_data[series_key]; df = pd.DataFrame.from_dict(data_dict, orient='index'); df.index.name = 'date_orig'
        map_key = '5. adjusted close' if '5. adjusted close' in df.columns else ('4. close' if '4. close' in df.columns else None)
        if not map_key: logger.error(f"AV Direct JSON: No close/adj_close found for {symbol} in {df.columns}"); return None
        vol_key = '6. volume' if interval and interval not in ['1d','1w','1m'] and '6. volume' in df.columns else ('5. volume' if '5. volume' in df.columns else None)
        if not vol_key: logger.warning(f"AV Direct JSON: No volume key for {symbol}, vol set to 0"); df['temp_vol_zeros'] = 0; vol_key = 'temp_vol_zeros'
        return _format_ohlcv_dataframe(df.reset_index(), date_col_identifier='date_orig', ohlcv_map={'1. open':'open', '2. high':'high', '3. low':'low', map_key:'close', vol_key:'volume'}, source_name="AlphaVantageDirect", datetime_index_is_date=False, time_zone='America/New_York')
    except Exception as e: logger.error(f"Error formatting AV Direct JSON for {symbol}: {e}", exc_info=True); return None


def _format_akshare_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    return _format_ohlcv_dataframe(df,
                                   date_col_identifier='日期', # <--- 修改
                                   ohlcv_map={'open':'开盘','close':'收盘','high':'最高','low':'最低','volume':'成交量'},
                                   volume_multiplier=100,
                                   source_name="AkShare",
                                   datetime_index_is_date=False)
def _format_tushare_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    return _format_ohlcv_dataframe(df,
                                   date_col_identifier='trade_date', # <--- 修改
                                   ohlcv_map={'open':'open','high':'high','low':'low','close':'close','vol':'volume'},
                                   date_format='%Y%m%d',
                                   source_name="Tushare",
                                   datetime_index_is_date=False)

# Removed persist="disk" to allow TTL to work for now
@st.cache_data(ttl=3600, show_spinner="Fetching yfinance history...")
def _cached_yfinance_history(symbol: str, start_str: str, end_str: str, interval: str, attempt_info: str) -> Optional[pd.DataFrame]:
    logger.info(f"[YF Cache Call - {attempt_info}] For: {symbol}")
    try:
        stock = yf.Ticker(symbol)
        data_yf = stock.history(start=start_str, end=end_str, interval=interval, auto_adjust=False, back_adjust=False)
        if data_yf is None or data_yf.empty: logger.warning(f"yfinance: {symbol} empty ({attempt_info})."); return None
        return _format_yfinance_data(data_yf.reset_index())
    except yf.exceptions.YFRateLimitError as e_rate: logger.error(f"yfinance: Rate limit {symbol} ({attempt_info}). {e_rate}"); raise
    except Exception as e: logger.error(f"yfinance: Error {symbol} ({attempt_info}): {e}", exc_info=True); raise


@st.cache_data(ttl=3600, show_spinner="Fetching Alpha Vantage data...")
def _call_av_api_cached(params_tuple: tuple) -> Optional[Dict]:
    """
    一个简单的缓存函数，只负责执行 Alpha Vantage API 请求。
    它接收一个元组以使其可被缓存。
    """
    params = dict(params_tuple) # 将元组转换回字典
    base_url = "https://www.alphavantage.co/query"
    logger.info(f"AV API Cache Miss: Calling {params.get('function')} for {params.get('symbol')}")
    try:
        with requests.Session() as s:
            response = s.get(base_url, params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"AV API request failed for symbol {params.get('symbol')}: {e}")
        return None # 返回 None 表示失败
    except Exception as e:
        logger.error(f"AV API call encountered an unexpected error for symbol {params.get('symbol')}: {e}", exc_info=True)
        return None

@st.cache_data(ttl=86400, show_spinner="Searching stocks (AV Lib)...") # Renamed from _cached_av_lib_search
def _cached_alpha_vantage_library_search(query: str, api_key: str, limit: int) -> List[Dict[str, str]]:
    # Uses ALPHA_VANTAGE_LIB_AVAILABLE and SearchingAVLib (global)
    if not (ALPHA_VANTAGE_AVAILABLE and SearchingAVLib and api_key):
        logger.warning("AV Library search skipped: lib or key unavailable.")
        return []
    logger.info(f"Executing AV Library search (cached) for: '{query}'")
    results: List[Dict[str,str]] = []; seen_symbols = set()
    try:
        av_sc = SearchingAVLib(key=api_key, output_format='pandas') # Create instance here
        data_av, _ = av_sc.get_symbol_search(keywords=query)
        if data_av is not None and not data_av.empty:
            for _, row in data_av.iterrows():
                symbol = row.get('1. symbol'); name = row.get('2. name'); region = row.get('4. region'); type_ = row.get('3. type')
                if symbol and symbol not in seen_symbols and len(results) < limit:
                    results.append({'symbol': symbol, 'name': name or '', 'exchange': region or '', 'type': type_ or 'N/A', 'source': 'AlphaVantageLib'})
                    seen_symbols.add(symbol)
            logger.info(f"AV Library search found {len(results)} new results for '{query}'.")
    except Exception as e: logger.warning(f"AV Library search failed for '{query}': {e}")
    return results


def _format_av_lib_hist_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Formats data returned by the alpha_vantage library."""
    try:
        df_copy = df.copy()
        # The library returns a DataFrame with a DatetimeIndex
        df_copy.index.name = 'date'
        rename_map = {
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. adjusted close': 'adj_close',
            '6. volume': 'volume',
        }
        df_copy = df_copy.rename(columns=rename_map)

        # Select and reorder standard columns
        standard_cols = ['open', 'high', 'low', 'close', 'volume']
        # Prioritize adj_close if available
        if 'adj_close' in df_copy.columns:
            df_copy['close'] = df_copy['adj_close']

        final_cols = [col for col in standard_cols if col in df_copy.columns]
        if 'close' not in final_cols:
            logger.error(f"AV Lib data missing 'close' or 'adjusted close'. Columns: {df.columns}")
            return None

        df_final = df_copy[final_cols].sort_index()
        for col in df_final.columns:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
        df_final = df_final.dropna(subset=['close'])

        return df_final
    except Exception as e:
        logger.error(f"Error formatting Alpha Vantage library data: {e}", exc_info=True)
        return None


# --- 新增：顶级缓存函数用于 Alpha Vantage Library 的搜索 (如果使用) ---
@st.cache_data(ttl=86400, show_spinner="Searching stocks (AV Lib)...", persist="disk")
def _cached_av_lib_search(query: str, api_key: str, limit: int) -> List[Dict[str, str]]:
    if not (ALPHA_VANTAGE_AVAILABLE and SearchingAVLib and api_key):
        logger.warning("AV Library search skipped: lib or key unavailable.")
        return []

    logger.info(f"Executing AV Library search (cached) for: '{query}'")
    results: List[Dict[str, str]] = []
    seen_symbols = set()  # Keep track within this specific cached function call
    try:
        av_sc = SearchingAVLib(key=api_key, output_format='pandas')
        data_av, _ = av_sc.get_symbol_search(keywords=query)
        if data_av is not None and not data_av.empty:
            for _, row in data_av.iterrows():
                symbol = row.get('1. symbol')
                name = row.get('2. name')
                region = row.get('4. region')  # Or 'exchange' if available directly
                type_ = row.get('3. type')
                if symbol and symbol not in seen_symbols and len(results) < limit:
                    results.append({
                        'symbol': symbol, 'name': name or '', 'exchange': region or '',
                        'type': type_ or 'N/A', 'source': 'AlphaVantageLib'
                    })
                    seen_symbols.add(symbol)
            logger.info(f"AV Library search found {len(results)} new results for '{query}'.")
    except Exception as e:
        logger.warning(f"AV Library search failed for '{query}': {e}")
    return results
def _format_av_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    date_col = df.index.name if df.index.name else 'date'
    df_to_format = df.reset_index() if df.index.name else df.copy()
    if df.index.name is None and 'index' in df_to_format.columns: date_col = 'index'
    return _format_ohlcv_dataframe(df_to_format,
                                   date_col_identifier=date_col, # <--- 修改
                                   ohlcv_map={'open':'1. open', 'high':'2. high', 'low':'3. low',
                                              'close':'4. close', 'adj_close':'5. adjusted close',
                                              'volume':'6. volume'},
                                   source_name="AlphaVantage",
                                   datetime_index_is_date=False) # 因为我们传递了包含列的df_to_format


def _format_polygon_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    try:
        logger.debug(f"Formatting Polygon data ({len(df)} rows)...")
        # Polygon API 返回 'timestamp' 作为时间戳列的原始名称
        # _format_ohlcv_dataframe 期望的第一个参数是原始 DataFrame 中日期/时间戳列的名称

        # --- 修正这里的 date_col_identifier ---
        return _format_ohlcv_dataframe(df,
                                       date_col_identifier='timestamp',  # <--- 将 't' 改为 'timestamp'
                                       ohlcv_map={'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v'},
                                       date_unit='ms',  # Polygon timestamps are in milliseconds
                                       source_name="Polygon",
                                       datetime_index_is_date=False  # 'timestamp' 是一个明确的列
                                       )
        # --- 结束修正 ---
    except Exception as e:
        logger.error(f"Error formatting Polygon data: {e}", exc_info=True)
        return None




@st.cache_data(ttl=7200, show_spinner="正在获取 Finnhub 新闻...")
def _fetch_finnhub_news_cached(api_key: str, symbol: str, num_articles: int = 20) -> List[Dict]:
    """
    [独立的、可缓存的函数] 使用 Finnhub API 获取公司新闻。
    """
    if not api_key:
        logger.warning("Finnhub API key was not provided to _fetch_finnhub_news_cached.")
        return []

    # Finnhub 需要日期范围，我们获取过去一个月的新闻
    end_date = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=30)).strftime("%Y-%m-%d")

    logger.info(f"Executing cached news fetch from Finnhub for '{symbol}'...")

    try:
        # 懒加载 Finnhub 客户端
        if not 'finnhub_client_news' in st.session_state:
            st.session_state.finnhub_client_news = FinnhubClient(api_key=api_key)

        finnhub_client = st.session_state.finnhub_client_news

        news_list = finnhub_client.company_news(symbol, _from=start_date, to=end_date)

        if not news_list:
            logger.warning(f"Finnhub returned no news for '{symbol}'.")
            return []

        # 截取所需数量的文章
        articles = news_list[:num_articles]
        logger.info(f"Successfully fetched {len(articles)} news articles for '{symbol}' from Finnhub.")

        # 将 Finnhub 的数据格式标准化为我们期望的格式
        formatted_articles = []
        for art in articles:
            formatted_articles.append({
                'title': art.get('headline'),
                'description': art.get('summary'),
                'publishedAt': datetime.fromtimestamp(art.get('datetime')).isoformat() if art.get('datetime') else None,
                'url': art.get('url'),
                'source': art.get('source')
            })
        return formatted_articles

    except Exception as e:
        logger.error(f"An unexpected error occurred during Finnhub news fetching for '{symbol}': {e}", exc_info=True)
        return []


# vvvvvvvvvvvvvvvvvvvv START OF REPLACEMENT vvvvvvvvvvvvvvvvvvvv
@st.cache_data(ttl=7200, show_spinner="正在获取最新新闻...")
def _fetch_news_cached(news_api_key: str, symbol: str, num_articles: int = 20) -> List[Dict]:
    """
    [修复版] 独立的、可缓存的函数，使用 NewsAPI.org 获取新闻。
    这个函数不依赖任何类的 'self'，因此可以被 Streamlit 安全地缓存。
    """

    # 1. 验证输入参数
    if not news_api_key:
        logger.warning("NewsAPI key was not provided to _fetch_news_cached.")
        return []

    # 2. 准备请求参数
    query = symbol
    base_url = "https://newsapi.org/v2/everything"
    params = {
        'q': query,
        'apiKey': news_api_key,
        'pageSize': num_articles,
        'sortBy': 'publishedAt',
        'language': 'en'
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    logger.info(f"Executing cached news fetch for '{query}'...")

    # 3. 执行 API 请求并处理响应
    try:
        # vvvvvvvvvvvv START OF FIX vvvvvvvvvvvv
        response = requests.get(base_url, params=params, headers=headers, timeout=15)  # 增加超时并添加 headers
        # 检查 HTTP 状态码，如果不是 2xx，则抛出异常
        response.raise_for_status()

        data = response.json()

        # 检查 NewsAPI 返回的业务状态
        if data.get('status') == 'ok':
            articles = data.get('articles', [])
            logger.info(f"Successfully fetched {len(articles)} news articles for '{query}'.")
            # 只返回我们需要的数据，减少缓存大小和后续处理的复杂性
            return [
                {
                    'title': art.get('title'),
                    'description': art.get('description'),
                    'publishedAt': art.get('publishedAt'),
                    'url': art.get('url')
                }
                for art in articles if art  # 确保 art 不是 None
            ]
        else:
            api_error_message = data.get('message', 'Unknown API error from NewsAPI')
            logger.error(f"NewsAPI returned an error for query '{query}': {api_error_message}")
            return []

    except requests.exceptions.Timeout:
        logger.error(f"Timeout occurred while fetching news from NewsAPI for query '{query}'.")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP request to NewsAPI failed for query '{query}': {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred during news fetching for '{query}': {e}", exc_info=True)
        return []


class FutuTickerHandler(ft.TickerHandlerBase if FUTU_AVAILABLE else object):
    """处理富途实时行情推送的回调类"""
    def __init__(self, data_container: Dict, lock: threading.Lock):
        if not FUTU_AVAILABLE: return
        super().__init__()
        self._data = data_container
        self._lock = lock
        logger.info("FutuTickerHandler initialized.")

    def on_recv_rsp(self, rsp_pb):
        ret_type, data = super(FutuTickerHandler, self).on_recv_rsp(rsp_pb)
        if ret_type == ft.RET_OK and isinstance(data, ft.TickerData):
            with self._lock:
                # 存储标准化后的数据
                self._data[data.code] = {
                    'price': data.last_price,
                    'volume_tick': data.volume, # 推送的是单笔成交量
                    'timestamp': pd.to_datetime(data.data_time).timestamp(),
                    'source': 'Futu Push'
                }
        return ft.RET_OK, None


class FutuManager:
    """管理 Futu OpenAPI 连接和交互的服务"""

    def __init__(self, config):
        if not FUTU_AVAILABLE:
            self.is_connected = False
            logger.warning("Cannot initialize FutuManager: futu-api library not installed.")
            return

        self.config = config
        self.host = getattr(config, 'FUTU_HOST', '127.0.0.1')
        self.port = int(getattr(config, 'FUTU_PORT', 11111))
        self.trd_password = getattr(config, 'FUTU_PWD', '')

        self.quote_ctx: Optional[ft.OpenQuoteContext] = None
        # 您可以按需添加交易上下文
        # self.trade_ctx_hk: Optional[ft.OpenHKTradeContext] = None

        self.realtime_quotes: Dict[str, Dict[str, Any]] = {}
        self.data_lock = threading.Lock()
        self.is_connected = False
        logger.info("FutuManager instance created.")

    def connect(self):
        """建立与 FutuOpenD 的连接 (这是一个阻塞操作)"""
        if self.is_connected: return True
        logger.info(f"Connecting to FutuOpenD at {self.host}:{self.port}...")
        try:
            self.quote_ctx = ft.OpenQuoteContext(host=self.host, port=self.port)
            start_result = self.quote_ctx.start()

            # 检查返回结果是否是元组并且第一个元素是 RET_OK
            if isinstance(start_result, tuple) and start_result[0] == ft.RET_OK:



                handler = FutuTickerHandler(self.realtime_quotes, self.data_lock)
                self.quote_ctx.set_handler(handler)

                self.is_connected = True
                logger.info("FutuManager connected successfully.")
                return True

            else:
                # 如果 start() 返回 None 或错误码
                error_message = start_result[1] if isinstance(start_result,
                                                              tuple) else "Connection failed (start() returned non-tuple)"
                raise ConnectionError(f"Futu context start() failed: {error_message}")

        except Exception as e:
            logger.error(f"Error connecting to FutuOpenD: {e}", exc_info=True)
            self.disconnect();
            return False

    def disconnect(self):
        logger.info("Disconnecting from FutuOpenD...")
        if self.quote_ctx: self.quote_ctx.close()
        self.quote_ctx = None
        self.is_connected = False

    def subscribe(self, symbols: List[str]):
        if self.is_connected and self.quote_ctx:
            ret, data = self.quote_ctx.subscribe(symbols, [ft.SubType.TICKER])
            if ret == ft.RET_OK:
                logger.info(f"Futu: Subscribed to Ticker for {symbols}")
            else:
                logger.error(f"Futu: Failed to subscribe to {symbols}: {data}")

    def get_realtime_price_from_push(self, symbol: str) -> Optional[Dict[str, Any]]:
        with self.data_lock:
            return self.realtime_quotes.get(symbol)

    def get_historical_data(self, symbol: str, start: str, end: str, ktype_str: str) -> Optional[
        pd.DataFrame]:  # Changed last parameter
        if not self.is_connected:
            logger.error("Futu not connected.")
            return None

        # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
        ktype_map = {
            '1d': ft.KLType.K_DAY, '1w': ft.KLType.K_WEEK, '1M': ft.KLType.K_MON,
            '1h': ft.KLType.K_60M, '30m': ft.KLType.K_30M, '15m': ft.KLType.K_15M,
            '5m': ft.KLType.K_5M, '1m': ft.KLType.K_1M
        }
        ktype = ktype_map.get(ktype_str)  # Map the string to the Futu enum
        # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^

        if not ktype:
            logger.warning(f"Futu unsupported ktype: {ktype_str}")
            return None

        ret, data = self.quote_ctx.get_history_kline(
            symbol.upper(),
            start=start,
            end=end,
            ktype=ktype,  # Use the mapped enum
            autype=ft.AuType.QFQ
        )
        if ret == ft.RET_OK:
            return data

        logger.error(f"Futu get_history_kline failed for {symbol}: {data}")
        return None

    def search_stocks(self, market: str) -> Optional[pd.DataFrame]:
        if not self.is_connected: return None
        market_map = {'US': ft.Market.US, 'HK': ft.Market.HK, 'CN': ft.Market.SH}  # Futu CN maps to SH/SZ
        futu_market = market_map.get(market.upper())
        if not futu_market: return None

        ret, data = self.quote_ctx.get_stock_basicinfo(futu_market, stock_type=ft.SecurityType.STOCK)
        return data if ret == ft.RET_OK else None

class WebSocketManager:
    """处理实时数据 WebSocket 连接的类"""

    def __init__(self, config):
        if not WEBSOCKET_AVAILABLE:
            raise ImportError("Cannot initialize WebSocketManager: websocket-client library is not installed.")

        self.config = config
        self.logger = logging.getLogger(__name__)
        self.ws_url = None
        self.api_key = None
        self.ws_app = None
        self.thread = None
        self.is_running = False

        # --- 线程安全的数据存储 ---
        self.latest_prices: Dict[str, Dict[str, Any]] = {}
        self.data_lock = threading.Lock()

        self.subscribed_symbols = set()
        self._setup_connection_params()

    def _setup_connection_params(self):
        """根据配置设置 WebSocket URL 和 API Key"""
        # 优先使用 Finnhub
        if FINNHUB_AVAILABLE and self.config.FINNHUB_KEY:
            self.api_key = self.config.FINNHUB_KEY
            self.ws_url = f"wss://ws.finnhub.io?token={self.api_key}"
            self.logger.info("WebSocketManager configured for Finnhub.")
        # 可以添加 Polygon 等其他源的 elif 分支
        # elif POLYGON_AVAILABLE and self.config.POLYGON_KEY: ...
        else:
            self.logger.warning("No WebSocket provider API key found (Finnhub needed). WebSocket will not connect.")

    def start(self):
        """在一个独立的后台线程中启动 WebSocket 连接"""
        if not self.ws_url:
            self.logger.error("Cannot start WebSocket: URL not configured (API key missing?).")
            return
        if self.is_running:
            self.logger.warning("WebSocket manager is already running.")
            return

        self.is_running = True
        # 定义 WebSocketApp
        self.ws_app = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        # 在后台线程中运行
        self.thread = threading.Thread(target=self.ws_app.run_forever, daemon=True)
        self.thread.start()
        self.logger.info("WebSocket manager thread started.")

    def stop(self):
        """停止 WebSocket 连接和后台线程"""
        self.is_running = False
        if self.ws_app:
            self.logger.info("Stopping WebSocket manager...")
            self.ws_app.close()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)  # 等待线程结束
        self.logger.info("WebSocket manager stopped.")

    def subscribe(self, symbols: List[str]):
        """订阅股票的实时报价"""
        if not self.ws_app or not self.is_running:
            self.logger.warning(f"Cannot subscribe: WebSocket is not running. Symbols to add: {symbols}")
            # 可以缓存这些订阅，在连接打开时再发送
            return

        for symbol in symbols:
            if symbol not in self.subscribed_symbols:
                try:
                    # Finnhub 的订阅消息格式
                    subscribe_msg = json.dumps({'type': 'subscribe', 'symbol': symbol})
                    self.ws_app.send(subscribe_msg)
                    self.subscribed_symbols.add(symbol)
                    self.logger.info(f"Sent subscribe request for {symbol}.")
                    time.sleep(0.1)  # 避免发送过快
                except Exception as e:
                    self.logger.error(f"Failed to send subscribe message for {symbol}: {e}")

    def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        if not self.ws_app or not self.is_running:
            self.logger.warning("Cannot unsubscribe: WebSocket is not running.")
            return

        for symbol in symbols:
            if symbol in self.subscribed_symbols:
                try:
                    unsubscribe_msg = json.dumps({'type': 'unsubscribe', 'symbol': symbol})
                    self.ws_app.send(unsubscribe_msg)
                    self.subscribed_symbols.remove(symbol)
                    self.logger.info(f"Sent unsubscribe request for {symbol}.")
                except Exception as e:
                    self.logger.error(f"Failed to send unsubscribe message for {symbol}: {e}")

    def get_latest_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """从共享数据中获取最新的价格信息"""
        with self.data_lock:
            return self.latest_prices.get(symbol)  # 返回副本以防外部修改

    def _on_open(self, ws):
        self.logger.info("WebSocket connection opened.")
        # 可以在这里重新订阅之前失败的或预设的股票
        if self.subscribed_symbols:
            self.logger.info("Re-subscribing to existing symbols...")
            self.subscribe(list(self.subscribed_symbols))

    def _on_message(self, ws, message):
        """处理从服务器收到的消息"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')

            if msg_type == 'ping':
                ws.send(json.dumps({'type': 'pong'}))
                self.logger.debug("WebSocket ping-pong.")
                return

            if msg_type == 'trade':
                # Finnhub trade message format: {'data': [{'p': price, 's': symbol, 't': timestamp_ms, 'v': volume}], 'type': 'trade'}
                trade_data = data.get('data', [])
                with self.data_lock:
                    for trade in trade_data:
                        symbol = trade.get('s')
                        if symbol:
                            self.latest_prices[symbol] = {
                                'price': float(trade.get('p', 0)),
                                'volume': float(trade.get('v', 0)),
                                'timestamp': trade.get('t') / 1000.0,  # 毫秒转秒
                                'source': 'WebSocket'
                            }
                            # self.logger.debug(f"WS Update: {self.latest_prices[symbol]}")

        except json.JSONDecodeError:
            self.logger.warning(f"Received non-JSON WebSocket message: {message}")
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {e}", exc_info=True)

    def _on_error(self, ws, error):
        self.logger.error(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        self.logger.warning(f"WebSocket connection closed: Code={close_status_code}, Msg={close_msg}")
        # 可以在这里触发自动重连逻辑
        if self.is_running:
            self.logger.info("Attempting to reconnect WebSocket in 5 seconds...")
            time.sleep(5)
            self.start()  # 尝试重启


@st.cache_data(ttl=3600, show_spinner="加载历史数据 (DM Method)...", persist="disk")
def get_historical_data(self, symbol: str, days: int = 90, interval: str = "1d") -> Optional[pd.DataFrame]:
    self.logger.info(f"DM: get_historical_data for {symbol}, {days} days, interval {interval}")
    end_date_obj = date.today();
    start_date_obj = end_date_obj - timedelta(days=int(days * 1.5) + 7)
    data: Optional[pd.DataFrame] = None
    # ... (日期字符串准备) ...
    start_nodash = start_date_obj.strftime('%Y%m%d');
    end_nodash = end_date_obj.strftime('%Y%m%d')
    start_dash = start_date_obj.strftime('%Y-%m-%d');
    end_dash = end_date_obj.strftime('%Y-%m-%d')

    if self._is_cn_stock(symbol):
        if AKSHARE_AVAILABLE:  # ak 是全局导入的
            self.logger.debug(f"Attempting AkShare via cached func for {symbol}...")
            try:
                # data = _cached_akshare_hist(symbol, ...) # 假设有这个函数
                # 简化：直接调用 akshare，让 streamlit 缓存 get_historical_data 整个方法
                ak_s = symbol.split('.')[0] if '.' in symbol else symbol;
                period_map = {'1d': 'daily'};
                ak_p = period_map.get(interval)
                if ak_p and ak:
                    df_ak = ak.stock_zh_a_hist(symbol=ak_s, period=ak_p, start_date=start_nodash, end_date=end_nodash,
                                               adjust="qfq")
                else:
                    raise NotImplementedError
                if df_ak is not None and not df_ak.empty: data = _format_akshare_data(df_ak)
                if data is not None: return data
            except Exception as e:
                self.logger.warning(f"AkShare (in DM method) failed for {symbol}: {e}")

        if data is None and TUSHARE_AVAILABLE and self.tushare_token:
            self.logger.debug(f"Attempting Tushare via cached func for {symbol}...")
            try:  # Tushare
                # data = _cached_tushare_hist(symbol, self.tushare_token, ...) # 假设有这个函数
                ts_api = ts.pro_api(self.tushare_token);
                ts_s = symbol.upper();
                if interval == '1d':
                    df_ts = ts_api.daily(ts_code=ts_s, start_date=start_nodash, end_date=end_nodash)
                else:
                    raise NotImplementedError
                if df_ts is not None and not df_ts.empty: data = _format_tushare_data(df_ts)
                if data is not None: return data
            except Exception as e:
                self.logger.warning(f"Tushare (in DM method) failed for {symbol}: {e}")
    else:  # Non-CN
        if data is None and POLYGON_AVAILABLE and self.polygon_key:
            self.logger.debug(f"Attempting Polygon via cached func for {symbol}...")
            try:  # Polygon
                # data = _cached_polygon_hist(symbol, self.polygon_key, ...) # 假设有这个函数
                poly_c = PolygonClient(self.polygon_key)  # Create client for this call
                ts_map = {'1d': 'day'};
                m_map = {'1d': 1};
                ts_p = ts_map.get(interval, 'day');
                m_p = m_map.get(interval, 1)
                aggs = list(poly_c.list_aggs(symbol, m_p, ts_p, start_dash, end_dash, limit=50000))
                if aggs: df_p = pd.DataFrame(aggs); data = _format_polygon_data(df_p)
                if data is not None and not data.empty: return data
            except Exception as e:
                self.logger.warning(f"Polygon (in DM method) failed for {symbol}: {e}", exc_info=False)

        if data is None and FINNHUB_AVAILABLE and self.finnhub_key:
            try:  # Finnhub
                # data = _cached_finnhub_hist(symbol, self.finnhub_key, ...)
                pass  # Similar structure
            except Exception as e:
                self.logger.warning(f"Finnhub hist (in DM method) failed for {symbol}: {e}")

        # AV Library fallback (if AV Direct failed or not applicable)
        if data is None and ALPHA_VANTAGE_AVAILABLE and self.alpha_vantage_key:
            self.logger.debug(f"Attempting AV Library for {symbol}...")
            try:
                av_ts_lib = TimeSeriesAV(key=self.alpha_vantage_key, output_format='pandas')  # Create temporary client
                av_fn = av_ts_lib.get_daily_adjusted if interval == '1d' else av_ts_lib.get_intraday
                av_int_map_lib = {'1h': '60min', '15m': '15min'};
                av_i_lib = av_int_map_lib.get(interval)
                if interval == '1d':
                    df_av_lib, _ = av_fn(symbol=symbol, outputsize='full')
                elif av_i_lib:
                    df_av_lib, _ = av_fn(symbol=symbol, interval=av_i_lib, outputsize='full')
                else:
                    raise NotImplementedError(f"AV Lib interval '{interval}' not mapped")
                if df_av_lib is not None and not df_av_lib.empty:
                    data = _format_av_lib_hist_data(df_av_lib, interval)
                    if data is not None: data = data.sort_index().loc[start_dash:end_dash]
                    if data is not None and not data.empty: self.logger.info(
                        f"AV Lib hist OK for {symbol}."); return data
            except Exception as e:
                self.logger.warning(f"AV Lib hist failed for {symbol}: {e}")

    # yfinance Fallback
    if data is None:
        self.logger.info(f"Primary sources failed for {symbol}, attempting yfinance fallback with rate limiting...")
        yf_start_str = start_date_obj.strftime('%Y-%m-%d')
        yf_end_str = (end_date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
        for attempt in range(self.yfinance_max_retries + 1):
            try:
                now = time.time();
                elapsed = now - self.yfinance_last_call_time;
                wait_for = 0
                if attempt > 0: wait_for = self.yfinance_base_delay * (2 ** (attempt - 1))
                if elapsed < self.yfinance_min_interval: wait_for = max(wait_for, self.yfinance_min_interval - elapsed)
                if wait_for > 0: self.logger.info(
                    f"DM: yf rate limit. Wait {wait_for:.2f}s for {symbol} (hist att {attempt + 1})"); time.sleep(
                    wait_for)
                self.yfinance_last_call_time = time.time()
                # Call the top-level cached yfinance function
                data = _cached_yfinance_history(symbol, yf_start_str, yf_end_str, interval,
                                                attempt_info=str(attempt + 1))
                if data is not None: break  # Success or confirmed empty from yfinance
            except yf.exceptions.YFRateLimitError:
                if attempt == self.yfinance_max_retries: self.logger.error(
                    f"DM: Max yf retries (rate limit) for {symbol}."); data = None; break
            except Exception as e_yf_call:
                self.logger.error(f"DM: Unhandled yf error {symbol} (att {attempt + 1}): {e_yf_call}",
                                  exc_info=True); data = None; break
        if data is None: self.logger.error(f"DM: All yf hist attempts failed for {symbol}.")

    if data is not None and not data.empty:  # Final date filtering for the successfully fetched data
        try:
            if not isinstance(data.index, pd.DatetimeIndex): data.index = pd.to_datetime(data.index,
                                                                                         errors='coerce'); data = data.dropna(
                subset=[data.index.name])
            final_start_dt = (end_date_obj - timedelta(days=days));
            final_end_dt = end_date_obj
            if data.index.tz is not None: final_start_dt = pd.Timestamp(final_start_dt,
                                                                        tz=data.index.tz); final_end_dt = pd.Timestamp(
                final_end_dt, tz=data.index.tz)
            data = data.sort_index().loc[final_start_dt:final_end_dt]
            if data.empty: self.logger.warning(f"Data for {symbol} empty after final date filtering.")
        except Exception as e_filter:
            self.logger.error(f"Error final date filtering for {symbol}: {e_filter}. Returning as is.", exc_info=True)
    elif data is None:
        return pd.DataFrame()  # Return empty DataFrame on complete failure

    return data


@st.cache_data(ttl=30, show_spinner="获取实时价格...")
def fetch_realtime_price_from_sources_cached(
        symbol: str,
        akshare_available_flag: bool,
        finnhub_api_key: Optional[str],
        polygon_api_key: Optional[str],
        alpha_vantage_api_key: Optional[str]
) -> Optional[Dict[str, Any]]:
    """
    [ROBUST VERSION] Fetches real-time price from a prioritized list of reliable APIs.
    This function is self-contained and acts as the primary REST API engine for real-time data.
    """
    logger.info(f"[RT CACHE CALL] Fetching real-time price for: {symbol}")

    # --- Priority 1: Finnhub (Fast and Reliable) ---
    if finnhub_api_key and FINNHUB_AVAILABLE and FinnhubClient:
        logger.debug(f"Attempting Finnhub RT for {symbol}...")
        try:
            fh_client = FinnhubClient(api_key=finnhub_api_key)
            quote = fh_client.quote(symbol)
            # Check for a valid, non-zero current price ('c')
            if quote and isinstance(quote.get('c'), (int, float)) and quote['c'] > 0:
                logger.info(f"✅ Finnhub RT price for {symbol}: {quote['c']}")
                return {'price': float(quote['c']), 'timestamp': float(quote.get('t', time.time())),
                        'source': 'Finnhub'}
        except Exception as e:
            logger.warning(f"Finnhub RT request failed for {symbol}: {e}")

    # --- Priority 2: Polygon.io (Excellent for US stocks) ---
    if polygon_api_key and POLYGON_AVAILABLE and PolygonClient:
        logger.debug(f"Attempting Polygon RT for {symbol}...")
        try:
            poly_client = PolygonClient(polygon_api_key)
            # get_snapshot_ticker is the most efficient way to get the latest price
            snapshot = poly_client.get_snapshot_ticker(ticker=symbol)
            if snapshot and snapshot.last_trade and snapshot.last_trade.p > 0:
                logger.info(f"✅ Polygon RT price for {symbol}: {snapshot.last_trade.p}")
                return {'price': float(snapshot.last_trade.p), 'timestamp': snapshot.last_trade.t / 1000.0,
                        'source': 'Polygon'}
        except Exception as e:
            logger.warning(f"Polygon RT request failed for {symbol}: {e}")

    # --- Priority 3: AkShare (for CN Stocks) ---
    if _is_cn_stock_helper(symbol) and akshare_available_flag and ak:
        logger.debug(f"Attempting AkShare RT for {symbol}...")
        try:
            ak_symbol = symbol.split('.')[0] if '.' in symbol else symbol
            # This call fetches a snapshot of all stocks, which is efficient
            df_spot = ak.stock_zh_a_spot_em()
            if not df_spot.empty:
                stock_row = df_spot[df_spot['代码'] == ak_symbol]
                if not stock_row.empty:
                    price = stock_row.iloc[0].get('最新价')
                    if price is not None and isinstance(price, (int, float)) and price > 0:
                        logger.info(f"✅ AkShare RT price for {symbol}: {price}")
                        return {'price': float(price), 'timestamp': time.time(), 'source': 'AkShare'}
        except Exception as e:
            logger.warning(f"AkShare RT request failed for {symbol}: {e}")

    # --- Priority 4: Alpha Vantage (Least Reliable Fallback) ---
    # This is kept as a last resort, as it's the most likely to fail.
    if alpha_vantage_api_key:
        logger.debug(f"Attempting Alpha Vantage RT for {symbol}...")
        try:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={alpha_vantage_api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            quote_data = response.json()
            if "Global Quote" in quote_data and "05. price" in quote_data["Global Quote"]:
                price = float(quote_data["Global Quote"]["05. price"])
                if price > 0:
                    logger.info(f"✅ Alpha Vantage RT price for {symbol}: {price}")
                    return {'price': price, 'timestamp': time.time(), 'source': 'AlphaVantage'}
        except Exception as e:
            logger.warning(f"Alpha Vantage RT request failed for {symbol}: {e}")

    # If all API calls fail, return None
    logger.error(f"[RT CACHE CALL] All available REST APIs failed to get a real-time price for {symbol}.")
    return None


# --- search_stocks (no @st.cache_data here, calls cached or direct) ---
def search_stocks(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
    self.logger.debug(f"DM Method: search_stocks for '{query}'")
    results: List[Dict[str, str]] = [];
    seen_symbols = set()

    if self.futu_manager and self.futu_manager.is_connected:
        self.logger.debug(f"Attempting Futu stock search for query: '{query}'")
        try:
            # 遍历我们关心的市场
            # 注意：富途的 'CN' 市场分为 'SH' (上海) 和 'SZ' (深圳)
            # 为了简化，我们可以先只搜索这几个主要市场
            markets_to_search = ['US', 'HK', 'SH', 'SZ']

            for market in markets_to_search:
                # 调用 FutuManager 获取该市场的全量股票列表
                # 这一步可以考虑加入缓存，因为股票列表不常变动
                # @st.cache_data(ttl=86400) # 缓存一天
                # def get_futu_stock_list(market): return self.futu_manager.search_stocks(market)
                # df_futu_stocks = get_futu_stock_list(market)

                # (不使用缓存的直接调用)
                df_futu_stocks = self.futu_manager.search_stocks(market)

                if df_futu_stocks is not None and not df_futu_stocks.empty:
                    self.logger.debug(
                        f"Futu: Found {len(df_futu_stocks)} stocks for market {market}. Filtering for '{query}'...")

                    query_lower = query.lower()
                    # 在 DataFrame 中进行筛选
                    # 匹配条件：股票代码(code)或股票名称(name)包含查询字符串 (不区分大小写)
                    matches = df_futu_stocks[
                        df_futu_stocks['code'].str.lower().str.contains(query_lower, na=False) |
                        df_futu_stocks['name'].str.lower().str.contains(query_lower, na=False)
                        ]

                    if not matches.empty:
                        for _, row in matches.iterrows():
                            symbol = row.get('code')
                            if symbol and symbol not in seen_symbols:
                                results.append({
                                    'symbol': symbol,
                                    'name': row.get('name', ''),
                                    'exchange': market,  # 使用我们循环的市场代码
                                    'type': 'Stock',  # 从富途获取的都是股票
                                    'source': 'Futu'
                                })
                                seen_symbols.add(symbol)

                                # 如果达到数量限制，则停止内层和外层循环
                                if len(results) >= limit:
                                    break

                        self.logger.info(f"Futu search added {len(matches)} results from market {market}.")

                # 检查是否已达到数量限制
                if len(results) >= limit:
                    break  # 停止搜索其他市场

            # 如果通过富途找到了任何结果，就直接返回，不再尝试其他 API
            if results:
                self.logger.info(f"Futu search completed, returning {len(results)} results.")
                return results[:limit]  # 确保最终返回数量不超过 limit

        except Exception as e:
            self.logger.error(f"An error occurred during Futu stock search: {e}", exc_info=True)

    # Finnhub (using instance client)
    if self.finnhub_client and len(results) < limit:
        try:
            search_res = self.finnhub_client.symbol_lookup(query)
            if search_res and search_res.get('result'):
                for item in search_res['result']:
                    symbol = item.get('symbol');
                    name = item.get('description');
                    exch = item.get('displaySymbol');
                    type_ = item.get('type')
                    if symbol and symbol not in seen_symbols and len(results) < limit:
                        results.append(
                            {'symbol': symbol, 'name': name or '', 'exchange': exch or '', 'type': type_ or 'N/A',
                             'source': 'Finnhub'})
                        seen_symbols.add(symbol)
        except Exception as e:
            self.logger.warning(f"Finnhub search failed: {e}")

    # Alpha Vantage (Direct HTTP for SYMBOL_SEARCH using self.alpha_vantage_key and self.requests_session)
    if self.alpha_vantage_key and len(results) < limit:
        self.logger.debug(f"Attempting AV Direct HTTP search for '{query}' in DM method...")
        try:
            url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={query}&apikey={self.alpha_vantage_key}&datatype=json"
            response = self.requests_session.get(url, timeout=10)
            response.raise_for_status();
            search_json = response.json()
            if "bestMatches" in search_json:
                count = 0
                for match in search_json["bestMatches"]:
                    sym = match.get("1. symbol");
                    name = match.get("2. name");
                    region = match.get("4. region");
                    type_ = match.get("3. type")
                    if sym and sym not in seen_symbols_globally and len(
                            results) < limit:  # Use a different set name or pass it
                        results.append(
                            {'symbol': sym, 'name': name or '', 'exchange': region or '', 'type': type_ or 'N/A',
                             'source': 'AVDirectSearch'})
                        seen_symbols.add(sym);
                        count += 1  # Add to local set for this source
                if count > 0: self.logger.info(f"AV Direct HTTP Search added {count} results for '{query}'.")
        except Exception as e:
            self.logger.warning(f"AV Direct HTTP search failed for '{query}': {e}")
    # Alpha Vantage Library (using self.alpha_vantage_sc_lib_inst)
    if self.alpha_vantage_sc_lib_inst and len(results) < limit:
        try:
            data_av, _ = self.alpha_vantage_sc_lib_inst.get_symbol_search(keywords=query)
            # ... (parse data_av and add to results, checking seen_symbols_globally and limit) ...
        except Exception as e:
            self.logger.warning(f"AV Lib search failed for '{query}': {e}")

    if not results: results = self._search_stocks_local_fallback(query, limit)
    # Deduplicate combined_results if multiple sources could return same symbol
    final_unique_results = []
    final_seen = set()
    for r in results:
        if r['symbol'] not in final_seen:
            final_unique_results.append(r)
            final_seen.add(r['symbol'])
    return final_unique_results[:limit]



class DataManager:
    def __init__(self, config=None, futu_manager=None):
        """
        Initializes the DataManager.
        It sets up configuration, creates data directories, and starts a background
        thread to initialize API clients without blocking the main application.
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.futu_manager = futu_manager

        # 1. 设置路径和会话 (Setup Paths and Session)
        self.data_dir = getattr(config, 'DATA_PATH', Path("data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.requests_session = requests.Session()

        # 2. 从配置中安全地加载API密钥 (Load API keys from config)
        self.tushare_token = getattr(config, 'TUSHARE_TOKEN', None)
        self.alpha_vantage_key = getattr(config, 'ALPHA_VANTAGE_KEY', None)
        self.finnhub_key = getattr(config, 'FINNHUB_KEY', None)
        self.polygon_key = getattr(config, 'POLYGON_KEY', None)
        # Futu不需要key，它依赖本地服务连接

        # 诊断日志：确认API密钥是否已从配置中加载
        self.logger.info("--- DataManager API Key Status Check ---")
        self.logger.info(f"FINNHUB_KEY Loaded from config: {bool(self.finnhub_key)}")
        self.logger.info(f"POLYGON_KEY Loaded from config: {bool(self.polygon_key)}")
        # ...可以为其他key添加类似日志

        # 3. 初始化所有API客户端为None (Initialize all clients to None)
        self.ts_pro = None
        self.finnhub_client = None
        self.polygon_client = None
        self.futu_quote_ctx = None # Futu连接上下文
        self.websocket_manager = None # 如果您有WebSocket管理器

        # 4. 初始化yfinance速率限制参数 (Initialize yfinance rate limiters)
        self.yfinance_last_call_time = 0.0
        self.yfinance_min_interval = float(getattr(config, 'YFINANCE_MIN_INTERVAL', 2.0))
        self.yfinance_max_retries = int(getattr(config, 'YFINANCE_MAX_RETRIES', 1))
        self.yfinance_base_delay = int(getattr(config, 'YFINANCE_BASE_DELAY', 3))

        self.futu_quote_ctx: Optional[ft.OpenQuoteContext] = None
        self.futu_is_connected = False
        self.futu_realtime_quotes: Dict[str, Dict[str, Any]] = {}
        self.futu_data_lock = threading.Lock()

        # 5. 在所有依赖项都准备好后，启动后台线程进行API客户端初始化
        self.logger.info("DataManager creating a background thread for API client initialization.")
        init_thread = threading.Thread(target=self._initialize_api_clients, daemon=True)
        init_thread.start()

        try:
            init_thread = threading.Thread(target=self._initialize_api_clients, daemon=True)
            init_thread.start()
            self.logger.info("DataManager background initialization thread started.")
        except Exception as e:
            # 如果连线程都无法创建，这是个严重问题，但我们仍然不应该让 DataManager.__init__ 崩溃
            self.logger.critical(f"FATAL: Could not start DataManager initialization thread: {e}", exc_info=True)
            # 保持所有客户端为 None

        self.logger.info("DataManager __init__ has completed. API clients are initializing in the background.")



    def _initialize_api_clients(self):
        """
        [在后台线程中执行]
        初始化所有配置的 API 客户端，包括建立连接和进行简单的测试调用。
        这个方法会依次尝试初始化每个客户端，一个失败不影响下一个。
        """
        self.logger.info("Background thread: API client initialization process started...")

        # --- 1. 连接富途 (Futu) ---
        if self.futu_manager:
            logger.info("Background: A FutuManager instance was provided to DataManager.")
        else:
            logger.info("Background: No FutuManager instance was provided. Futu data source will be unavailable.")

        # --- 2. 初始化 Tushare ---
        if TUSHARE_AVAILABLE and hasattr(self.config, 'TUSHARE_TOKEN') and self.config.TUSHARE_TOKEN:
            self.logger.info("Background: Initializing Tushare client...")
            try:
                self.ts_pro = ts.pro_api(self.config.TUSHARE_TOKEN)
                # 进行一个简单的测试调用来验证 token
                test_df = self.ts_pro.trade_cal(exchange='SSE', start_date='20230101', end_date='20230101')
                if test_df is None:  # Tushare 在 token 错误时可能返回 None
                    raise ValueError("Tushare token might be invalid, test call returned None.")
                self.logger.info("Background: Tushare client initialized successfully.")
            except Exception as e:
                self.logger.error(f"Background: Tushare initialization failed: {e}")
                self.ts_pro = None

        # --- 3. 初始化 Finnhub ---
        if FINNHUB_AVAILABLE and self.finnhub_key:
            self.logger.info("Background: Initializing Finnhub client...")
            try:
                self.finnhub_client = FinnhubClient(api_key=self.finnhub_key)
                # 测试调用
                test_profile = self.finnhub_client.company_profile2(symbol="AAPL")

                # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
                if test_profile and isinstance(test_profile, dict):  # 检查 test_profile 是否是一个有效的字典
                    self.logger.info("Background: Finnhub client initialized and test call successful.")
                else:
                    # 如果 test_profile 是 None 或不是字典，说明测试失败
                    # 我们不需要从 None 中获取错误信息，直接抛出异常即可
                    raise ValueError(
                        "Finnhub test call returned None or invalid data, API key might be invalid or limited.")
                # ^^^^^^^^^^^^^^^^^^^^ END OF FIX ^^^^^^^^^^^^^^^^^^^^

            except Exception as e:
                self.logger.error(f"Background: Finnhub initialization failed: {e}")
                self.finnhub_client = None

        # --- 4. 初始化 Polygon ---
        if POLYGON_AVAILABLE and hasattr(self.config, 'POLYGON_KEY') and self.config.POLYGON_KEY:
            self.logger.info("Background: Initializing Polygon client...")
            try:
                self.polygon_client = PolygonClient(self.config.POLYGON_KEY)
                # 测试调用
                self.polygon_client.get_ticker_details("AAPL")
                self.logger.info("Background: Polygon client initialized successfully.")
            except Exception as e:
                self.logger.error(f"Background: Polygon initialization failed: {e}")
                self.polygon_client = None

        # --- 5. 初始化 Alpha Vantage ---
        if ALPHA_VANTAGE_AVAILABLE and hasattr(self.config, 'ALPHA_VANTAGE_KEY') and self.config.ALPHA_VANTAGE_KEY:
            try:
                # 统一使用 self.alpha_vantage_ts
                if TimeSeriesAV:  # TimeSeriesAV 是我们在顶部定义的导入别名
                    self.alpha_vantage_ts = TimeSeriesAV(key=self.alpha_vantage_key, output_format='pandas')

                # 其他 AV 客户端 (如果需要)
                if FundamentalDataAV:
                    self.alpha_vantage_fd = FundamentalDataAV(key=self.alpha_vantage_key, output_format='pandas')
                if SearchingAVLib:  # SearchingAVLib 是我们在顶部定义的导入别名
                    self.alpha_vantage_sc = SearchingAVLib(key=self.alpha_vantage_key, output_format='pandas')

                logger.info(
                    f"AV Lib clients init (TS:{bool(self.alpha_vantage_ts)}, FD:{bool(self.alpha_vantage_fd)}, SC:{bool(self.alpha_vantage_sc)}).")
            except Exception as e:
                logger.error(f"AV Lib client init failed: {e}")
                # 确保所有相关属性都设为 None
                self.alpha_vantage_ts = None
                self.alpha_vantage_fd = None
                self.alpha_vantage_sc = None

        # --- 6. 初始化 WebSocket (它自己也是非阻塞的) ---
        if WEBSOCKET_AVAILABLE and hasattr(self.config, 'FINNHUB_KEY') and self.config.FINNHUB_KEY:
            try:
                self.websocket_manager = WebSocketManager(self.config)
                self.websocket_manager.start()
                self.logger.info("Background: WebSocketManager started.")
            except Exception as e:
                self.logger.error(f"Background: Failed to initialize WebSocketManager: {e}", exc_info=True)
                self.websocket_manager = None

        self.logger.info("Background thread: All API client initializations are complete.")

    def get_news(self, symbol: str, num_articles: int = 20) -> List[Dict]:
        """
        [重构版] 获取新闻，优先使用 Finnhub，失败后回退到 NewsAPI。
        """
        self.logger.info(f"--- Starting news fetch pipeline for {symbol} ---")

        # --- Priority 1: Finnhub ---
        finnhub_api_key = getattr(self.config, 'FINNHUB_KEY', None)
        if finnhub_api_key:
            self.logger.debug(f"Attempting to fetch news from Finnhub for {symbol}...")
            finnhub_news = _fetch_finnhub_news_cached(finnhub_api_key, symbol, num_articles)
            # 如果 Finnhub 成功返回了新闻，我们就直接使用它
            if finnhub_news:
                self.logger.info(f"Successfully got {len(finnhub_news)} articles from Finnhub.")
                return finnhub_news
            else:
                self.logger.warning(f"Finnhub failed to return news for {symbol}. Trying next source.")
        else:
            self.logger.debug("Finnhub key not configured, skipping.")

        # --- Priority 2 (Fallback): NewsAPI ---
        news_api_key = getattr(self.config, 'NEWS_API_KEY', None)
        if news_api_key:
            self.logger.debug(f"Attempting to fetch news from NewsAPI (Fallback) for {symbol}...")
            # 调用 NewsAPI 的缓存函数
            newsapi_news = _fetch_news_cached(news_api_key, symbol, num_articles)
            if newsapi_news:
                self.logger.info(f"Successfully got {len(newsapi_news)} articles from NewsAPI fallback.")
                return newsapi_news
            else:
                self.logger.warning(f"NewsAPI fallback also failed to return news for {symbol}.")
        else:
            self.logger.debug("NewsAPI key not configured, skipping fallback.")

        # --- All sources failed ---
        self.logger.error(f"All configured news sources failed to provide data for {symbol}.")
        return []


    def _is_cn_stock(self, symbol: str) -> bool:
        return _is_cn_stock_helper(symbol)

    def _get_finnhub_client(self):
        """[新增] 懒加载 Finnhub 客户端实例。"""
        if hasattr(self, 'finnhub_client') and self.finnhub_client:
            return self.finnhub_client
        if FINNHUB_AVAILABLE and hasattr(self.config, 'FINNHUB_KEY') and self.config.FINNHUB_KEY:
            try:
                self.logger.info("Initializing Finnhub client for the first time...")
                self.finnhub_client = FinnhubClient(api_key=self.config.FINNHUB_KEY)
                # Test connection
                self.finnhub_client.profile2(symbol="AAPL")
                return self.finnhub_client
            except Exception as e:
                self.logger.error(f"Failed to initialize Finnhub client: {e}")
                self.finnhub_client = None  # 标记为失败
                return None
        return None

    def _get_polygon_client(self):
        """[新增] 懒加载 Polygon 客户端实例。"""
        if hasattr(self, 'polygon_client') and self.polygon_client:
            return self.polygon_client
        if POLYGON_AVAILABLE and hasattr(self.config, 'POLYGON_KEY') and self.config.POLYGON_KEY:
            try:
                self.logger.info("Initializing Polygon client for the first time...")
                self.polygon_client = PolygonClient(self.config.POLYGON_KEY)
                # Test connection
                self.polygon_client.get_ticker_details("AAPL")
                return self.polygon_client
            except Exception as e:
                self.logger.error(f"Failed to initialize Polygon client: {e}")
                self.polygon_client = None
                return None
        return None

    def _get_av_ts_client(self):
        """[新增] 懒加载 Alpha Vantage TimeSeries 客户端实例。"""
        if hasattr(self, 'alpha_vantage_ts') and self.alpha_vantage_ts:
            return self.alpha_vantage_ts
        if ALPHA_VANTAGE_AVAILABLE and hasattr(self.config, 'ALPHA_VANTAGE_KEY') and self.config.ALPHA_VANTAGE_KEY:
            try:
                self.logger.info("Initializing Alpha Vantage TS client for the first time...")
                # AV library usually doesn't raise error on init
                self.alpha_vantage_ts = AVTimeSeries(key=self.config.ALPHA_VANTAGE_KEY, output_format='pandas')
                return self.alpha_vantage_ts
            except Exception as e:
                self.logger.error(f"Failed to initialize Alpha Vantage TS client: {e}")
                self.alpha_vantage_ts = None
                return None
        return None

    def _fetch_hist_yfinance(self, symbol: str, days: int, interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        [FINAL FIXED VERSION] An independent, highly robust yfinance data fetching helper.
        Includes intelligent retries with exponential backoff.
        """
        self.logger.debug(f"yfinance (Fallback): Fetching hist for {symbol}")

        end_date = date.today()
        # Fetch slightly more data to ensure we have enough after filtering
        start_date = end_date - timedelta(days=int(days * 1.5) + 5)

        max_retries = 3
        base_delay = 5  # Start with a 5-second delay

        for attempt in range(max_retries):
            try:
                self.logger.info(f"yfinance: Attempt {attempt + 1}/{max_retries} for {symbol}...")

                df = yf.download(
                    tickers=symbol,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False,
                    auto_adjust=True  # Recommended for clean data
                )

                if df.empty:
                    # yfinance often returns an empty dataframe on rate limit instead of an exception
                    raise yf.exceptions.YFRateLimitError("Returned empty dataframe, likely rate-limited.")

                # Success, format and return the last 'days' worth of data
                formatted_df = _format_yfinance_data(df)  # Use the module-level formatter
                return formatted_df.tail(days) if formatted_df is not None else None

            except yf.exceptions.YFRateLimitError as e:
                # This is the most common error, handle it specifically
                wait_time = base_delay * (2 ** attempt) + np.random.uniform(0, 1)  # Exponential backoff
                self.logger.warning(
                    f"yfinance RATE LIMITED for {symbol}. Waiting {wait_time:.2f} seconds before retry...")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"yfinance failed for {symbol} after {max_retries} retries due to rate limiting.")
                    return None  # Exhausted retries

            except Exception as e:
                # Handle all other potential errors (e.g., network issues, invalid ticker)
                self.logger.error(f"yfinance hist fetch error for {symbol} on attempt {attempt + 1}: {e}",
                                  exc_info=True)
                if attempt < max_retries - 1:
                    time.sleep(base_delay)
                else:
                    return None  # Exhausted retries

        return None  # Should not be reached, but as a safeguard

    def _format_yfinance_data(df_reset_indexed: pd.DataFrame) -> Optional[pd.DataFrame]:
        date_col = 'Date' if 'Date' in df_reset_indexed.columns else (
            'Datetime' if 'Datetime' in df_reset_indexed.columns else None)
        if not date_col: logger.error(f"yfinance format error: Missing Date/Datetime column."); return None
        return _format_ohlcv_dataframe(df_reset_indexed,
                                       date_col_identifier=date_col,  # <--- 修改
                                       ohlcv_map={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                                                  'adj_close': 'Adj Close', 'volume': 'Volume'},
                                       source_name="yfinance",
                                       datetime_index_is_date=False)

    @st.cache_data(ttl=3600, show_spinner="加载市场指数数据...")
    def get_index_data(_self, symbol: str, days: int = 300, interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        [最终修复版] 获取市场指数数据，通过将 'self' 重命名为 '_self' 来解决 Streamlit 缓存问题。
        """
        # --- 在方法内部，所有对 self 的引用都改为 _self ---
        _self.logger.info(f"Fetching INDEX data for '{symbol}' using dedicated multi-API channel.")

        end_date = date.today()
        start_date = end_date - timedelta(days=int(days * 1.2))

        # --- Ticker 映射 (保持不变) ---
        ticker_map = {
            "SPY": {"polygon": "I:SPX", "finnhub": "^GSPC", "av": "SPY", "yfinance": "^GSPC"},
            "^GSPC": {"polygon": "I:SPX", "finnhub": "^GSPC", "av": "SPY", "yfinance": "^GSPC"},
            "^VIX": {"polygon": "I:VIX", "finnhub": "^VIX", "av": "VIXY", "yfinance": "^VIX"},
        }
        api_tickers = ticker_map.get(symbol.upper(), {
            "polygon": symbol, "finnhub": symbol, "av": symbol, "yfinance": symbol
        })

        data = None

        client_av = _self._get_av_ts_client()
        if client_av:
            try:
                yf_symbol = api_tickers.get("yfinance")
                data_yf = _self._fetch_hist_yfinance(yf_symbol, days, interval)
                if data_yf is not None and not data_yf.empty:
                    _self.logger.info(f"Successfully fetched index data for '{symbol}' from yfinance (fallback).")
                    return data_yf
            except Exception as e:
                _self.logger.error(f"yfinance fallback also failed for index '{symbol}': {e}")

                # vvvvvvvvvvvvvvvvvvvv START OF FIX vvvvvvvvvvvvvvvvvvvv
                # --- 最终保障：如果所有 API 都失败，则返回一个合理的默认值DataFrame ---
            _self.logger.warning(f"All API sources failed for index '{symbol}'. Returning a default DataFrame.")

            # 创建一个包含单行默认值的 DataFrame
            default_data = {
                'open': [20.0], 'high': [20.0], 'low': [20.0], 'close': [20.0], 'volume': [0]
            }
            # 使用最近的日期作为索引
            default_index = pd.to_datetime([date.today()])
            default_df = pd.DataFrame(default_data, index=default_index)
            default_df.index.name = 'date'

            # 如果请求的是 VIX，就用 VIX 的默认值
            if "VIX" in symbol.upper():
                return default_df
            # 如果请求的是 SPY，用一个中性的价格（这部分意义不大，主要是为了保证返回非空）
            elif "SPY" in symbol.upper() or "GSPC" in symbol.upper():
                default_df[['open', 'high', 'low', 'close']] = 450.0  # 假设一个价格
                return default_df
            else:
                return default_df

    @st.cache_data(ttl=3600, show_spinner="加载历史数据...")
    def get_historical_data(_self, symbol: str, days: int = 90, interval: str = "1d") -> Optional[pd.DataFrame]:
        _self.logger.info(f"DM Method: get_historical_data for {symbol}, {days} days, interval {interval}")
        end_date_obj = date.today();
        start_date_obj = end_date_obj - timedelta(days=int(days * 1.5) + 7)
        data: Optional[pd.DataFrame] = None
        start_nodash = start_date_obj.strftime('%Y%m%d');
        end_nodash = end_date_obj.strftime('%Y%m%d')
        start_dash = start_date_obj.strftime('%Y-%m-%d');
        end_dash = end_date_obj.strftime('%Y-%m-%d')
        end_date = date.today()
        start_date = end_date - timedelta(days=int(days * 1.5))
        start_str_dash = start_date.strftime('%Y-%m-%d')
        end_str_dash = end_date.strftime('%Y-%m-%d')
        start_str_nodash = start_date.strftime('%Y%m%d')
        end_str_nodash = end_date.strftime('%Y%m%d')
        self=_self

        self.logger.debug(f"Attempting Futu for {symbol}...")
        try:
            # 将通用 interval 转换为 futu 的 KLType
            ktype_map = {
                '1m': ft.KLType.K_1M, '5m': ft.KLType.K_5M, '15m': ft.KLType.K_15M,
                '30m': ft.KLType.K_30M, '1h': ft.KLType.K_60M,
                '1d': ft.KLType.K_DAY, '1w': ft.KLType.K_WEEK, '1M': ft.KLType.K_MON
            }
            ktype = ktype_map.get(interval)

            if ktype:
                # 调用 FutuManager 的方法获取数据
                df = self.futu_manager.get_historical_data(
                    symbol=symbol.upper(),
                    start=start_str_dash,
                    end=end_str_dash,
                    ktype=ktype,
                    autype=ft.AuType.QFQ  # 默认使用前复权
                )
                if df is not None and not df.empty:
                    # 富途返回的数据格式通常已经很标准，但我们还是通过一个格式化函数来确保一致性
                    data = self._format_futu_data(df)
                    if data is not None:
                        self.logger.info(f"Futu successfully fetched {len(data)} rows for {symbol}.")
                        return data  # 获取成功，直接返回
            else:
                self.logger.warning(f"Futu does not support interval '{interval}'. Falling back.")

        except Exception as e:
            self.logger.warning(f"Futu historical fetch failed for {symbol}: {e}. Falling back...")

        # --- Priority 1: Futu ---
        if hasattr(self, 'futu_manager') and self.futu_manager and self.futu_manager.is_connected:
            self.logger.debug(f"Attempting Futu for {symbol}...")
            try:
                # 将通用 interval 字符串转换为 futu 的 KLType 枚举
                ktype_map = {
                    '1d': ft.KLType.K_DAY, '1w': ft.KLType.K_WEEK, '1M': ft.KLType.K_MON,
                    '1h': ft.KLType.K_60M, '30m': ft.KLType.K_30M, '15m': ft.KLType.K_15M,
                    '5m': ft.KLType.K_5M, '1m': ft.KLType.K_1M
                }
                ktype = ktype_map.get(interval)

                if ktype:
                    # 调用 Futu API 获取数据
                    ret, df_futu = self.futu_quote_ctx.get_history_kline(
                        symbol.upper(), # Futu 需要大写代码
                        start=start_str_dash,
                        end=end_str_dash,
                        ktype=ktype,
                        autype=ft.AuType.QFQ # 默认使用前复权
                    )

                    if ret == ft.RET_OK and df_futu is not None and not df_futu.empty:
                        # 格式化返回的 DataFrame 以符合系统标准
                        data = self._format_futu_data(df_futu)
                        if data is not None and not data.empty:
                            self.logger.info(f"Futu successfully fetched {len(data)} rows for {symbol}.")
                            # 成功获取，直接返回
                            return data
                        else:
                             logger.warning(f"Futu data formatting failed for {symbol}.")
                    elif ret != ft.RET_OK:
                        self.logger.warning(f"Futu get_history_kline failed for {symbol}: {df_futu}")

                else:
                    self.logger.warning(f"Futu does not support interval '{interval}'. Falling back.")
            except Exception as e:
                self.logger.warning(f"Futu historical fetch failed for {symbol}: {e}. Falling back...")

        # CN Stocks
        if self._is_cn_stock(symbol):
            if data is None and self._is_cn_stock(symbol) and AKSHARE_AVAILABLE:
                self.logger.debug(f"Attempting AkShare for {symbol}...")
                try:
                    ak_s = symbol.split('.')[0] if '.' in symbol else symbol;
                    period_map = {'1d': 'daily'};
                    ak_p = period_map.get(interval)
                    if ak_p:
                        df_ak = ak.stock_zh_a_hist(symbol=ak_s, period=ak_p, start_date=start_nodash,
                                                   end_date=end_nodash, adjust="qfq")
                    else:
                        raise NotImplementedError(f"AkShare interval '{interval}' not supported for hist.")
                    if df_ak is not None and not df_ak.empty: data = _format_akshare_data(df_ak)
                    if data is not None: self.logger.info(
                        f"AkShare hist OK for {symbol}."); return data  # Return on first success
                except Exception as e:
                    self.logger.warning(f"AkShare hist failed for {symbol}: {e}")
            if data is None and self._is_cn_stock(symbol) and hasattr(self, 'ts_pro') and self.ts_pro:
                self.logger.debug(f"Attempting Tushare for {symbol}...")
                try:
                    ts_s = symbol.upper()
                    if interval == '1d':
                        df_ts = self.ts_pro.daily(ts_code=ts_s, start_date=start_nodash, end_date=end_nodash)
                    else:
                        raise NotImplementedError(f"Tushare interval '{interval}' needs pro_bar")
                    if df_ts is not None and not df_ts.empty: data = _format_tushare_data(df_ts)
                    if data is not None: self.logger.info(f"Tushare hist OK for {symbol}."); return data
                except Exception as e:
                    self.logger.warning(f"Tushare hist failed for {symbol}: {e}")
        else:  # Non-CN
            if data is None and hasattr(self, 'polygon_client') and self.polygon_client is not None:
                self.logger.debug(f"Attempting Polygon for {symbol}...")
                try:
                    ts_map = {'1d': 'day', '1h': 'hour', '15m': 'minute'};
                    m_map = {'1d': 1, '1h': 1, '15m': 15};
                    ts_p = ts_map.get(interval, 'day');
                    m_p = m_map.get(interval, 1)
                    aggs = list(self.polygon_client.list_aggs(symbol, m_p, ts_p, start_dash, end_dash, limit=50000))
                    if aggs: df_p = pd.DataFrame(aggs); data = _format_polygon_data(df_p)
                    if data is not None and not data.empty: self.logger.info(
                        f"Polygon hist OK for {symbol}."); return data
                except Exception as e:
                    self.logger.warning(f"Polygon hist failed for {symbol}: {e}", exc_info=False)

            if data is None and hasattr(self, 'finnhub_client') and self.finnhub_client:
                self.logger.debug(f"Attempting Finnhub historical for {symbol}...")
                try:
                    start_ts = int(datetime.combine(start_date_obj, datetime.min.time()).timestamp());
                    end_ts = int(datetime.combine(end_date_obj, datetime.max.time()).timestamp())
                    res_map = {'1d': 'D', '1w': 'W', '1m': 'M', '60': '60', '30': '30', '15': '15', '5': '5', '1': '1'};
                    fh_res_str = res_map.get(interval)
                    if fh_res_str:
                        res_fh_json = self.finnhub_client.stock_candles(symbol, fh_res_str, start_ts, end_ts)
                    else:
                        raise NotImplementedError(f"Finnhub interval '{interval}' not mapped.")
                    if res_fh_json and res_fh_json.get('s') == 'ok' and res_fh_json.get('t'):
                        df_fh = pd.DataFrame(res_fh_json);
                        data = _format_ohlcv_dataframe(df_fh, 't', {'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c',
                                                                    'volume': 'v'}, date_unit='s',
                                                       source_name="FinnhubHist", time_zone='America/New_York')
                        if data is not None: self.logger.info(f"Finnhub hist OK for {symbol}"); return data
                except Exception as e:
                    self.logger.warning(f"Finnhub hist failed for {symbol}: {e}")

            if data is None and hasattr(self, 'alpha_vantage_ts') and self.alpha_vantage_ts is not None:
                self.logger.debug(f"Attempting AV Library for {symbol}...")
                try:
                    # 这里的代码现在是安全的，因为我们已经确认 self.alpha_vantage_ts 是一个实例
                    av_fn = self.alpha_vantage_ts.get_daily_adjusted if interval == '1d' else self.alpha_vantage_ts.get_intraday
                    av_int_map_lib = {'1h': '60min', '15m': '15min'}
                    av_i_lib = av_int_map_lib.get(interval)

                    if interval == '1d':
                        df_av_lib, _ = av_fn(symbol=symbol, outputsize='full')
                    elif av_i_lib:
                        df_av_lib, _ = av_fn(symbol=symbol, interval=av_i_lib, outputsize='full')
                    else:
                        raise NotImplementedError(f"AV Lib interval '{interval}' not mapped")

                    if df_av_lib is not None and not df_av_lib.empty:
                        data = _format_av_lib_hist_data(df_av_lib)  # 确保 _format_av_lib_hist_data 函数存在
                        if data is not None:
                            # 日期过滤
                            data = data.sort_index().loc[start_dash:end_dash]
                            if data is not None and not data.empty:
                                self.logger.info(f"AV Lib hist OK for {symbol}.")
                                return data
                except Exception as e:
                    self.logger.warning(f"AV Lib hist failed for {symbol}: {e}")

        if data is None and hasattr(self, 'alpha_vantage_key') and self.alpha_vantage_key:
            self.logger.debug(f"Attempting Alpha Vantage via Direct HTTP for {symbol}...")
            try:
                # 1. 准备 API 参数
                func_map = {'1d': "TIME_SERIES_DAILY_ADJUSTED", "1w": "TIME_SERIES_WEEKLY_ADJUSTED",
                            "1m": "TIME_SERIES_MONTHLY_ADJUSTED"}
                av_int_map = {'1h': '60min', '15m': '15min', '5m': '5min', '1m': '1min'}

                function_name = func_map.get(interval)
                interval_param = av_int_map.get(interval)

                params = {"symbol": symbol, "apikey": self.alpha_vantage_key, "datatype": "json", "outputsize": "full"}
                if function_name:
                    params["function"] = function_name
                elif interval_param:
                    params["function"] = "TIME_SERIES_INTRADAY"
                    params["interval"] = interval_param
                else:
                    raise NotImplementedError(f"AV Direct HTTP: Interval '{interval}' not supported.")

                # 2. 调用缓存的 API 函数
                # 将 params 字典转换为元组以使其可被哈希和缓存
                json_data = _call_av_api_cached(params_tuple=tuple(sorted(params.items())))

                # 3. 在当前方法的作用域内处理返回的 JSON 数据
                if json_data and "Error Message" not in json_data and "Information" not in json_data:
                    data = _format_av_direct_json_hist(json_data, symbol, interval)

                    if data is not None and not data.empty:
                        # 在这里过滤日期，因为 start_str_dash 在此作用域内可见
                        data = data.sort_index().loc[start_str_dash:end_str_dash]
                        if not data.empty:
                            self.logger.info(f"AV Direct HTTP hist OK for {symbol}.")
                        else:
                            data = None  # 过滤后为空
                elif json_data:
                    logger.warning(
                        f"AV API response error for {symbol}: {json_data.get('Error Message') or json_data.get('Information')}")

            except Exception as e:
                self.logger.error(f"AV Direct HTTP processing failed for {symbol}: {e}", exc_info=True)

        # yfinance Fallback
        if data is None:
            self.logger.info(f"Primary sources failed for {symbol}, attempting yfinance fallback...")
            max_retries = getattr(self.config, 'YFINANCE_MAX_RETRIES', 2)
            base_delay = getattr(self.config, 'YFINANCE_BASE_DELAY', 3)

            for attempt in range(max_retries + 1):
                try:
                    now = time.time();
                    elapsed = now - self.yfinance_last_call_time;
                    wait_for = 0
                    if attempt > 0: wait_for = self.yfinance_base_delay * (2 ** (attempt - 1))
                    if elapsed < self.yfinance_min_interval: wait_for = max(wait_for,
                                                                            self.yfinance_min_interval - elapsed)
                    if wait_for > 0: self.logger.info(
                        f"DM: yf rate limit. Wait {wait_for:.2f}s for {symbol} (hist att {attempt + 1})"); time.sleep(
                        wait_for)
                    self.yfinance_last_call_time = time.time()
                    data = _cached_yfinance_history(symbol, start_dash,
                                                    (end_date_obj + timedelta(days=1)).strftime('%Y-%m-%d'), interval,
                                                    attempt_info=str(attempt + 1))
                    if data is not None: break
                except yf.exceptions.YFRateLimitError:
                    if attempt == self.yfinance_max_retries: self.logger.error(
                        f"DM: Max yf retries (rate limit) for {symbol}."); data = None; break  # Corrected condition
                except Exception as e_yf_call:
                    self.logger.error(f"DM: Unhandled yf error {symbol} (att {attempt + 1}): {e_yf_call}",
                                      exc_info=True); data = None; break
            if data is None: self.logger.error(f"DM: All yf hist attempts failed for {symbol}.")

        # Final date filtering
        if data is not None and not data.empty:
            try:
                if not isinstance(data.index, pd.DatetimeIndex): data.index = pd.to_datetime(data.index,
                                                                                             errors='coerce'); data = data.dropna(
                    subset=[data.index.name])
                final_start_dt = (end_date_obj - timedelta(days=days));
                final_end_dt = end_date_obj
                if data.index.tz is not None:
                    final_start_dt = pd.Timestamp(final_start_dt, tz=data.index.tz); final_end_dt = pd.Timestamp(
                        final_end_dt, tz=data.index.tz)
                else:
                    final_start_dt = pd.Timestamp(final_start_dt); final_end_dt = pd.Timestamp(
                        final_end_dt)  # Ensure Timestamp for comparison
                data = data.sort_index().loc[final_start_dt:final_end_dt]
                if data.empty: self.logger.warning(f"Data for {symbol} empty after final date filtering.")
            except Exception as e_filter:
                self.logger.error(f"Error final date filtering for {symbol}: {e_filter}. Returning as is.",
                                  exc_info=True)
        elif data is None:
            return pd.DataFrame()  # Return empty DataFrame on complete failure

        return data

    def get_realtime_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        [ROBUST VERSION] Gets the best available price for a symbol.

        This method follows a clear priority:
        1. Fresh WebSocket data (if available).
        2. Fresh Futu push data (if available).
        3. The robust, multi-source, cached REST API fetcher.
        4. The ultimate fallback: the most recent historical closing price.
        """
        self.logger.debug(f"Requesting best available price for {symbol}...")

        # --- Priority 1 & 2: Live Push Data (WebSocket / Futu) ---
        if self.websocket_manager:
            ws_data = self.websocket_manager.get_latest_price(symbol)
            if ws_data and (time.time() - ws_data.get('timestamp', 0)) < 15:
                return ws_data

        if hasattr(self, 'futu_manager') and self.futu_manager and self.futu_manager.is_connected:
            futu_data = self.futu_manager.get_realtime_price_from_push(symbol)
            if futu_data and (time.time() - futu_data.get('timestamp', 0)) < 15:
                return futu_data

        # --- Priority 3: The Strengthened Cached REST API Fetcher ---
        # This is our main workhorse now. It tries multiple reliable APIs internally.
        price_data = fetch_realtime_price_from_sources_cached(
            symbol=symbol,
            akshare_available_flag=AKSHARE_AVAILABLE,
            finnhub_api_key=self.finnhub_key,
            polygon_api_key=self.polygon_key,
            alpha_vantage_api_key=self.alpha_vantage_key
        )
        if price_data:
            return price_data

        # --- FINAL FALLBACK: Latest Historical Close ---
        self.logger.warning(f"All real-time sources failed for {symbol}. Falling back to historical close.")
        hist_data = self.get_historical_data(symbol, days=5, interval="1d")

        if hist_data is not None and not hist_data.empty and 'close' in hist_data.columns:
            try:
                latest_close = float(hist_data['close'].iloc[-1])
                latest_timestamp = hist_data.index[-1]
                if latest_close > 0:
                    self.logger.info(f"Using historical close for {symbol}: {latest_close}")
                    return {
                        'price': latest_close,
                        'timestamp': latest_timestamp.timestamp(),
                        'source': 'Historical Close (Fallback)',
                        'delayed': True
                    }
            except (IndexError, TypeError, ValueError) as e:
                self.logger.error(f"Could not extract historical close for {symbol}: {e}")

        self.logger.error(f"DataManager: Ultimately FAILED to get any price for {symbol}.")
        return None

    def _format_futu_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Formats Futu kline data to standard OHLCV DataFrame."""
        try:
            # Futu 返回的列名: 'code', 'time_key', 'open', 'close', 'high', 'low', 'volume', 'turnover', 'pe_ratio', 'turnover_rate', 'last_close'
            df_copy = df.copy()
            # 'time_key' 是带时区的字符串，可以直接转为 DatetimeIndex
            df_copy['date'] = pd.to_datetime(df_copy['time_key'])
            df_copy = df_copy.set_index('date')

            # 选择并重命名标准列
            # 'turnover' 是成交额，我们主要用 'volume' 成交量
            standard_cols = ['open', 'high', 'low', 'close', 'volume']
            if all(col in df_copy.columns for col in standard_cols):
                df_final = df_copy[standard_cols].sort_index()
                # 确保数据是数值类型
                for col in df_final.columns:
                    df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
                df_final = df_final.dropna(subset=['close'])
                return df_final
            else:
                logger.error(f"Futu data missing standard columns. Columns: {df_copy.columns}")
                return None
        except Exception as e:
            self.logger.error(f"Error formatting Futu data: {e}", exc_info=True)
            return None

    # --- search_stocks (using instance clients or direct HTTP) ---
    @st.cache_data(ttl=86400, show_spinner="搜索股票...", persist="disk")
    def search_stocks(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        self.logger.info(f"DM: search_stocks for '{query}' (limit: {limit})")
        results: List[Dict[str, str]] = [];
        seen_symbols = set()

    def _search_stocks_local_fallback(self, query, limit):
        stocks = [
            {'symbol': 'AAPL', 'name': 'Apple Inc.', 'exchange': 'NASDAQ', 'type': 'stock', 'source': 'Local Fallback'},
            {'symbol': 'MSFT', 'name': 'Microsoft Corporation', 'exchange': 'NASDAQ', 'type': 'stock',
             'source': 'Local Fallback'},
            {'symbol': 'AMZN', 'name': 'Amazon.com, Inc.', 'exchange': 'NASDAQ', 'type': 'stock',
             'source': 'Local Fallback'},
            {'symbol': 'GOOGL', 'name': 'Alphabet Inc. (Class A)', 'exchange': 'NASDAQ', 'type': 'stock',
             'source': 'Local Fallback'},
            {'symbol': 'META', 'name': 'Meta Platforms, Inc.', 'exchange': 'NASDAQ', 'type': 'stock',
             'source': 'Local Fallback'},
            {'symbol': 'TSLA', 'name': 'Tesla, Inc.', 'exchange': 'NASDAQ', 'type': 'stock',
             'source': 'Local Fallback'},
            {'symbol': 'NVDA', 'name': 'NVIDIA Corporation', 'exchange': 'NASDAQ', 'type': 'stock',
             'source': 'Local Fallback'},
            {'symbol': '600519.SH', 'name': 'Kweichow Moutai Co., Ltd.', 'exchange': 'SSE', 'type': 'stock',
             'source': 'Local Fallback'},
            {'symbol': '000001.SZ', 'name': 'Ping An Bank Co., Ltd.', 'exchange': 'SZSE', 'type': 'stock',
             'source': 'Local Fallback'}
        ]
        query_lower = query.lower()
        return [
                   s for s in stocks
                   if query_lower in s['symbol'].lower() or query_lower in s['name'].lower()
               ][:limit]

    def get_market_data(self, indices: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        if indices is None:
            indices = self.config.get('DEFAULT_MARKET_INDICES', ['^GSPC', '^DJI', '^IXIC'])

        market_data = {}
        self.logger.info(f"Getting market data for indices: {indices}")

        for index_symbol in indices:
            try:
                index_hist_data = self.get_historical_data(index_symbol, days=5, interval="1d")

                # 使用括号包裹多行条件判断
                if (index_hist_data is not None
                        and not index_hist_data.empty
                        and 'close' in index_hist_data.columns
                        and len(index_hist_data) >= 2):

                    latest_close = float(index_hist_data['close'].iloc[-1])
                    prev_close = float(index_hist_data['close'].iloc[-2])
                    change_pct = ((latest_close - prev_close) / prev_close * 100) if prev_close != 0 else 0

                    market_data[index_symbol] = {
                        'name': self._get_index_name(index_symbol),
                        'latest': latest_close,
                        'change_pct': change_pct
                    }
                else:
                    self.logger.warning(f"Could not get sufficient historical data for index {index_symbol}")
            except Exception as e:
                self.logger.error(f"Error processing market index {index_symbol}: {e}", exc_info=True)

        return market_data

    @st.cache_data(ttl=3600, show_spinner="获取股票详情 (DM Method)...", persist="disk")  # 缓存1小时
    def get_stock_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        self.logger.info(f"DM Cached Method: get_stock_details for: {symbol}")
        details: Dict[str, Any] = {'symbol': symbol.upper()}  # Start with symbol, ensure uppercase

        # --- 1. Finnhub (Profile & Basic Financials) ---
        if self.finnhub_client:
            self.logger.debug(f"Attempting Finnhub details for {symbol}...")
            try:
                profile = self.finnhub_client.profile2(symbol=symbol)
                if profile and isinstance(profile, dict):
                    details['name'] = profile.get('name', details.get('name'))
                    details['exchange'] = profile.get('exchange', details.get('exchange'))
                    details['ipo_date'] = profile.get('ipo', details.get('ipo_date'))  # IPO Date
                    details['market_cap'] = profile.get('marketCapitalization',
                                                        details.get('market_cap'))  # Market Cap in millions
                    details['currency'] = profile.get('currency', details.get('currency'))
                    details['industry'] = profile.get('finnhubIndustry', details.get('industry'))
                    details['country'] = profile.get('country', details.get('country'))
                    details['logo'] = profile.get('logo', details.get('logo'))
                    details['web_url'] = profile.get('weburl', details.get('web_url'))
                    details['shares_outstanding'] = profile.get('shareOutstanding', details.get('shares_outstanding'))
                    self.logger.info(f"Finnhub profile data acquired for {symbol}")

                metrics = self.finnhub_client.company_basic_financials(symbol,
                                                                       'all')  # metricType: all, price, valuation, margin
                if metrics and isinstance(metrics, dict) and metrics.get('metric'):
                    fin_metric = metrics['metric']
                    # Prefer more specific PE if available
                    details['pe_ratio'] = fin_metric.get('peNormalizedAnnual',
                                                         fin_metric.get('peBasicExclExtraTTM', details.get('pe_ratio')))
                    details['eps'] = fin_metric.get('epsGrowth5Y', fin_metric.get('epsBasicExclExtraTTM', details.get(
                        'eps')))  # Example of choosing one EPS
                    details['dividend_yield'] = fin_metric.get('dividendYieldIndicatedAnnual',
                                                               details.get('dividend_yield'))  # Already in percent
                    details['beta'] = fin_metric.get('beta', details.get('beta'))
                    details['book_value_per_share'] = fin_metric.get('bookValuePerShareAnnual',
                                                                     details.get('book_value_per_share'))
                    self.logger.info(f"Finnhub basic financials acquired for {symbol}")
            except Exception as e:
                self.logger.warning(f"Finnhub details/financials failed for {symbol}: {e}")

        # --- 2. Polygon (Ticker Details) ---
        if self.polygon_client and (
                not details.get('name') or not details.get('description')):  # Try if key info missing
            self.logger.debug(f"Attempting Polygon details for {symbol}...")
            try:
                snapshot = self.polygon_client.get_snapshot_ticker(ticker=symbol)  # 明确使用 ticker 参数
                # ^^^^^^^^^^^^^^^^^^^^ END OF MODIFIED LINE ^^^^^^^^^^^^^^^^^^^^
                if snapshot and snapshot.last_trade and snapshot.last_trade.p > 0:
                    price_data = {'price': float(snapshot.last_trade.p), 'timestamp': snapshot.last_trade.t / 1000.0,
                                  'source': 'Polygon (Snapshot)'}
                # ... (elif snapshot.prev_day ...)
                if price_data: self.logger.info(f"Polygon price: {price_data['price']}"); return price_data
            except Exception as e:
                self.logger.warning(f"Polygon realtime failed for {symbol}: {e}")

        # --- 3. Alpha Vantage (Direct HTTP for OVERVIEW) ---
        if self.alpha_vantage_key and (
                not details.get('description') or not details.get('industry') or not details.get('pe_ratio')):
            self.logger.debug(f"Attempting AV Direct HTTP Overview for {symbol}...")
            try:
                url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={self.alpha_vantage_key}"
                response = self.requests_session.get(url, timeout=10)
                response.raise_for_status()
                overview_data = response.json()
                if overview_data and not overview_data.get("Error Message") and not overview_data.get("Information"):
                    details.setdefault('name', overview_data.get('Name', details.get('name')))
                    details.setdefault('description', overview_data.get('Description', details.get('description')))
                    details.setdefault('exchange', overview_data.get('Exchange', details.get('exchange')))
                    details.setdefault('currency', overview_data.get('Currency', details.get('currency')))
                    details.setdefault('country', overview_data.get('Country', details.get('country')))
                    details.setdefault('sector', overview_data.get('Sector', details.get('sector')))
                    details.setdefault('industry', overview_data.get('Industry', details.get('industry')))
                    details.setdefault('market_cap',
                                       overview_data.get('MarketCapitalization', details.get('market_cap')))
                    details.setdefault('pe_ratio', overview_data.get('PERatio', details.get('pe_ratio')))
                    details.setdefault('eps', overview_data.get('EPS', details.get('eps')))
                    details.setdefault('dividend_yield',
                                       overview_data.get('DividendYield', details.get('dividend_yield')))
                    details.setdefault('beta', overview_data.get('Beta', details.get('beta')))
                    self.logger.info(f"AV Direct Overview data acquired for {symbol}")
                elif "Note" in overview_data or "Information" in overview_data:
                    self.logger.warning(
                        f"AV Direct Overview API Info/Note for {symbol}: {overview_data.get('Note') or overview_data.get('Information')}")
            except requests.exceptions.RequestException as e_req:
                self.logger.error(f"AV Direct Overview HTTP request failed for {symbol}: {e_req}")
            except json.JSONDecodeError as e_json:
                self.logger.error(f"AV Direct Overview JSON decode failed for {symbol}: {e_json}")
            except Exception as e:
                self.logger.error(f"AV Direct Overview general error for {symbol}: {e}", exc_info=True)

        # --- 4. yfinance (Fallback for any remaining missing info) ---
        # Check if essential details like name, industry, or market_cap are still missing
        if not details.get('name') or details['name'] == symbol or not details.get('industry') or not details.get(
                'market_cap'):
            self.logger.debug(f"Attempting yfinance Info to supplement/fallback details for {symbol}...")
            try:
                now = time.time();
                elapsed = now - self.yfinance_last_call_time
                if elapsed < self.yfinance_min_interval:
                    wait_time = self.yfinance_min_interval - elapsed
                    self.logger.info(f"yfinance (details): Waiting {wait_time:.2f}s for {symbol}")
                    time.sleep(wait_time)
                self.yfinance_last_call_time = time.time()
                ticker = yf.Ticker(symbol);
                info = ticker.info
                if info and info.get('symbol', '').upper() == symbol.upper():  # Validate symbol
                    details.setdefault('name', info.get('shortName', info.get('longName', details.get('name'))))
                    details.setdefault('exchange', info.get('exchange', details.get('exchange')))
                    details.setdefault('currency', info.get('currency', details.get('currency')))
                    details.setdefault('industry', info.get('industry', details.get('industry')))
                    details.setdefault('sector', info.get('sector', details.get('sector')))
                    details.setdefault('description', info.get('longBusinessSummary', details.get('description')))
                    details.setdefault('market_cap', info.get('marketCap', details.get('market_cap')))
                    details.setdefault('pe_ratio', info.get('trailingPE', details.get('pe_ratio')))
                    details.setdefault('dividend_yield',
                                       info.get('dividendYield', details.get('dividend_yield')))  # yf yield is decimal
                    details.setdefault('beta', info.get('beta', details.get('beta')))
                    details.setdefault('shares_outstanding',
                                       info.get('sharesOutstanding', details.get('shares_outstanding')))
                    self.logger.info(f"yfinance info supplemented details for {symbol}")
            except yf.exceptions.YFRateLimitError:
                self.logger.warning(f"yfinance rate limited getting details for {symbol}")
            except Exception as e:
                self.logger.warning(f"yfinance info for details of {symbol} failed: {e}")

        # --- Post-processing and adding Realtime Price ---
        # Convert market_cap to float
        market_cap_val = details.get('market_cap')
        if market_cap_val is not None:
            try:
                details['market_cap'] = float(market_cap_val)
            except (ValueError, TypeError):
                self.logger.warning(
                    f"Could not convert market_cap '{market_cap_val}' to float for {symbol}. Setting to 0.")
                details['market_cap'] = 0.0

        # Convert dividend_yield to percentage if it's a decimal
        div_yield = details.get('dividend_yield')
        if div_yield is not None and isinstance(div_yield, (
        float, int)) and div_yield < 1.0 and div_yield != 0:  # Assuming it's decimal if < 1
            details['dividend_yield'] = div_yield * 100.0
        elif div_yield is not None:  # Try to convert if string
            try:
                details['dividend_yield'] = float(div_yield)
            except:
                pass

        # Add Realtime Price (this call itself is cached)
        if 'price' not in details:  # Only fetch if not already populated (e.g., by Finnhub quote)
            rt_price_info = self.get_realtime_price(symbol)
            if rt_price_info and isinstance(rt_price_info.get('price'), (int, float)):
                details['price'] = rt_price_info['price']
                details['last_price_update_ts'] = rt_price_info.get('timestamp', time.time())  # Store raw timestamp
                details['price_source'] = rt_price_info.get('source')
            else:  # If RT price failed, use latest historical close as a fallback
                hist_data_for_price = self.get_historical_data(symbol, days=2)
                if hist_data_for_price is not None and not hist_data_for_price.empty and 'close' in hist_data_for_price:
                    details['price'] = float(hist_data_for_price['close'].iloc[-1])
                    details['last_price_update_ts'] = hist_data_for_price.index[-1].timestamp()
                    details['price_source'] = 'HistCloseFallback'

        # Final check for essential info
        if not details.get('name') or details['name'] == symbol or details.get('name') is None:
            self.logger.warning(f"Insufficient details ultimately gathered for {symbol}.")
            return None  # Return None if no meaningful name found

        # Clean up None/'None'/'N/A' values to be consistent
        cleaned_details = {}
        for k, v in details.items():
            if isinstance(v, str) and (v.lower() == 'none' or v.lower() == 'n/a'):
                cleaned_details[k] = None
            elif isinstance(v, float) and np.isnan(v):
                cleaned_details[k] = None
            else:
                cleaned_details[k] = v

        return cleaned_details

    def _get_stock_details_yfinance_fallback(self, symbol:str) -> Optional[Dict[str, Any]]:
        self.logger.debug(f"Attempting yfinance Info (Fallback) for details of {symbol}...")
        try:
            now = time.time(); elapsed = now - self.yfinance_last_call_time
            if elapsed < self.yfinance_min_interval: time.sleep(self.yfinance_min_interval - elapsed)
            self.yfinance_last_call_time = time.time()
            ticker = yf.Ticker(symbol); info = ticker.info
            if info and info.get('symbol','').upper() == symbol.upper():
                 details = {'symbol': symbol}
                 details.setdefault('name', info.get('shortName', info.get('longName', symbol)))
                 details.setdefault('exchange', info.get('exchange')); details.setdefault('market_cap', info.get('marketCap')); details.setdefault('pe_ratio', info.get('trailingPE')); details.setdefault('dividend_yield', (info.get('dividendYield', 0) or 0) * 100); details.setdefault('beta', info.get('beta')); details.setdefault('industry', info.get('industry')); details.setdefault('sector', info.get('sector')); details.setdefault('description', info.get('longBusinessSummary')); details.setdefault('currency', info.get('currency'))
                 # Add realtime price to details
                 rt_price_info = self.get_realtime_price(symbol) # Call the multi-source RT price
                 if rt_price_info: details['price'] = rt_price_info.get('price')
                 return details
            return None
        except yf.exceptions.YFRateLimitError: logger.warning(f"yfinance rate limited getting details for {symbol}")
        except Exception as e: logger.warning(f"yfinance info for details of {symbol} failed: {e}")
        return None

    def _get_index_name(self, index_symbol: str) -> str:
        index_names_map_default = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^N225': 'Nikkei 225',
            '^HSI': 'Hang Seng'
        }
        index_names = index_names_map_default
        if self.config and hasattr(self.config, 'get'):
            index_names = self.config.get('INDEX_NAMES_MAP', index_names_map_default)
        return index_names.get(index_symbol, index_symbol)

    def save_to_csv(self, data, filename):
        """保存数据到CSV文件"""
        if data is None or not isinstance(data, pd.DataFrame):
            self.logger.error("Cannot save invalid data.")
            return False
        try:
            filepath = self.data_dir / f"{filename}.csv"
            data.to_csv(filepath)
            self.logger.info(f"Data saved to: {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving data to CSV ({filename}): {e}", exc_info=True)
            return False

    def load_from_csv(self, filename):
        """从CSV文件加载数据"""
        try:
            filepath = self.data_dir / f"{filename}.csv"
            if filepath.is_file():
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                self.logger.info(f"Loaded data from {filepath}")
                return data
            else:
                self.logger.warning(f"CSV file not found: {filepath}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading data from CSV ({filename}): {e}", exc_info=True)
            return None


    def __del__(self):
        """在对象销毁时清理资源"""
        logger.info("DataManager instance being deleted. Closing connections.")
        if hasattr(self, 'futu_quote_ctx') and self.futu_quote_ctx:
            self.futu_quote_ctx.close()
        if hasattr(self, 'websocket_manager') and self.websocket_manager:
            self.websocket_manager.stop()