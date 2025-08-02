import pandas as pd
import numpy as np
import logging
import yfinance as yf
from datetime import datetime, timedelta, date
import requests
import time
import os
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import streamlit as st
import random

# --- API Client Libraries ---
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
    from alpha_vantage.timeseries import TimeSeries
    from alpha_vantage.fundamentaldata import FundamentalData
    from alpha_vantage.searching import Searching
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    TimeSeries, FundamentalData, Searching = None, None, None; ALPHA_VANTAGE_AVAILABLE = False; logging.warning("alpha_vantage library not found. Alpha Vantage functionality disabled.")

logger = logging.getLogger(__name__)

# --- Helper function for CN stock check (Global Scope) ---
def _is_cn_stock_helper(symbol: str) -> bool:
    return isinstance(symbol, str) and (symbol.upper().endswith(('.SH', '.SZ')) or (symbol.isdigit() and len(symbol) == 6))

# --- Formatting Helper Functions (Global Scope) ---
def _format_ohlcv_dataframe(df: pd.DataFrame, date_col_name: str, ohlcv_map: Dict[str, str],
                            date_format: Optional[str] = None, volume_multiplier: int = 1,
                            source_name: str = "API", date_unit: Optional[str] = None,
                            datetime_index: bool = True) -> Optional[pd.DataFrame]:
    try:
        logger.debug(f"Formatting data from {source_name} ({len(df)} rows) for date_col '{date_col_name}'...")
        if date_col_name not in df.columns and not (datetime_index and isinstance(df.index, pd.DatetimeIndex)):
            logger.error(f"{source_name} format error: Missing date column '{date_col_name}' and not a datetime index. Available: {list(df.columns)}")
            return None
        df_copy = df.copy()
        current_ohlcv_map = {v_orig: v_std for v_std, v_orig in ohlcv_map.items() if v_orig in df_copy.columns}
        df_copy.rename(columns=current_ohlcv_map, inplace=True)
        if datetime_index and isinstance(df_copy.index, pd.DatetimeIndex): df_copy['date'] = df_copy.index
        elif date_unit: df_copy['date'] = pd.to_datetime(df_copy[date_col_name], unit=date_unit, errors='coerce', utc=True if date_unit=='ms' else False)
        else: df_copy['date'] = pd.to_datetime(df_copy[date_col_name], format=date_format, errors='coerce')
        df_copy = df_copy.dropna(subset=['date'])
        if df_copy.empty: logger.warning(f"{source_name}: No valid dates after conversion for {date_col_name}."); return None
        df_copy = df_copy.set_index('date')
        standard_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close']
        cols_to_keep = [col for col in standard_cols if col in df_copy.columns]
        if 'adj_close' in cols_to_keep and 'close' in cols_to_keep : df_copy['close'] = df_copy['adj_close']; cols_to_keep.remove('adj_close')
        elif 'adj_close' in cols_to_keep and 'close' not in cols_to_keep: df_copy.rename(columns={'adj_close': 'close'}, inplace=True); cols_to_keep = [col if col != 'adj_close' else 'close' for col in cols_to_keep]
        if 'close' not in cols_to_keep: logger.error(f"{source_name} format error: Missing 'close'. Mapped: {cols_to_keep}"); return None
        df_final = df_copy[list(set(cols_to_keep))].sort_index()
        if 'volume' in df_final.columns and volume_multiplier != 1: df_final['volume'] = df_final['volume'] * volume_multiplier
        df_final = df_final.apply(pd.to_numeric, errors='coerce'); df_final = df_final.dropna(subset=['close'])
        if df_final.empty: logger.warning(f"{source_name} data empty after formatting."); return None
        logger.debug(f"{source_name} data formatted: {len(df_final)} rows.")
        return df_final
    except Exception as e: logger.error(f"Error formatting {source_name} (date_col:'{date_col_name}', map:{ohlcv_map}): {e}", exc_info=True); return None

def _format_akshare_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    return _format_ohlcv_dataframe(df, '日期', {'开盘':'open','收盘':'close','最高':'high','最低':'low','成交量':'volume'}, volume_multiplier=100, source_name="AkShare")
def _format_tushare_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    return _format_ohlcv_dataframe(df, 'trade_date', {'open':'open','high':'high','low':'low','close':'close','vol':'volume'}, date_format='%Y%m%d', source_name="Tushare")
def _format_av_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    date_col = df.index.name or 'date'; df_reset = df.reset_index() if df.index.name is None else df
    return _format_ohlcv_dataframe(df_reset, date_col, {'1. open':'open', '2. high':'high', '3. low':'low', '4. close':'close', '5. adjusted close': 'adj_close', '6. volume':'volume', '5. volume': 'volume'}, source_name="AlphaVantage", datetime_index=True)
def _format_polygon_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    return _format_ohlcv_dataframe(df, 'timestamp', {'o':'open','h':'high','l':'low','c':'close','v':'volume'}, date_unit='ms', source_name="Polygon")
def _format_yfinance_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    date_col = 'Date' if 'Date' in df.columns else ('Datetime' if 'Datetime' in df.columns else None)
    if not date_col: logger.error(f"yfinance format error: Missing date column. Available: {df.columns}"); return None
    return _format_ohlcv_dataframe(df, date_col, {'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Adj Close':'adj_close', 'Volume':'volume'}, source_name="yfinance")

# --- Cached Data Fetching Functions ---
# YFINANCE fetcher is now simpler, rate limiting and retries handled by DataManager
@st.cache_data(ttl=3600, show_spinner="正在加载股票数据 (yf fallback)...")
def fetch_yfinance_hist_data_cached(symbol: str, start_str: str, end_str: str, interval: str ="1d", attempt_for_log: str = "N/A"):
    logger.info(f"[YF Cache Call - Attempt {attempt_for_log}] Executing yfinance API for: {symbol}")
    try:
        stock = yf.Ticker(symbol)
        data_yf = stock.history(start=start_str, end=end_str, interval=interval, auto_adjust=False, back_adjust=False)
        if data_yf is None or data_yf.empty:
            logger.warning(f"yfinance: {symbol} returned empty/None (attempt {attempt_for_log}).")
            return None # yfinance 明确返回空或None
        logger.info(f"yfinance: {symbol} SUCCEEDED, {len(data_yf)} rows (attempt {attempt_for_log}).")
        # Assuming _format_yfinance_data handles reset_index and all formatting
        return _format_yfinance_data(data_yf.reset_index())
    except yf.exceptions.YFRateLimitError as e_rate:
        logger.error(f"yfinance: Rate limit for {symbol} (Attempt {attempt_for_log}). {e_rate}")
        raise # Re-raise for DataManager to handle retry
    except Exception as e:
        logger.error(f"yfinance: Error fetching {symbol} (Attempt {attempt_for_log}): {e}", exc_info=True)
        raise # Re-raise for DataManager

@st.cache_data(ttl=3600, show_spinner="加载历史数据...")
def fetch_historical_data_from_sources_cached(
    symbol: str, start_date_obj: date, end_date_obj: date, interval: str,
    akshare_available_flag: bool, # Pass boolean availability
    tushare_token: Optional[str],
    polygon_key: Optional[str],
    finnhub_key: Optional[str],
    av_key: Optional[str]
) -> Optional[pd.DataFrame]:
    logger.info(f"[Cache Call] Primary Hist Fetch: {symbol}, {start_date_obj} to {end_date_obj}, {interval}")
    data = None
    start_nodash=start_date_obj.strftime('%Y%m%d'); end_nodash=end_date_obj.strftime('%Y%m%d')
    start_dash=start_date_obj.strftime('%Y-%m-%d'); end_dash=end_date_obj.strftime('%Y-%m-%d')

    if _is_cn_stock_helper(symbol):
        if AKSHARE_AVAILABLE and akshare_available_flag: # Use passed flag
            logger.debug(f"Attempting AkShare for {symbol}...")
            try:
                ak_s = symbol.split('.')[0] if '.' in symbol else symbol
                period_map={'1d':'daily','1w':'weekly','1m':'monthly'}; ak_p=period_map.get(interval)
                if ak_p: df_ak = ak.stock_zh_a_hist(symbol=ak_s, period=ak_p, start_date=start_nodash, end_date=end_nodash, adjust="qfq")
                else: raise NotImplementedError(f"AkShare interval '{interval}' not supported")
                if df_ak is not None and not df_ak.empty: data = _format_akshare_data(df_ak)
                if data is not None: logger.info(f"AkShare OK for {symbol} ({len(data)} rows)."); return data
            except Exception as e: logger.warning(f"AkShare failed for {symbol}: {e}")
        if data is None and TUSHARE_AVAILABLE and tushare_token:
            logger.debug(f"Attempting Tushare for {symbol}...")
            try:
                ts_api = ts.pro_api(tushare_token); ts_s = symbol.upper()
                if interval=='1d': df_ts = ts_api.daily(ts_code=ts_s, start_date=start_nodash, end_date=end_nodash)
                else: raise NotImplementedError(f"Tushare interval '{interval}' needs pro_bar")
                if df_ts is not None and not df_ts.empty: data = _format_tushare_data(df_ts)
                if data is not None: logger.info(f"Tushare OK for {symbol} ({len(data)} rows)."); return data
            except Exception as e: logger.warning(f"Tushare failed for {symbol}: {e}")
    else: # Non-CN stocks
        if POLYGON_AVAILABLE and polygon_key: # Polygon First for US
            logger.debug(f"Attempting Polygon for {symbol}...")
            try:
                poly_c = PolygonClient(polygon_key); ts_map={'1d':'day','1h':'hour','15m':'minute'}; m_map={'1d':1,'1h':1,'15m':15}
                ts_p=ts_map.get(interval,'day'); m_p=m_map.get(interval,1)
                aggs = list(poly_c.list_aggs(symbol, m_p, ts_p, start_dash, end_dash, limit=50000)) # Corrected date
                if aggs: df_p = pd.DataFrame(aggs); data = _format_polygon_data(df_p)
                if data is not None: logger.info(f"Polygon OK for {symbol} ({len(data)} rows)."); return data
            except Exception as e: logger.warning(f"Polygon failed for {symbol}: {e}")
        if data is None and FINNHUB_AVAILABLE and finnhub_key: # Finnhub Second for US/International
            logger.debug(f"Attempting Finnhub for {symbol}...")
            try:
                fh_client = FinnhubClient(api_key=finnhub_key)
                start_ts_unix = int(datetime.strptime(start_dash, '%Y-%m-%d').timestamp()); end_ts_unix = int(datetime.strptime(end_dash, '%Y-%m-%d').timestamp())
                resolution_map = {'1d':'D', '1h':'60', '15m':'15'}; fh_res = resolution_map.get(interval)
                if fh_res:
                    candle_data = fh_client.stock_candles(symbol, fh_res, start_ts_unix, end_ts_unix)
                    if candle_data and candle_data.get('s') == 'ok':
                        df_fh = pd.DataFrame(candle_data)
                        data = _format_ohlcv_dataframe(df_fh, 't', {'o':'open','h':'high','l':'low','c':'close','v':'volume'}, date_unit='s', source_name="Finnhub", datetime_index=False)
                        if data is not None: logger.info(f"Finnhub OK for {symbol}."); return data
                    else: logger.warning(f"Finnhub returned no/error data for {symbol}: {candle_data.get('s') if candle_data else 'None'}")
                else: raise NotImplementedError(f"Finnhub resolution for interval '{interval}' not mapped.")
            except Exception as e: logger.warning(f"Finnhub failed for {symbol}: {e}")
        if data is None and ALPHA_VANTAGE_AVAILABLE and av_key:
            logger.debug(f"Attempting Alpha Vantage for {symbol}...")
            try:
                av_ts_client = TimeSeries(key=av_key, output_format='pandas')
                av_fn = av_ts_client.get_daily_adjusted if interval=='1d' else av_ts_client.get_intraday
                av_int_map={'1h':'60min','15m':'15min'}; av_i=av_int_map.get(interval)
                if interval=='1d': df_av, _ = av_fn(symbol=symbol, outputsize='full')
                elif av_i: df_av, _ = av_fn(symbol=symbol, interval=av_i, outputsize='full')
                else: raise NotImplementedError(f"AV interval '{interval}' not mapped")
                if df_av is not None and not df_av.empty: data_fmt = _format_av_data(df_av);
                if data_fmt is not None: data = data_fmt.loc[start_dash:end_dash]; # Date filter
                if data is not None and not data.empty: logger.info(f"Alpha Vantage OK for {symbol}."); return data
            except Exception as e: logger.warning(f"Alpha Vantage failed for {symbol}: {e}")

    logger.warning(f"Primary sources failed for historical data of {symbol}.")
    return None



@st.cache_data(ttl=30, show_spinner="获取实时价格...")
def fetch_realtime_price_from_sources_cached(
        symbol: str,
        finnhub_key: Optional[str], polygon_key: Optional[str], av_key: Optional[str], ts_token: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Fetches realtime price from primary sources (AkShare(CN), Finnhub, Polygon, AV)."""
    logger.info(f"[Cache Call] Executing Primary Realtime Fetch: {symbol}")
    price_data = None

    # --- Priority 1: AkShare (CN) ---
    if _is_cn_stock_helper(symbol) and AKSHARE_AVAILABLE:
        logger.debug(f"Attempting AkShare realtime for {symbol} in cached func...")
        try:
            ak_symbol = symbol.split('.')[0]
            df_spot = ak.stock_zh_a_spot_em()
            stock_row = df_spot[df_spot['代码'] == ak_symbol]
            if not stock_row.empty:
                price = stock_row.iloc[0].get('最新价')
                if price is not None and isinstance(price, (int, float)) and price > 0:
                    price_data = {
                        'price': float(price),
                        'timestamp': time.time(),
                        'source': 'AkShare'
                    }
                    logger.info(f"AkShare realtime price: {price}")
                    return price_data
        except Exception as e:
            logger.warning(f"AkShare realtime fetch failed for {symbol}: {e}")

    # --- Priority 2: Finnhub ---
    if price_data is None and FINNHUB_AVAILABLE and finnhub_key:
        logger.debug(f"Attempting Finnhub realtime for {symbol} in cached func...")
        try:
            fh_client = FinnhubClient(api_key=finnhub_key)
            quote = fh_client.quote(symbol)
            if quote and isinstance(quote.get('c'), (int, float)) and quote['c'] > 0:
                price_data = {
                    'price': float(quote['c']),
                    'timestamp': float(quote.get('t', time.time())),
                    'source': 'Finnhub'
                }
                logger.info(f"Finnhub price: {price_data['price']}")
                return price_data
        except Exception as e:
            logger.warning(f"Finnhub realtime failed for {symbol}: {e}")

    # --- Priority 3: Polygon ---
    if price_data is None and POLYGON_AVAILABLE and polygon_key:
        logger.debug(f"Attempting Polygon realtime for {symbol} in cached func...")
        try:
            poly_client = PolygonClient(polygon_key)
            snapshot = poly_client.get_snapshot_ticker(symbol)
            if snapshot and snapshot.last_trade and snapshot.last_trade.p > 0:
                price_data = {
                    'price': float(snapshot.last_trade.p),
                    'timestamp': snapshot.last_trade.t / 1000.0,
                    'source': 'Polygon (Snapshot)'
                }
            elif snapshot and snapshot.prev_day and snapshot.prev_day.c > 0:
                price_data = {
                    'price': float(snapshot.prev_day.c),
                    'timestamp': time.time(),
                    'source': 'Polygon (PrevClose)'
                }
            if price_data:
                logger.info(f"Polygon price: {price_data['price']}")
                return price_data
        except Exception as e:
            logger.warning(f"Polygon realtime failed for {symbol}: {e}")

    # --- Priority 4: Alpha Vantage ---
    if price_data is None and ALPHA_VANTAGE_AVAILABLE and av_key:
        logger.debug(f"Attempting Alpha Vantage quote for {symbol} in cached func...")
        try:
            av_ts = TimeSeries(key=av_key, output_format='pandas')
            data_av, _ = av_ts.get_quote_endpoint(symbol=symbol)
            if data_av is not None and not data_av.empty and '05. price' in data_av.columns:
                price = float(data_av['05. price'].iloc[0])
                if price > 0:
                    price_data = {
                        'price': price,
                        'timestamp': time.time(),
                        'source': 'AlphaVantage'
                    }
            if price_data:
                logger.info(f"Alpha Vantage price: {price_data['price']}")
                return price_data
        except Exception as e:
            logger.warning(f"Alpha Vantage quote failed for {symbol}: {e}")

    logger.warning(f"Failed to fetch realtime price for {symbol} from primary sources.")
    return None


@st.cache_data(ttl=86400, show_spinner="搜索股票代码...")
def search_stocks_cached(
        query: str,
        finnhub_key: Optional[str],
        av_key: Optional[str],
        limit: int = 10
) -> List[Dict[str, str]]:
    """Searches stocks using primary sources (Finnhub, AV)."""
    if not query:
        return []
    logger.info(f"[Cache Call] Executing Primary Search: '{query}'")
    results = []
    seen_symbols = set()

    # Try Finnhub
    if FINNHUB_AVAILABLE and finnhub_key and len(results) < limit:
        try:
            fh_client = FinnhubClient(api_key=finnhub_key)
            search_res = fh_client.symbol_lookup(query)
            if search_res and search_res.get('result'):
                count = 0
                for item in search_res['result']:
                    symbol = item.get('symbol')
                    name = item.get('description')
                    exch = item.get('displaySymbol')
                    type_ = item.get('type')
                    if symbol and symbol not in seen_symbols and len(results) < limit:
                        results.append({
                            'symbol': symbol,
                            'name': name,
                            'exchange': exch,
                            'type': type_,
                            'source': 'Finnhub'
                        })
                        seen_symbols.add(symbol)
                        count += 1
                if count > 0:
                    logger.info(f"Finnhub search found {count} results.")
        except Exception as e:
            logger.warning(f"Finnhub search failed: {e}")

    # Try Alpha Vantage
    if ALPHA_VANTAGE_AVAILABLE and av_key and len(results) < limit:
        try:
            av_sc = Searching(key=av_key, output_format='pandas')
            data_av, _ = av_sc.get_symbol_search(keywords=query)
            if data_av is not None and not data_av.empty:
                current_count = len(results)
                for _, row in data_av.iterrows():
                    symbol = row.get('1. symbol')
                    name = row.get('2. name')
                    region = row.get('4. region')
                    type_ = row.get('3. type')
                    if symbol and symbol not in seen_symbols and len(results) < limit:
                        results.append({
                            'symbol': symbol,
                            'name': name,
                            'exchange': region,
                            'type': type_,
                            'source': 'AlphaVantage'
                        })
                        seen_symbols.add(symbol)
                added = len(results) - current_count
                if added > 0:
                    logger.info(f"AV search added {added} results.")
        except Exception as e:
            logger.warning(f"AV search failed: {e}")

    logger.info(f"Primary source search complete for '{query}'. Total: {len(results)}")
    return results[:limit]


class DataManager:
    def __init__(self, config=None):
        self.config = config or {};
        self.data_dir = Path(getattr(self.config, 'DATA_PATH', Path("data")));
        os.makedirs(self.data_dir, exist_ok=True);
        self.logger = logging.getLogger(__name__)
        self.ts_token = getattr(self.config, 'TUSHARE_TOKEN', None);
        self.av_key = getattr(self.config, 'ALPHA_VANTAGE_KEY', None)
        self.polygon_key = getattr(self.config, 'POLYGON_KEY', None);
        self.finnhub_key = getattr(self.config, 'FINNHUB_KEY', None)

        # Rate limiting state specific to this DataManager instance for yfinance
        self.yfinance_last_call_time = 0.0
        self.yfinance_min_interval = float(getattr(self.config, 'YFINANCE_MIN_INTERVAL', 2.5))
        self.yfinance_max_retries = int(getattr(self.config, 'YFINANCE_MAX_RETRIES', 2))  # Default to 1 retry for yf
        self.yfinance_base_delay = int(getattr(self.config, 'YFINANCE_BASE_DELAY', 3))  # Default to 3s base delay

        # Initialize API clients that are used directly by instance methods (e.g., get_stock_details)
        self.finnhub_client = FinnhubClient(
            api_key=self.finnhub_key) if FINNHUB_AVAILABLE and self.finnhub_key else None
        self.polygon_client = PolygonClient(self.polygon_key) if POLYGON_AVAILABLE and self.polygon_key else None
        self.alpha_vantage_fd = FundamentalData(key=self.av_key,
                                                output_format='pandas') if ALPHA_VANTAGE_AVAILABLE and self.av_key else None

        self.logger.info(
            "DataManager initialized. API Clients (direct use) status: Finnhub={}, Polygon={}, AV_FD={}".format(
                bool(self.finnhub_client), bool(self.polygon_client), bool(self.alpha_vantage_fd)
            ))
        self.logger.info(
            "Other API Keys available for cached functions: AkShare={}, Tushare={}, AV_TS/SC={}, Polygon={}, Finnhub={}".format(
                AKSHARE_AVAILABLE, bool(self.ts_token), bool(self.av_key and ALPHA_VANTAGE_AVAILABLE),
                bool(self.polygon_key and POLYGON_AVAILABLE), bool(self.finnhub_key and FINNHUB_AVAILABLE)
            ))

    def _is_cn_stock(self, symbol: str) -> bool:
        return _is_cn_stock_helper(symbol)

    def get_historical_data(self, symbol: str, days: int = 90, interval: str = "1d") -> Optional[pd.DataFrame]:
        self.logger.debug(f"Instance method get_historical_data called for {symbol}")
        end_date_obj = date.today()
        start_date_obj = end_date_obj - timedelta(days=int(days * 1.5) + 7)  # Fetch a bit more for margin

        data = fetch_historical_data_from_sources_cached(
            symbol, start_date_obj, end_date_obj, interval,
            AKSHARE_AVAILABLE, self.ts_token,
            self.polygon_key, self.finnhub_key, self.av_key
        )

        if data is None:  # Fallback to yfinance if primary sources failed
            self.logger.info(f"Primary sources failed for {symbol} historical data, attempting yfinance fallback...")
            yf_start_str = start_date_obj.strftime('%Y-%m-%d')
            # yfinance end date is exclusive, so add 1 day to include end_date_obj
            yf_end_str = (end_date_obj + timedelta(days=1)).strftime('%Y-%m-%d')

            for attempt in range(self.yfinance_max_retries):  # Loop for retries
                try:
                    # --- Rate Limiting for yfinance ---
                    now = time.time()
                    elapsed = now - self.yfinance_last_call_time
                    # Apply wait if not the very first call (attempt 0) OR if time elapsed is too short
                    if attempt > 0 or elapsed < self.yfinance_min_interval:
                        wait_time = self.yfinance_min_interval - elapsed if elapsed < self.yfinance_min_interval else 0
                        if wait_time > 0:
                            self.logger.warning(
                                f"DataManager: yfinance rate limiting. Waiting {wait_time:.2f}s for {symbol} (attempt {attempt + 1})")
                            time.sleep(wait_time)  # BLOCKING!
                    self.yfinance_last_call_time = time.time()  # Update timestamp AFTER potential sleep
                    # --- End Rate Limiting ---

                    data = fetch_yfinance_hist_data_cached(symbol, yf_start_str, yf_end_str, interval,
                                                           attempt_number_for_log=f"{attempt + 1}/{self.yfinance_max_retries}")
                    if data is not None:  # Success, even if empty DataFrame (means yf responded but no data for range)
                        break
                except yf.exceptions.YFRateLimitError:
                    self.logger.warning(
                        f"DataManager: yfinance rate limit caught for {symbol} (attempt {attempt + 1}).")
                    if attempt == self.yfinance_max_retries - 1:
                        self.logger.error(
                            f"DataManager: Max yfinance retries ({self.yfinance_max_retries}) reached for {symbol} due to rate limiting.")
                        break  # Exit loop, data will remain None
                    delay = self.yfinance_base_delay * (2 ** attempt) + random.uniform(0.1,
                                                                                       0.5)  # Shorter random jitter
                    self.logger.info(
                        f"DataManager: Waiting {delay:.2f}s before yfinance retry for {symbol} (rate limit)...")
                    time.sleep(delay)  # BLOCKING!
                except Exception as e_yf_call:  # Catch other errors from the cached yf function
                    self.logger.error(
                        f"DataManager: Unhandled error during yfinance call for {symbol} (attempt {attempt + 1}): {e_yf_call}",
                        exc_info=True)
                    data = None  # Ensure data is None on other errors
                    break  # Don't retry on unknown errors

            if data is None:
                self.logger.error(f"DataManager: All yfinance attempts failed for {symbol}.")

        # Final date range filtering
        if data is not None and not data.empty:
            try:
                if not isinstance(data.index, pd.DatetimeIndex):
                    data.index = pd.to_datetime(data.index, errors='coerce')
                    data = data.dropna(subset=[data.index.name])

                final_start_filter = (end_date_obj - timedelta(days=days)).strftime('%Y-%m-%d')
                final_end_filter = end_date_obj.strftime('%Y-%m-%d')
                data = data.sort_index().loc[final_start_filter:final_end_filter]  # Apply precise day filtering

                if data.empty:
                    self.logger.warning(
                        f"Data for {symbol} became empty after final date filtering ({final_start_filter} to {final_end_filter}). Consider fetching more initial days.")
                else:
                    self.logger.debug(f"Final filtering applied to {symbol} data, {len(data)} rows remain.")
            except Exception as e_filter:
                self.logger.error(f"Error during final date filtering for {symbol}: {e_filter}. Returning as is.",
                                  exc_info=True)
        elif data is not None and data.empty:
            self.logger.warning(f"Data for {symbol} is empty after all fetch attempts and formatting.")
        else:  # data is None
            self.logger.error(f"All data sources (including yfinance fallback) failed for {symbol}.")
        return data

    # --- Public Methods ---
    def get_realtime_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        # ... (完整的 get_realtime_price 方法，如我上次提供的，它调用 fetch_realtime_price_from_sources_cached)
        # ... (并包含 yfinance info 回退，也需要加入速率控制逻辑)
        self.logger.debug(f"Instance method get_realtime_price called for {symbol}")
        price_data = fetch_realtime_price_from_sources_cached(symbol, self.finnhub_key, self.polygon_key, self.av_key, self.ts_token)

        if price_data is None:
            self.logger.debug(f"Realtime price fetch failed for {symbol}, attempting latest close...")
            hist_data = self.get_historical_data(symbol, days=5)
            if hist_data is not None and not hist_data.empty and 'close' in hist_data.columns:
                try:
                    latest_close = float(hist_data['close'].iloc[-1])
                    if latest_close > 0:
                        price_data = {
                            'price': latest_close,
                            'timestamp': hist_data.index[-1].timestamp(),
                            'source': 'Historical Close'
                        }
                        self.logger.info(f"Using latest hist close for {symbol}: {latest_close}")
                except IndexError:
                    self.logger.warning(f"Not enough historical data for {symbol}")
                except Exception as e:
                    self.logger.warning(f"Could not extract latest close for {symbol}: {e}")
        if price_data is None:
            self.logger.error(f"最终无法获取 {symbol} 的实时价格。")
        return price_data

    def search_stocks(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        self.logger.debug(f"Instance method search_stocks called for query: '{query}'")
        results = search_stocks_cached(query, self.finnhub_key, self.av_key, limit)
        if not results:
            self.logger.info(f"API search yielded no results for '{query}', using local fallback.")
            results = self._search_stocks_local_fallback(query, limit)
        return results

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

    def get_stock_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取股票详细信息，尝试多个来源"""
        self.logger.info(f"Getting stock details for: {symbol}")
        details = {'symbol': symbol}

        # ===== 1. Finnhub数据源 =====
        if self.finnhub_client:
            try:
                profile = self.finnhub_client.profile2(symbol=symbol)
                if profile:
                    details.update({
                        'name': profile.get('name'),
                        'industry': profile.get('finnhubIndustry'),
                        'market_cap': profile.get('marketCapitalization'),
                        'shares_outstanding': profile.get('shareOutstanding'),
                        'web_url': profile.get('weburl'),
                        'ipo_date': profile.get('ipo'),
                        'country': profile.get('country'),
                        'exchange': profile.get('exchange'),
                        'currency': profile.get('currency'),
                        'logo': profile.get('logo')
                    })
                    # 财务指标
                    metrics = self.finnhub_client.company_basic_financials(symbol, 'all')
                    if metrics and metrics.get('metric'):
                        details.update({
                            'pe_ratio': metrics['metric'].get('peNormalizedAnnual'),
                            'eps': metrics['metric'].get('epsAnnual'),
                            'dividend_yield': metrics['metric'].get('dividendYieldIndicatedAnnual')
                        })
                    self.logger.debug(f"Finnhub details acquired for {symbol}")
            except Exception as e:
                self.logger.warning(f"Finnhub profile failed for {symbol}: {e}")

        # ===== 2. Polygon数据源 =====
        if self.polygon_client and not details.get('name'):
            try:
                poly_details = self.polygon_client.get_ticker_details(symbol)
                if poly_details:
                    details.update({
                        'name': poly_details.name,
                        'description': poly_details.description,
                        'sic_code': poly_details.sic_code,
                        'employees': poly_details.total_employees,
                        'list_date': poly_details.list_date,
                        'updated': poly_details.updated
                    })
                    # 补充市场数据
                    if poly_details.market == 'stocks':
                        details.update({
                            'market_cap': getattr(poly_details, 'market_cap', None),
                            'share_class_shares': getattr(poly_details, 'share_class_shares', None)
                        })
                    self.logger.debug(f"Polygon details acquired for {symbol}")
            except Exception as e:
                self.logger.warning(f"Polygon details failed for {symbol}: {e}")

        # ===== 3. Alpha Vantage数据源 =====
        if self.alpha_vantage_fd and (not details.get('industry') or not details.get('sector')):
            try:
                data_av, _ = self.alpha_vantage_fd.get_company_overview(symbol=symbol)
                if data_av is not None and not data_av.empty:
                    av_mappings = {
                        'name': 'Name',
                        'sector': 'Sector',
                        'industry': 'Industry',
                        'description': 'Description',
                        'address': 'Address',
                        'fiscal_year_end': 'FiscalYearEnd',
                        'latest_quarter': 'LatestQuarter',
                        'eps': 'EPS',
                        'pe_ratio': 'PERatio',
                        'dividend_yield': 'DividendYield',
                        'book_value': 'BookValue'
                    }
                    for key, av_key in av_mappings.items():
                        value = data_av.get(av_key, [None])[0]
                        if isinstance(value, str) and value.strip() == 'None':
                            value = None
                        details[key] = details.get(key) or value

                    # 数值类型转换
                    for num_key in ['market_cap', 'eps', 'pe_ratio', 'dividend_yield', 'book_value']:
                        if details.get(num_key) and isinstance(details[num_key], str):
                            if 'T' in details[num_key]:  # 处理万亿单位
                                details[num_key] = float(details[num_key].replace('T', '')) * 1e12
                            elif 'B' in details[num_key]:
                                details[num_key] = float(details[num_key].replace('B', '')) * 1e9
                            elif 'M' in details[num_key]:
                                details[num_key] = float(details[num_key].replace('M', '')) * 1e6
                            else:
                                details[num_key] = float(details[num_key])
                    self.logger.debug(f"Alpha Vantage details acquired for {symbol}")
            except Exception as e:
                self.logger.warning(f"AV overview failed for {symbol}: {e}")

        # ===== 4. yfinance数据源 =====
        if not details.get('name') or not details.get('market_cap'):
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                if info and info.get('symbol', '').upper() == symbol.upper():
                    details.setdefault('name', info.get('shortName', symbol))
                    details.update({
                        'sector': info.get('sector'),
                        'industry': info.get('industry'),
                        'market_cap': info.get('marketCap'),
                        'employees': info.get('fullTimeEmployees'),
                        'website': info.get('website'),
                        'city': info.get('city'),
                        'state': info.get('state'),
                        'country': info.get('country'),
                        'business_summary': info.get('longBusinessSummary')
                    })
                    # 财务指标
                    if not details.get('pe_ratio'):
                        details['pe_ratio'] = info.get('trailingPE')
                    if not details.get('dividend_yield'):
                        details['dividend_yield'] = info.get('dividendYield')
                    if not details.get('eps'):
                        details['eps'] = info.get('trailingEps')
                    self.logger.debug(f"yfinance details acquired for {symbol}")
            except Exception as e:
                self.logger.warning(f"yfinance info failed for {symbol}: {e}")

        # ===== 5. 合并实时价格数据 =====
        if details.get('name') and 'price' not in details:
            realtime_data = self.get_realtime_price(symbol)
            if realtime_data:
                details.update({
                    'price': realtime_data.get('price'),
                    'last_update': datetime.fromtimestamp(realtime_data['timestamp']).isoformat(),
                    'data_source': realtime_data['source']
                })

        # ===== 6. 数据验证和后处理 =====
        try:
            # 市值单位统一处理（转换为美元）
            market_cap = details.get('market_cap')
            if market_cap:
                if isinstance(market_cap, str):
                    if '万亿' in market_cap:  # 处理中文单位
                        details['market_cap'] = float(market_cap.replace('万亿', '')) * 1e12
                    elif '亿' in market_cap:
                        details['market_cap'] = float(market_cap.replace('亿', '')) * 1e8
                details['market_cap'] = float(details['market_cap'])

            # 日期格式化
            for date_field in ['ipo_date', 'list_date', 'latest_quarter']:
                if details.get(date_field):
                    if isinstance(details[date_field], str):
                        details[date_field] = datetime.strptime(details[date_field], '%Y-%m-%d').date().isoformat()
                    elif isinstance(details[date_field], (int, float)):
                        details[date_field] = datetime.fromtimestamp(details[date_field]).date().isoformat()
        except Exception as e:
            self.logger.error(f"Data post-processing failed: {e}")

        # ===== 最终验证 =====
        if not details.get('name') or details['name'] == symbol:
            self.logger.warning(f"Insufficient details for {symbol}")
            return None

        # 清理空值字段
        return {k: v for k, v in details.items() if v not in [None, '', 0.0, 0, 'None']}

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