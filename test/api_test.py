import os
from dotenv import load_dotenv

# 加载 .env.txt 文件中的环境变量
# 确保您的 .env.txt 文件和这个脚本在同一个目录下，或者提供正确路径
dotenv_path = '.env.txt'
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print(f"Loaded environment variables from: {dotenv_path}")
else:
    print(f"Warning: {dotenv_path} not found. Make sure it's in the same directory.")


def test_finnhub():
    print("\n--- Testing Finnhub ---")
    api_key = os.getenv("FINNHUB_KEY")
    if not api_key:
        print("Finnhub API key not found in .env.txt. Skipping.")
        return

    try:
        import finnhub
        finnhub_client = finnhub.Client(api_key=api_key)
        quote = finnhub_client.quote('AAPL')
        if quote and quote.get('c') and quote.get('c') > 0:
            print(f"SUCCESS: Finnhub returned price for AAPL: {quote.get('c')}")
        else:
            print(f"FAILED: Finnhub call succeeded but returned no valid data. Response: {quote}")
    except Exception as e:
        print(f"FAILED: Could not connect to Finnhub. Error: {e}")


def test_polygon():
    print("\n--- Testing Polygon.io ---")
    api_key = os.getenv("POLYGON_KEY")
    if not api_key:
        print("Polygon API key not found in .env.txt. Skipping.")
        return

    try:
        from polygon import RESTClient
        client = RESTClient(api_key)
        aggs = client.get_aggs(ticker="AAPL", multiplier=1, timespan="day", from_="2023-01-01", to="2023-01-05")
        if aggs:
            print(f"SUCCESS: Polygon returned {len(aggs)} daily aggregates for AAPL.")
        else:
            print(f"FAILED: Polygon call succeeded but returned no data.")
    except Exception as e:
        print(f"FAILED: Could not connect to Polygon. Error: {e}")


def test_yfinance():
    print("\n--- Testing yfinance ---")
    try:
        import yfinance as yf
        aapl = yf.Ticker("AAPL")
        info = aapl.info
        if info and info.get('currentPrice'):
            print(f"SUCCESS: yfinance returned current price for AAPL: {info['currentPrice']}")
        else:
            # 尝试获取历史数据作为后备测试
            hist = aapl.history(period="1d")
            if not hist.empty:
                print(f"SUCCESS: yfinance returned historical data for AAPL. Last close: {hist['Close'].iloc[-1]}")
            else:
                print("FAILED: yfinance call succeeded but returned no data.")
    except Exception as e:
        print(f"FAILED: Could not connect to yfinance servers. Error: {e}")


if __name__ == "__main__":
    print("Running API connectivity tests...")
    test_finnhub()
    test_polygon()
    test_yfinance()