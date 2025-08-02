# proxy_test.py
import os
import requests
import socket
import logging

# --- 配置 ---
# 确保这个地址和端口与您的 Clash 客户端完全一致
PROXY_URL = "http://127.0.0.1:7890" 
# Gemini API 的一个核心域名
TEST_URL = "https://generativelanguage.googleapis.com/.well-known/genai-supported-models?key=DUMMY_KEY"

# 设置环境变量 (模拟 .env.txt 的效果)
os.environ['HTTP_PROXY'] = PROXY_URL
os.environ['HTTPS_PROXY'] = PROXY_URL

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 猴子补丁：强制 IPv4 (与您项目中 manager.py 的补丁相同) ---
try:
    import requests.packages.urllib3.util.connection as urllib3_cn
    _allowed_gai_family_orig = urllib3_cn.allowed_gai_family
    def allowed_gai_family_ipv4_only():
        return socket.AF_INET
    urllib3_cn.allowed_gai_family = allowed_gai_family_ipv4_only
    logging.info("Network patch applied: Forcing IPv4 for requests/urllib3.")
except Exception as e:
    logging.error(f"Failed to apply IPv4 patch: {e}")

# --- 测试函数 ---
def test_proxy_connection():
    logging.info(f"--- Starting Proxy Connection Test ---")
    logging.info(f"Using Proxy: {os.environ.get('HTTPS_PROXY')}")
    logging.info(f"Target URL: {TEST_URL}")

    proxies = {
        "http": os.environ.get('HTTP_PROXY'),
        "https'": os.environ.get('HTTPS_PROXY'),
    }

    try:
        # 1. 测试到代理服务器本身的连通性
        logging.info("Step 1: Pinging the proxy server address (127.0.0.1)...")
        with socket.create_connection(("127.0.0.1", 7890), timeout=5) as sock:
            logging.info("SUCCESS: Proxy port 7890 is open and listening.")
    except Exception as e:
        logging.error(f"FAILURE: Cannot connect to the proxy server at 127.0.0.1:7890. Error: {e}")
        logging.error("Please ensure your Clash client is running and the port is correct.")
        return

    try:
        # 2. 通过代理发送请求
        logging.info(f"Step 2: Sending request to {TEST_URL} via proxy...")
        response = requests.get(TEST_URL, proxies=proxies, timeout=30, verify=True)
        
        logging.info(f"SUCCESS: Received a response. Status Code: {response.status_code}")
        
        # 检查响应内容
        if response.status_code == 200:
            logging.info("Response content seems OK (received data).")
            try:
                # 尝试解析 JSON
                data = response.json()
                logging.info("Successfully parsed JSON response.")
                # 打印一些模型名称
                if 'models' in data and data['models']:
                     logging.info(f"Found models, e.g., {data['models'][0].get('name')}")
                     logging.info("\n结论: 您的 Python 环境可以通过 Clash 代理成功访问 Google API！")
                else:
                     logging.warning("Response is JSON, but no 'models' key found.")
            except Exception as e_json:
                logging.error(f"Could not parse response as JSON. Error: {e_json}")
                logging.info(f"Raw response text: {response.text[:200]}")
        else:
            logging.error(f"Received an error status code: {response.status_code}")
            logging.error(f"Response text: {response.text}")
            logging.info("\n结论: 连接成功，但 API 返回了错误。这可能是 API Key 问题或权限问题。")

    except requests.exceptions.ProxyError as e:
        logging.error(f"FAILURE: A ProxyError occurred. This means the connection TO THE PROXY succeeded, but the proxy's connection TO THE TARGET failed.", exc_info=True)
        logging.info("\n结论: Clash 节点本身有问题。请在 Clash 客户端中切换到另一个节点，或者更新您的 Clash 订阅/配置文件。")
    except requests.exceptions.ConnectTimeout as e:
        logging.error(f"FAILURE: A ConnectTimeout occurred. Could not connect to the target URL within the timeout period.", exc_info=True)
        logging.info("\n结论: 可能是 Clash 节点网络质量差，或者目标服务器无响应。请切换节点。")
    except requests.exceptions.SSLError as e:
        logging.error(f"FAILURE: An SSLError occurred. This indicates a problem with SSL certificates.", exc_info=True)
        logging.info("\n结论: 可能是 Clash 的 MITM (中间人) 证书没有被您的系统信任。请检查 Clash 的证书设置。")
    except Exception as e:
        logging.error(f"FAILURE: An unexpected error occurred.", exc_info=True)
        logging.info("\n结论: 发生了未知错误，请检查上面的详细 Traceback。")

# --- 运行测试 ---
if __name__ == "__main__":
    test_proxy_connection()