# test_network.py
import os
import requests
import urllib3

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def test_network_configs():
    """测试不同的网络配置"""

    print("🔍 检查环境变量...")
    print(f"HTTP_PROXY: {os.environ.get('HTTP_PROXY', 'Not set')}")
    print(f"HTTPS_PROXY: {os.environ.get('HTTPS_PROXY', 'Not set')}")
    print()

    # 测试URL
    test_url = "https://generativelanguage.googleapis.com"

    configs = [
        {"name": "直连", "proxies": None, "verify": True},
        {"name": "直连(无SSL)", "proxies": None, "verify": False},
        {"name": "代理", "proxies": {"https": os.environ.get('HTTPS_PROXY')}, "verify": True},
        {"name": "代理(无SSL)", "proxies": {"https": os.environ.get('HTTPS_PROXY')}, "verify": False},
    ]

    for config in configs:
        if config["proxies"] and not config["proxies"]["https"]:
            continue

        try:
            print(f"🧪 测试 {config['name']}...")
            response = requests.get(
                test_url,
                proxies=config["proxies"],
                verify=config["verify"],
                timeout=10
            )
            print(f"✅ {config['name']} 成功 (状态码: {response.status_code})")
        except Exception as e:
            print(f"❌ {config['name']} 失败: {e}")
        print()


if __name__ == "__main__":
    test_network_configs()