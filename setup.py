from setuptools import setup, find_packages

setup(
    name="trading_system",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'streamlit>=1.10.0',
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'scikit-learn>=0.24.2',
        'tensorflow>=2.6.0',
        'python-dotenv>=0.19.0',
        'requests>=2.26.0',
        'yfinance>=0.1.63',
        'tushare>=1.2.62',
        'textblob>=0.15.3',
        'vaderSentiment>=3.3.2',
        'newspaper3k>=0.2.8',
        'python-binance>=1.0.15',
        'sqlalchemy>=1.4.23',
        'joblib>=1.0.1',
    ],
)