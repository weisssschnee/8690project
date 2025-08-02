# core/analysis/sentiment.py
import pandas as pd
import numpy as np
from textblob import TextBlob
import aiohttp
import feedparser
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import pytz
from newspaper import Article
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    def __init__(self, config):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.vader = SentimentIntensityAnalyzer()
        self.news_cache = {}
        self.cache_duration = timedelta(hours=1)

    async def analyze_market_sentiment(self, symbol: str) -> Dict[str, float]:
        """分析市场情绪"""
        try:
            # 获取多个来源的情绪数据
            news_sentiment = await self._analyze_news_sentiment(symbol)
            social_sentiment = await self._analyze_social_sentiment(symbol)
            technical_sentiment = await self._analyze_technical_sentiment(symbol)

            # 综合情绪分析
            combined_sentiment = self._combine_sentiment_scores(
                news_sentiment,
                social_sentiment,
                technical_sentiment
            )

            # 获取情绪状态描述
            sentiment_status = self._get_sentiment_status(combined_sentiment)

            return {
                'composite_score': combined_sentiment,
                'news_score': news_sentiment,
                'social_score': social_sentiment,
                'technical_score': technical_sentiment,
                'sentiment_status': sentiment_status,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'analysis_details': {
                    'news_count': len(self.news_cache.get(f"news_{symbol}", [])),
                    'social_count': len(self.news_cache.get(f"social_{symbol}", [])),
                    'indicators': self.news_cache.get(f"technical_{symbol}", {})
                }
            }

        except Exception as e:
            self.logger.error(f"Error analyzing market sentiment: {e}")
            return {
                'composite_score': 0.0,
                'news_score': 0.0,
                'social_score': 0.0,
                'technical_score': 0.0,
                'sentiment_status': '数据不足',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'analysis_details': {}
            }

    # 添加get_sentiment方法作为analyze_market_sentiment的别名
    async def get_sentiment(self, symbol: str) -> Dict[str, float]:
        """获取市场情绪（analyze_market_sentiment的别名）"""
        return await self.analyze_market_sentiment(symbol)

    def _get_sentiment_status(self, score: float) -> str:
        """根据情绪分数返回状态描述"""
        try:
            # 如果配置不存在，使用默认阈值
            default_thresholds = {
                'extreme_positive': 0.8,
                'positive': 0.3,
                'neutral': 0.0,
                'negative': -0.3,
                'extreme_negative': -0.8
            }

            thresholds = getattr(self.config, 'SENTIMENT_CONFIG', {}).get('thresholds', default_thresholds)

            if score >= thresholds['extreme_positive']:
                return '极度乐观'
            elif score >= thresholds['positive']:
                return '乐观'
            elif score >= thresholds['neutral']:
                return '中性偏多'
            elif score > thresholds['negative']:
                return '中性'
            elif score > thresholds['extreme_negative']:
                return '悲观'
            else:
                return '极度悲观'
        except Exception as e:
            self.logger.error(f"Error getting sentiment status: {e}")
            return '未知'

    async def _analyze_news_sentiment(self, symbol: str) -> float:
        """分析新闻情绪"""
        try:
            # 检查缓存
            if self._check_cache(f"news_{symbol}"):
                return self.news_cache[f"news_{symbol}"]['score']

            # 获取新闻数据
            news_articles = await self._fetch_news(symbol)
            if not news_articles:
                return 0.0

            # 分析每篇新闻的情绪
            sentiment_scores = []
            for article in news_articles:
                # 获取文章内容
                text = article.get('title', '') + ' ' + article.get('description', '')

                # 新闻来源权重
                source_weight = self.config.SENTIMENT_CONFIG['news']['source_weights'].get(
                    article.get('source_type', 'standard'), 0.7
                )

                # 计算时间衰减因子
                time_decay = self._calculate_time_decay(article.get('publishedAt'))

                # 情感分析
                score = self._analyze_text_sentiment(text)

                # 加权分数
                weighted_score = score * source_weight * time_decay
                sentiment_scores.append(weighted_score)

            # 计算加权平均分数
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0

            # 更新缓存
            self._update_cache(f"news_{symbol}", {
                'score': avg_sentiment,
                'articles': news_articles,
                'timestamp': datetime.now()
            })

            return avg_sentiment

        except Exception as e:
            self.logger.error(f"Error analyzing news sentiment: {e}")
            return 0.0

    async def _fetch_news(self, symbol: str) -> List[Dict]:
        """获取新闻数据"""
        try:
            # 首先尝试使用 NewsAPI
            try:
                articles = await self._fetch_from_newsapi(symbol)
                if articles:
                    return articles
            except Exception as e:
                self.logger.warning(f"NewsAPI failed: {e}")

            # 如果 NewsAPI 失败，使用备用 RSS 源
            try:
                articles = await self._fetch_from_rss(symbol)
                return articles
            except Exception as e:
                self.logger.warning(f"RSS feeds failed: {e}")

            return []

        except Exception as e:
            self.logger.error(f"Error fetching news: {e}")
            return []

    async def _fetch_from_newsapi(self, symbol: str) -> List[Dict]:
        """从 NewsAPI 获取新闻"""
        try:
            if not self.config.NEWS_API_KEY:
                raise ValueError("News API key not found")

            url = "https://newsapi.org/v2/everything"
            params = {
                'q': symbol,
                'apiKey': self.config.NEWS_API_KEY,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 10
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = data.get('articles', [])
                        for article in articles:
                            article['source_type'] = 'newsapi'
                        return articles
                    elif response.status == 401:
                        self.logger.error("News API authentication failed, switching to fallback source")
                        return []
                    else:
                        self.logger.error(f"News API request failed: {response.status}")
                        return []

        except Exception as e:
            self.logger.error(f"Error in NewsAPI request: {e}")
            return []

    async def _fetch_from_rss(self, symbol: str) -> List[Dict]:
        """从 RSS feeds 获取新闻"""
        try:
            articles = []
            feeds = self.config.NEWS_SOURCES['fallback']['feeds']

            for feed_url in feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:5]:  # 每个源取前5条
                        # 检查新闻是否包含目标股票代码
                        if symbol.lower() in entry.title.lower() or \
                                (hasattr(entry, 'description') and
                                 symbol.lower() in entry.description.lower()):

                            # 转换发布时间为ISO格式
                            try:
                                published = datetime.strptime(
                                    entry.published,
                                    '%a, %d %b %Y %H:%M:%S %z'
                                ).isoformat()
                            except:
                                published = datetime.now(pytz.UTC).isoformat()

                            articles.append({
                                'title': entry.title,
                                'description': entry.description if hasattr(entry, 'description') else '',
                                'url': entry.link if hasattr(entry, 'link') else '',
                                'publishedAt': published,
                                'source': {
                                    'name': feed.feed.title if hasattr(feed, 'feed') else 'RSS Feed'
                                },
                                'source_type': 'rss'
                            })

                except Exception as e:
                    self.logger.warning(f"Error parsing RSS feed {feed_url}: {e}")
                    continue

            return articles[:10]  # 最多返回10条新闻

        except Exception as e:
            self.logger.error(f"Error fetching RSS news: {e}")
            return []

    async def _analyze_social_sentiment(self, symbol: str) -> float:
        """分析社交媒体情绪"""
        try:
            # 获取社交媒体数据
            social_posts = await self._fetch_social_data(symbol)

            sentiment_scores = []
            for post in social_posts:
                # 基础情感分数
                score = self._analyze_text_sentiment(post['text'])

                # 计算用户影响力权重
                influence_weight = self._calculate_user_influence(post['user_metrics'])

                # 计算时间衰减
                time_decay = self._calculate_time_decay(post['created_at'])

                # 加权分数
                weighted_score = score * influence_weight * time_decay
                sentiment_scores.append(weighted_score)

            # 计算加权平均
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0

            # 缓存结果
            self._update_cache(f"social_{symbol}", {
                'score': avg_sentiment,
                'posts': social_posts,
                'timestamp': datetime.now()
            })

            return avg_sentiment

        except Exception as e:
            self.logger.error(f"Error analyzing social sentiment: {e}")
            return 0.0

    async def _fetch_social_data(self, symbol: str) -> List[Dict]:
        """获取社交媒体数据"""
        # TODO: 实现社交媒体数据获取
        # 这里可以集成Twitter、Reddit等API
        return []

    async def _analyze_technical_sentiment(self, symbol: str) -> float:
        """分析技术指标情绪"""
        try:
            # 获取技术指标数据
            technical_data = await self._fetch_technical_data(symbol)

            # 使用配置中定义的技术指标
            indicators = self.config.SENTIMENT_CONFIG['technical']['indicators']

            sentiment_scores = []
            for indicator in indicators:
                if indicator in technical_data:
                    score = self._normalize_indicator(
                        technical_data[indicator],
                        technical_data.get(f"{indicator}_min", -1),
                        technical_data.get(f"{indicator}_max", 1)
                    )
                    sentiment_scores.append(score)

            # 计算技术指标综合得分
            technical_score = np.mean(sentiment_scores) if sentiment_scores else 0.0

            # 缓存结果
            self._update_cache(f"technical_{symbol}", {
                'score': technical_score,
                'indicators': technical_data,
                'timestamp': datetime.now()
            })

            return technical_score

        except Exception as e:
            self.logger.error(f"Error analyzing technical sentiment: {e}")
            return 0.0

    async def _fetch_technical_data(self, symbol: str) -> Dict:
        """获取技术指标数据"""
        try:
            # 示例技术指标数据
            # 实际应用中应该从数据源获取
            return {
                'rsi': 50,
                'macd': 0,
                'volume_change': 0,
                'rsi_min': 0,
                'rsi_max': 100,
                'macd_min': -2,
                'macd_max': 2,
                'volume_change_min': -100,
                'volume_change_max': 100
            }
        except Exception as e:
            self.logger.error(f"Error fetching technical data: {e}")
            return {}

    def _analyze_text_sentiment(self, text: str) -> float:
        """分析文本情绪"""
        try:
            # 使用VADER进行情感分析
            sentiment_scores = self.vader.polarity_scores(text)
            # 返回复合情感分数
            return sentiment_scores['compound']
        except Exception as e:
            self.logger.error(f"Error analyzing text sentiment: {e}")
            return 0.0

    def _calculate_time_decay(self, published_time: str) -> float:
        """计算时间衰减因子"""
        try:
            if not published_time:
                return 0.5

            publish_dt = datetime.fromisoformat(published_time.replace('Z', '+00:00'))
            time_diff = datetime.now(publish_dt.tzinfo) - publish_dt
            hours_diff = time_diff.total_seconds() / 3600

            # 使用指数衰减
            decay_factor = np.exp(-hours_diff / 24)  # 24小时半衰期
            return max(0.1, decay_factor)  # 确保最小权重为0.1

        except Exception as e:
            self.logger.error(f"Error calculating time decay: {e}")
            return 0.5

    def _calculate_user_influence(self, user_metrics: Dict) -> float:
        """计算用户影响力权重"""
        try:
            influence_factors = self.config.SENTIMENT_CONFIG['social']['influence_factors']

            # 标准化各个指标
            followers_score = min(1.0, user_metrics.get('followers', 0) / 10000)
            engagement_score = min(1.0, user_metrics.get('engagement', 0) / 100)
            reputation_score = min(1.0, user_metrics.get('reputation', 0) / 100)

            # 计算加权分数
            influence_score = (
                    followers_score * influence_factors['followers'] +
                    engagement_score * influence_factors['engagement'] +
                    reputation_score * influence_factors['reputation']
            )

            return max(0.1, influence_score)  # 确保最小权重为0.1

        except Exception as e:
            self.logger.error(f"Error calculating user influence: {e}")
            return 0.5

    def _normalize_indicator(self, value: float, min_val: float, max_val: float) -> float:
        """标准化指标值到[-1, 1]范围"""
        try:
            normalized = 2 * (value - min_val) / (max_val - min_val) - 1
            return max(min(normalized, 1), -1)
        except Exception:
            return 0.0

    def _combine_sentiment_scores(self, news_score: float, social_score: float,
                                  technical_score: float) -> float:
        """综合多个情绪分数"""
        try:
            # 如果配置不存在，使用默认权重
            default_weights = {
                'news': 0.3,
                'social': 0.3,
                'technical': 0.4
            }

            weights = getattr(self.config, 'SENTIMENT_CONFIG', {}).get('weights', default_weights)

            combined_score = (
                    news_score * weights['news'] +
                    social_score * weights['social'] +
                    technical_score * weights['technical']
            )

            return max(min(combined_score, 1), -1)  # 确保结果在[-1, 1]范围内

        except Exception as e:
            self.logger.error(f"Error combining sentiment scores: {e}")
            return 0.0

    def _check_cache(self, key: str) -> bool:
        """检查缓存是否有效"""
        if key in self.news_cache:
            cache_time = self.news_cache[key]['timestamp']
            if datetime.now() - cache_time < self.cache_duration:
                return True
        return False

    def _update_cache(self, key: str, data: Dict) -> None:
        """更新缓存"""
        self.news_cache[key] = {
            **data,
            'timestamp': datetime.now() if 'timestamp' not in data else data['timestamp']
        }