# ai_modules/market_rag.py
"""
RAG для анализа новостей и sentiment
Усиливает P_j(S) компонент в XGBoostStrategy
"""

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import requests
from transformers import pipeline

class MarketIntelligenceRAG:
    """
    Анализирует новости и дополняет сигналы контекстом
    """
    
    def __init__(self):
        # Local embeddings (no API needed)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Vector store
        self.vectorstore = FAISS.load_local("./market_vectors", self.embeddings)
        
        # Sentiment analyzer
        self.sentiment = pipeline("sentiment-analysis", 
                                 model="ProsusAI/finbert")
        
    def analyze_market_context(self, symbol="BTC/USDT") -> Dict:
        """
        Анализирует текущий контекст рынка
        
        Returns:
            {
                'sentiment_score': -1 to 1,
                'fear_greed_index': 0-100,
                'key_events': [...],
                'risk_factors': [...],
                'recommendation': 'BULLISH/BEARISH/NEUTRAL'
            }
        """
        
        # 1. Fetch recent news
        news = self._fetch_crypto_news(symbol)
        
        # 2. Analyze sentiment
        sentiments = []
        for article in news[:10]:
            result = self.sentiment(article['title'])
            score = result[0]['score'] if result[0]['label'] == 'POSITIVE' else -result[0]['score']
            sentiments.append(score)
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        
        # 3. Get Fear & Greed Index
        fgi = self._get_fear_greed_index()
        
        # 4. Vector search for similar historical events
        query = f"Bitcoin price {symbol} volatility {avg_sentiment}"
        similar_events = self.vectorstore.similarity_search(query, k=3)
        
        # 5. Extract risk factors
        risk_factors = self._extract_risks(similar_events)
        
        # 6. Final recommendation
        if avg_sentiment > 0.3 and fgi > 50:
            recommendation = "BULLISH"
        elif avg_sentiment < -0.3 or fgi < 30:
            recommendation = "BEARISH"
        else:
            recommendation = "NEUTRAL"
        
        return {
            'sentiment_score': avg_sentiment,
            'fear_greed_index': fgi,
            'key_events': [e.page_content[:100] for e in similar_events],
            'risk_factors': risk_factors,
            'recommendation': recommendation
        }
    
    def _fetch_crypto_news(self, symbol):
        """Fetch последние новости"""
        # CryptoPanic API или другой источник
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token=YOUR_TOKEN&currencies={symbol.split('/')[0]}"
        
        try:
            response = requests.get(url)
            return response.json().get('results', [])
        except:
            return []
    
    def _get_fear_greed_index(self):
        """Get Fear & Greed Index"""
        try:
            response = requests.get("https://api.alternative.me/fng/")
            return int(response.json()['data'][0]['value'])
        except:
            return 50  # Neutral default