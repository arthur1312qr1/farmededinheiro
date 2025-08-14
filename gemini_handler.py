import json
import logging
import os
from google import genai
from google.genai import types
from pydantic import BaseModel
from config import Config

logger = logging.getLogger(__name__)

class MarketAnalysis(BaseModel):
    trend: str
    strength: float
    recommendation: str
    confidence: float
    key_levels: list

class GeminiHandler:
    def __init__(self):
        api_key = Config.GEMINI_API_KEY
        if not api_key:
            logger.warning("⚠️ Gemini API key not configured")
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=api_key)
                logger.info("✅ Gemini AI initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Gemini: {e}")
                self.client = None

    def analyze_market(self, klines_data):
        """Analyze market data using Gemini AI"""
        if not self.client:
            return {
                'trend': 'unknown',
                'strength': 0.5,
                'recommendation': 'hold',
                'confidence': 0.0,
                'analysis': 'AI analysis unavailable',
                'key_levels': []
            }

        try:
            # Prepare market data for analysis
            recent_prices = [kline['close'] for kline in klines_data[-20:]]
            current_price = recent_prices[-1]
            
            # Calculate basic metrics
            price_change = ((current_price - recent_prices[0]) / recent_prices[0]) * 100
            volatility = self._calculate_volatility(recent_prices)
            
            # Create analysis prompt
            prompt = f"""
            Analise os seguintes dados de mercado para ETH/USDT:
            
            Preços recentes (últimos 20 períodos): {recent_prices}
            Preço atual: ${current_price:.2f}
            Variação: {price_change:.2f}%
            Volatilidade: {volatility:.2f}%
            
            Por favor, forneça uma análise técnica e recomendação de trading em JSON com:
            - trend: "bullish", "bearish", ou "sideways"
            - strength: força da tendência (0.0-1.0)
            - recommendation: "buy", "sell", ou "hold"
            - confidence: confiança na análise (0.0-1.0)
            - analysis: análise detalhada em português
            - key_levels: níveis de suporte/resistência importantes
            """

            response = self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    types.Content(role="user", parts=[types.Part(text=prompt)])
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    max_output_tokens=1000
                )
            )

            if response and response.text:
                try:
                    analysis = json.loads(response.text)
                    logger.info(f"📊 AI Analysis: {analysis.get('trend', 'unknown')} - {analysis.get('recommendation', 'hold')}")
                    return analysis
                except json.JSONDecodeError:
                    logger.error("Failed to parse AI response as JSON")
                    return self._fallback_analysis(recent_prices)
            else:
                return self._fallback_analysis(recent_prices)

        except Exception as e:
            logger.error(f"Error in Gemini analysis: {e}")
            return self._fallback_analysis(recent_prices)

    def _calculate_volatility(self, prices):
        """Calculate price volatility"""
        if len(prices) < 2:
            return 0.0
        
        returns = []
        for i in range(1, len(prices)):
            returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        if not returns:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        volatility = (variance ** 0.5) * 100  # Convert to percentage
        
        return volatility

    def _fallback_analysis(self, prices):
        """Fallback analysis when AI is unavailable"""
        if not prices or len(prices) < 2:
            return {
                'trend': 'unknown',
                'strength': 0.5,
                'recommendation': 'hold',
                'confidence': 0.0,
                'analysis': 'Dados insuficientes para análise',
                'key_levels': []
            }
        
        # Simple trend analysis
        price_change = (prices[-1] - prices[0]) / prices[0]
        
        if price_change > 0.02:  # 2% up
            trend = 'bullish'
            recommendation = 'buy'
        elif price_change < -0.02:  # 2% down
            trend = 'bearish'
            recommendation = 'sell'
        else:
            trend = 'sideways'
            recommendation = 'hold'
        
        strength = min(abs(price_change) * 10, 1.0)  # Scale to 0-1
        
        return {
            'trend': trend,
            'strength': strength,
            'recommendation': recommendation,
            'confidence': 0.6,  # Moderate confidence for fallback
            'analysis': f'Análise básica: tendência {trend}, variação de {price_change*100:.2f}%',
            'key_levels': [round(min(prices), 2), round(max(prices), 2)]
        }

    def get_trading_sentiment(self, market_news=""):
        """Get overall market sentiment"""
        if not self.client:
            return {
                'sentiment': 'neutral',
                'score': 0.5,
                'summary': 'Sentiment analysis unavailable'
            }

        try:
            prompt = f"""
            Analise o sentimento do mercado de criptomoedas considerando:
            - Notícias: {market_news}
            - Contexto atual do mercado crypto
            - Fatores macroeconômicos relevantes
            
            Responda em JSON com:
            - sentiment: "positive", "negative", ou "neutral"
            - score: pontuação do sentimento (0.0 = muito negativo, 1.0 = muito positivo)
            - summary: resumo em português
            """

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2
                )
            )

            if response and response.text:
                return json.loads(response.text)
            else:
                return self._fallback_sentiment()

        except Exception as e:
            logger.error(f"Error getting sentiment: {e}")
            return self._fallback_sentiment()

    def _fallback_sentiment(self):
        """Fallback sentiment when AI is unavailable"""
        return {
            'sentiment': 'neutral',
            'score': 0.5,
            'summary': 'Sentimento neutro - análise AI indisponível'
        }
