"""
Gemini AI Integration for Market Analysis
Uses Google's Gemini AI for cryptocurrency market sentiment and trend analysis
"""

import logging
import requests
import json
from typing import Dict, Optional, Any
import time

logger = logging.getLogger(__name__)

class GeminiAI:
    def __init__(self, api_key: str):
        """Initialize Gemini AI client"""
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
        
        logger.info("🧠 Gemini AI inicializada")
    
    def analyze_market(self, symbol: str, market_data: Dict) -> Dict:
        """Analyze market conditions using Gemini AI"""
        try:
            if not self.api_key:
                # Return mock analysis for demo
                import random
                sentiments = ['bullish', 'bearish', 'neutral']
                return {
                    'sentiment': random.choice(sentiments),
                    'confidence': random.uniform(0.6, 0.9),
                    'trend': 'bullish',
                    'key_factors': [
                        'Volume acima da média confirmando breakout',
                        'RSI oversold com divergência positiva',
                        'Suporte forte em $3,800'
                    ],
                    'recommendation': 'Consider long position with tight stop loss',
                    'risk_level': 'medium',
                    'timestamp': int(time.time())
                }
            
            # Prepare market analysis prompt
            prompt = self._create_analysis_prompt(symbol, market_data)
            
            # Make request to Gemini AI
            response = self._make_gemini_request(prompt)
            
            if response:
                return self._parse_analysis_response(response)
            else:
                return self._get_fallback_analysis()
                
        except Exception as e:
            logger.error(f"❌ Erro na análise Gemini AI: {e}")
            return self._get_fallback_analysis()
    
    def _create_analysis_prompt(self, symbol: str, market_data: Dict) -> str:
        """Create analysis prompt for Gemini AI"""
        price = market_data.get('price', 0)
        volume = market_data.get('volume', 0)
        change_24h = market_data.get('change_24h', 0)
        high_24h = market_data.get('high_24h', 0)
        low_24h = market_data.get('low_24h', 0)
        
        prompt = f"""
        Analyze the current market conditions for {symbol} cryptocurrency trading:

        Current Market Data:
        - Price: ${price:.2f}
        - 24h Change: {change_24h:+.2f}%
        - 24h High: ${high_24h:.2f}
        - 24h Low: ${low_24h:.2f}
        - Volume: {volume:,.0f}

        Please provide a JSON response with the following structure:
        {{
            "sentiment": "bullish|bearish|neutral",
            "confidence": 0.0-1.0,
            "trend": "bullish|bearish|sideways",
            "key_factors": ["factor1", "factor2", "factor3"],
            "recommendation": "trading recommendation",
            "risk_level": "low|medium|high"
        }}

        Focus on:
        1. Short-term scalping opportunities (1-5 minute timeframes)
        2. Volume analysis and momentum
        3. Support and resistance levels
        4. Risk assessment for high-frequency trading
        5. Current market sentiment

        Provide only the JSON response, no additional text.
        """
        
        return prompt.strip()
    
    def _make_gemini_request(self, prompt: str) -> Optional[str]:
        """Make request to Gemini AI API"""
        try:
            url = f"{self.base_url}/models/gemini-pro:generateContent"
            params = {'key': self.api_key}
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 1024,
                    "topP": 0.8
                }
            }
            
            response = self.session.post(
                url,
                params=params,
                json=data,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                candidates = result.get('candidates', [])
                if candidates:
                    content = candidates[0].get('content', {})
                    parts = content.get('parts', [])
                    if parts:
                        return parts[0].get('text', '')
            
            logger.error(f"❌ Erro na resposta Gemini AI: {response.status_code} - {response.text}")
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erro de rede Gemini AI: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Erro inesperado Gemini AI: {e}")
            return None
    
    def _parse_analysis_response(self, response_text: str) -> Dict:
        """Parse Gemini AI response into structured data"""
        try:
            # Try to extract JSON from response
            response_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            # Parse JSON
            analysis = json.loads(response_text)
            
            # Validate required fields
            required_fields = ['sentiment', 'confidence', 'trend', 'key_factors', 'recommendation', 'risk_level']
            for field in required_fields:
                if field not in analysis:
                    logger.warning(f"⚠️ Campo obrigatório ausente na resposta AI: {field}")
                    return self._get_fallback_analysis()
            
            # Add timestamp
            analysis['timestamp'] = int(time.time())
            
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Erro ao parsear resposta JSON: {e}")
            logger.error(f"Resposta recebida: {response_text}")
            return self._get_fallback_analysis()
        except Exception as e:
            logger.error(f"❌ Erro inesperado ao parsear resposta: {e}")
            return self._get_fallback_analysis()
    
    def _get_fallback_analysis(self) -> Dict:
        """Get fallback analysis when AI is unavailable"""
        return {
            'sentiment': 'neutral',
            'confidence': 0.5,
            'trend': 'sideways',
            'key_factors': [
                'Análise AI temporariamente indisponível',
                'Usando análise técnica básica',
                'Recomenda-se cautela extra'
            ],
            'recommendation': 'Wait for clearer signals',
            'risk_level': 'high',
            'timestamp': int(time.time())
        }
    
    def get_market_summary(self, symbols: list) -> Dict:
        """Get market summary for multiple symbols"""
        try:
            if not self.api_key:
                return {
                    'overall_sentiment': 'neutral',
                    'market_trend': 'sideways',
                    'volatility': 'medium',
                    'recommendations': [
                        'Monitor key support levels',
                        'Look for volume confirmation',
                        'Use tight stop losses'
                    ]
                }
            
            prompt = f"""
            Provide a brief market summary for cryptocurrency trading focusing on these symbols: {', '.join(symbols)}.
            
            Please analyze:
            1. Overall market sentiment
            2. Current trend direction
            3. Volatility levels
            4. Key trading recommendations
            
            Respond in JSON format:
            {{
                "overall_sentiment": "bullish|bearish|neutral",
                "market_trend": "bullish|bearish|sideways",
                "volatility": "low|medium|high",
                "recommendations": ["rec1", "rec2", "rec3"]
            }}
            """
            
            response = self._make_gemini_request(prompt)
            
            if response:
                try:
                    return json.loads(response.strip())
                except json.JSONDecodeError:
                    pass
            
            return {
                'overall_sentiment': 'neutral',
                'market_trend': 'sideways',
                'volatility': 'medium',
                'recommendations': ['Use conservative position sizing', 'Monitor market closely']
            }
            
        except Exception as e:
            logger.error(f"❌ Erro no resumo de mercado: {e}")
            return {
                'overall_sentiment': 'neutral',
                'market_trend': 'sideways',
                'volatility': 'high',
                'recommendations': ['Exercise extreme caution']
            }
