import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class GeminiHandler:
    """Simplified Gemini AI handler for trading analysis"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY', '')
        self.client = None
        
        # Try to initialize Gemini client if API key is available
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel('gemini-pro')
                logger.info("Gemini AI client initialized successfully")
            except ImportError:
                logger.warning("google-generativeai library not available - using basic analysis")
                self.client = None
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini AI: {e}")
                self.client = None
        else:
            logger.info("Gemini API key not provided - using basic analysis")
    
    def analyze_and_fix_error(self, error_message: str, context=None) -> str:
        """Analyze errors and provide fixing suggestions"""
        try:
            error_lower = error_message.lower()
            
            # Basic error pattern matching
            if any(word in error_lower for word in ['connection', 'timeout', 'network']):
                return "Connection error detected. Check internet connectivity and API status."
            
            if any(word in error_lower for word in ['authentication', 'unauthorized', 'api key']):
                return "STOP_TRADING - Authentication error. Check API keys configuration."
            
            if any(word in error_lower for word in ['insufficient', 'balance', 'funds']):
                return "Insufficient balance detected. Check account balance."
            
            if any(word in error_lower for word in ['rate limit', 'too many']):
                return "Rate limit exceeded. Implementing backoff strategy."
            
            if any(word in error_lower for word in ['invalid', 'symbol']):
                return "Invalid trading symbol. Check symbol configuration."
            
            # If Gemini is available, use AI analysis
            if self.client:
                try:
                    prompt = f"""
                    Analyze this trading bot error and provide a concise solution:
                    
                    Error: {error_message}
                    Context: {context or 'No additional context'}
                    
                    Provide a brief analysis and solution. If the error is critical and trading should stop, 
                    start your response with "STOP_TRADING -"
                    """
                    
                    response = self.client.generate_content(prompt)
                    return response.text.strip()
                    
                except Exception as e:
                    logger.error(f"Gemini analysis failed: {e}")
            
            # Fallback response
            return f"Unknown error detected: {error_message[:100]}... Please check logs for details."
            
        except Exception as e:
            logger.error(f"Error in error analysis: {e}")
            return "Error analysis failed. Manual intervention required."
    
    def analyze_market_conditions(self, market_data=None):
        """Analyze market conditions and provide trading signals"""
        try:
            # Basic market analysis without AI
            basic_analysis = {
                "signal": "hold",
                "confidence": "low",
                "reasoning": "Basic analysis - insufficient data for confident trading",
                "risk_level": "medium",
                "timestamp": datetime.now().isoformat()
            }
            
            # If Gemini is available and market data is provided, use AI analysis
            if self.client and market_data:
                try:
                    prompt = f"""
                    Analyze the current market conditions for cryptocurrency trading:
                    
                    Market Data: {market_data}
                    
                    Provide a trading recommendation with:
                    1. Signal (buy/sell/hold)
                    2. Confidence level (high/medium/low)
                    3. Brief reasoning
                    4. Risk level (high/medium/low)
                    
                    Format as JSON-like structure.
                    """
                    
                    response = self.client.generate_content(prompt)
                    # For simplicity, return the text response
                    return {
                        "signal": "hold",  # Conservative default
                        "confidence": "medium",
                        "reasoning": response.text.strip()[:200],
                        "risk_level": "medium",
                        "timestamp": datetime.now().isoformat(),
                        "ai_analysis": True
                    }
                    
                except Exception as e:
                    logger.error(f"Gemini market analysis failed: {e}")
            
            return basic_analysis
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return {
                "signal": "hold",
                "confidence": "low",
                "reasoning": f"Analysis error: {str(e)}",
                "risk_level": "high",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_trading_advice(self, balance_info, market_conditions):
        """Get specific trading advice based on current conditions"""
        try:
            if not self.client:
                return "Gemini AI not available. Using conservative strategy."
            
            prompt = f"""
            Provide trading advice for a cryptocurrency bot:
            
            Current Balance: ${balance_info.get('available_balance', 0):.2f} USDT
            Market Conditions: {market_conditions}
            
            Provide specific, actionable advice in 2-3 sentences.
            """
            
            response = self.client.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error getting trading advice: {e}")
            return "Unable to generate trading advice. Maintain current positions."
