"""
Gemini AI Error Handler for Trading Bot
Handles AI-powered error analysis and trading decisions
"""

import logging
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)

class GeminiErrorHandler:
    """Enhanced Gemini AI handler for error analysis and trading decisions"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini AI handler
        
        Args:
            api_key: Gemini API key (optional)
        """
        self.api_key = api_key
        self.model = None
        
        # For now, we'll use basic analysis only since Gemini library is not properly installed
        logger.info("Gemini AI using basic analysis mode (library not available)")
    
    def analyze_and_fix_error(self, error_message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Analyze error and provide fix suggestions
        
        Args:
            error_message: Error message to analyze
            context: Additional context about the error
            
        Returns:
            Analysis and fix suggestions
        """
        return self._basic_error_analysis(error_message, context)
    
    def _basic_error_analysis(self, error_message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Basic error analysis without AI"""
        
        error_lower = error_message.lower()
        
        # API connection errors
        if any(keyword in error_lower for keyword in ['connection', 'timeout', 'network', 'requests']):
            return "Network error detected. Check internet connection and API status. Will retry automatically."
        
        # Authentication errors
        if any(keyword in error_lower for keyword in ['authentication', 'unauthorized', 'invalid key', 'api key']):
            return "STOP_TRADING - Authentication error. Check Bitget API keys and permissions."
        
        # Balance/funds errors
        if any(keyword in error_lower for keyword in ['insufficient', 'balance', 'funds', 'margin']):
            return "Insufficient funds error. Check account balance before continuing trades."
        
        # Rate limiting
        if any(keyword in error_lower for keyword in ['rate limit', 'too many requests', '429']):
            return "Rate limit exceeded. Increasing delay between requests automatically."
        
        # JSON/parsing errors
        if any(keyword in error_lower for keyword in ['json', 'parse', 'decode', 'invalid response']):
            return "Data parsing error. API response format may have changed. Will retry."
        
        # Bitget specific errors
        if any(keyword in error_lower for keyword in ['bitget', 'api error', 'code']):
            return "Bitget API error detected. Check API status and credentials."
        
        # Position errors
        if any(keyword in error_lower for keyword in ['position', 'leverage', 'margin mode']):
            return "Position management error. Check leverage settings and margin requirements."
        
        # Critical system errors
        if any(keyword in error_lower for keyword in ['memory', 'disk', 'system', 'worker']):
            return "RESTART_REQUIRED - System resource error detected."
        
        return f"Unknown error type. Continuing with caution. Error: {error_message[:100]}..."
    
    def analyze_market_conditions(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze market conditions and provide trading signals
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Trading analysis and recommendations
        """
        return self._basic_market_analysis(market_data)
    
    def _basic_market_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Basic market analysis without AI"""
        try:
            price_change = float(market_data.get('change_percent_24h', 0))
            volume = float(market_data.get('volume_24h', 0))
            current_price = float(market_data.get('last_price', 0))
            
            # Simple momentum-based analysis
            if price_change > 5 and volume > 1000000:
                return {
                    "signal": "buy",
                    "confidence": "medium",
                    "reasoning": f"Strong upward momentum (+{price_change:.1f}%) with good volume",
                    "risk_level": "medium",
                    "suggested_leverage": 5,
                    "entry_price": current_price,
                    "stop_loss": current_price * 0.95,
                    "take_profit": current_price * 1.08
                }
            elif price_change < -5 and volume > 1000000:
                return {
                    "signal": "sell",
                    "confidence": "medium", 
                    "reasoning": f"Strong bearish momentum ({price_change:.1f}%) with volume",
                    "risk_level": "medium",
                    "suggested_leverage": 3,
                    "entry_price": current_price,
                    "stop_loss": current_price * 1.05,
                    "take_profit": current_price * 0.92
                }
            elif abs(price_change) > 2:
                signal = "buy" if price_change > 0 else "sell"
                return {
                    "signal": signal,
                    "confidence": "low",
                    "reasoning": f"Moderate price movement ({price_change:.1f}%)",
                    "risk_level": "low",
                    "suggested_leverage": 2,
                    "entry_price": current_price
                }
            else:
                return {
                    "signal": "hold",
                    "confidence": "low",
                    "reasoning": f"Low volatility ({price_change:.1f}%) - market consolidating",
                    "risk_level": "low",
                    "suggested_leverage": 1,
                    "current_price": current_price
                }
                
        except Exception as e:
            logger.error(f"Basic market analysis failed: {e}")
            return {
                "signal": "hold",
                "confidence": "low", 
                "reasoning": "Analysis error - staying neutral for safety",
                "risk_level": "high",
                "suggested_leverage": 1,
                "error": str(e)
            }
    
    def analyze_trading_opportunity(self, market_data: Dict[str, Any], balance_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze trading opportunity based on market data and balance
        
        Args:
            market_data: Current market data
            balance_info: Account balance information
            
        Returns:
            Trading opportunity analysis
        """
        try:
            available_balance = float(balance_info.get('available_balance', 0))
            min_trade_amount = 10  # Minimum $10 USDT to trade
            
            if available_balance < min_trade_amount:
                return {
                    "trade_recommended": False,
                    "reason": f"Insufficient balance: ${available_balance:.2f} < ${min_trade_amount} minimum",
                    "action": "wait_for_funds"
                }
            
            market_analysis = self._basic_market_analysis(market_data)
            
            if market_analysis['signal'] == 'hold' or market_analysis['confidence'] == 'low':
                return {
                    "trade_recommended": False,
                    "reason": "Market conditions unclear - waiting for better signal",
                    "market_analysis": market_analysis,
                    "action": "monitor"
                }
            
            # Calculate position size based on available balance and risk
            risk_percentage = 0.02  # Risk 2% of balance per trade
            max_trade_amount = available_balance * 0.3  # Use max 30% of balance
            suggested_amount = min(available_balance * risk_percentage * market_analysis['suggested_leverage'], 
                                 max_trade_amount)
            
            return {
                "trade_recommended": True,
                "signal": market_analysis['signal'],
                "confidence": market_analysis['confidence'],
                "suggested_amount": round(suggested_amount, 2),
                "suggested_leverage": market_analysis['suggested_leverage'],
                "reasoning": market_analysis['reasoning'],
                "risk_level": market_analysis['risk_level'],
                "market_analysis": market_analysis,
                "balance_analysis": {
                    "available": available_balance,
                    "risk_amount": round(available_balance * risk_percentage, 2),
                    "position_size_percentage": round((suggested_amount / available_balance) * 100, 1)
                }
            }
            
        except Exception as e:
            logger.error(f"Trading opportunity analysis failed: {e}")
            return {
                "trade_recommended": False,
                "reason": f"Analysis error: {str(e)}",
                "action": "error_recovery",
                "error": str(e)
            }
    
    def validate_trade_conditions(self, signal: Dict[str, Any], market_data: Dict[str, Any], 
                                 balance_info: Dict[str, Any]) -> Dict[str, bool]:
        """
        Validate if conditions are met for executing a trade
        
        Returns:
            Dictionary with validation results
        """
        validations = {
            "sufficient_balance": False,
            "market_conditions_ok": False,
            "api_available": False,
            "risk_acceptable": False,
            "all_checks_passed": False
        }
        
        try:
            # Check balance
            available_balance = float(balance_info.get('available_balance', 0))
            suggested_amount = float(signal.get('suggested_amount', 0))
            
            validations["sufficient_balance"] = available_balance >= suggested_amount
            
            # Check market conditions
            confidence = signal.get('confidence', 'low')
            validations["market_conditions_ok"] = confidence in ['medium', 'high']
            
            # Check API availability
            validations["api_available"] = not balance_info.get('error') and not balance_info.get('api_error')
            
            # Check risk level
            risk_level = signal.get('risk_level', 'high')
            validations["risk_acceptable"] = risk_level in ['low', 'medium']
            
            # All checks passed
            validations["all_checks_passed"] = all(validations.values())
            
        except Exception as e:
            logger.error(f"Trade validation failed: {e}")
            validations["validation_error"] = str(e)
        
        return validations
    
    def is_available(self) -> bool:
        """Check if Gemini AI is available (always False for basic mode)"""
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get handler status"""
        return {
            "available": False,
            "model": "basic_analysis",
            "api_configured": bool(self.api_key and self.api_key.strip()),
            "mode": "basic",
            "features": [
                "basic_error_analysis",
                "market_trend_analysis", 
                "trading_opportunity_analysis",
                "trade_validation"
            ]
        }
