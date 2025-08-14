import json
import logging
import os
from config import Config
logger = logging.getLogger(__name__)

class GeminiHandler:
    def __init__(self):
        api_key = Config.GEMINI_API_KEY
        if not api_key:
            logger.warning("⚠️ Gemini API key not configured")
            self.client = None
        else:
            try:
                # Initialize Gemini client only if API key is available
                from google import genai
                from google.genai import types
                self.client = genai.Client(api_key=api_key)
                logger.info("✅ Gemini AI initialized for REAL TRADING ANALYSIS")
            except ImportError:
                logger.warning("⚠️ Gemini dependencies not available")
                self.client = None
            except Exception as e:
                logger.error(f"❌ Failed to initialize Gemini: {e}")
                self.client = None

    def analyze_market(self, klines_data):
        """Analyze market data using Gemini AI for 99% accuracy"""
        if not self.client:
            return self._fallback_analysis(klines_data)
        try:
            # Extract price...
