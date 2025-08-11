#!/usr/bin/env python3
"""
TRADING BOT BITGET - RAILWAY DEPLOYMENT
Versão corrigida e completa para Railway
"""
import os
import sys
import logging
import signal
import threading
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.middleware.proxy_fix import ProxyFix
# Configure logging para Railway
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
# ========================================
# CONFIGURAÇÃO PARA RAILWAY
# ========================================
class RailwayConfig:
    """Configuração otimizada para Railway"""
    
    def __init__(self):
        # Configurações Railway
        self.PORT = int(os.getenv("PORT", 5000))
        self.RAILWAY_ENVIRONMENT = os.getenv("RAILWAY_ENVIRONMENT", "production")
        
        # Chaves API (definir no Railway)
        self.BITGET_API_KEY = os.getenv("BITGET_API_KEY", "")
        self.BITGET_API_SECRET = os.getenv("BITGET_API_SECRET", "")
        self.BITGET_PASSPHRASE = os.getenv("BITGET_PASSPHRASE", "")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
        
        # Configuração Trading
        self.PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"
        self.SYMBOL = os.getenv("SYMBOL", "ETHUSDT")
        self.POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "3.0"))
        self.MIN_BALANCE_USDT = float(os.getenv("MIN_BALANCE_USDT", "20.0"))
        self.MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
        
        # URLs da API
        self.BITGET_BASE_URL = "https://api.bitget.com"
        
        logger.info(f"Configuração Railway - Paper Trading: {self.PAPER_TRADING}")
    
    def validate_api_keys(self) -> bool:
        """Valida chaves API"""
        if self.PAPER_TRADING:
            logger.info("Modo paper trading - chaves API não necessárias")
            return True
        
        keys = [self.BITGET_API_KEY, self.BITGET_API_SECRET, self.BITGET_PASSPHRASE]
        missing = [k for k in keys if not k or k.strip() == ""]
        
        if missing:
            logger.error(f"Faltam {len(missing)} chaves da API Bitget")
            return False
        
        logger.info("Chaves API validadas")
        return True
# ========================================
# CLASSE GEMINI HANDLER SIMPLIFICADA
# ========================================
class SimpleGeminiHandler:
    """Handler Gemini simplificado para Railway"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        logger.info("Handler Gemini inicializado (modo básico)")
    
    def analyze_and_fix_error(self, error_message: str, context=None) -> str:
        """Análise básica de erros"""
        error_lower = error_message.lower()
        
        if any(word in error_lower for word in ['connection', 'timeout', 'network']):
            return "Erro de conexão detectado. Verificar internet e status da API."
        
        if any(word in error_lower for word in ['authentication', 'unauthorized', 'api key']):
            return "STOP_TRADING - Erro de autenticação. Verificar chaves API."
        
        if any(word in error_lower for word in ['insufficient', 'balance', 'funds']):
            return "Saldo insuficiente. Verificar saldo da conta."
        
        if any(word in error_lower for word in ['rate limit', 'too many']):
            return "Limite de requisições atingido. Aguardando..."
        
        return f"Erro desconhecido: {error_message[:100]}..."
    
    def analyze_market_conditions(self, market_data=None):
        """Análise básica de mercado"""
        return {
            "signal": "hold",
            "confidence": "low",
            "reasoning": "Análise básica ativa",
            "risk_level": "medium"
        }
# ========================================
# CLASSE BITGET API SIMPLIFICADA
# ========================================
import hashlib
import hmac
import base64
import json
import requests
from urllib.parse import urlencode
class SimpleBitgetAPI:
    """API Bitget simplificada para Railway"""
    
    def __init__(self, api_key, api_secret, passphrase):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.base_url = "https://api.bitget.com"
        self.session = requests.Session()
        logger.info("API Bitget inicializada")
    
    def _generate_signature(self, timestamp, method, path, body=''):
        """Gera assinatura HMAC"""
        message = timestamp + method + path + body
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode()
    
    def _get_headers(self, timestamp, method, path, body=''):
        """Cabeçalhos da requisição"""
        signature = self._generate_signature(timestamp, method, path, body)
        
        return {
            'ACCESS-KEY': self.api_key,
            'ACCESS-SIGN': signature,
            'ACCESS-TIMESTAMP': timestamp,
            'ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
    
    def get_balance(self):
        """Obter saldo USDT"""
        try:
            timestamp = str(int(time.time() * 1000))
            method = 'GET'
            path = '/api/v2/mix/account/accounts'
            params = {'productType': 'USDT-FUTURES'}
            query_string = urlencode(params)
            full_path = f"{path}?{query_string}"
            
            headers = self._get_headers(timestamp, method, full_path)
            
            response = self.session.get(
                f"{self.base_url}{full_path}",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000' and data.get('data'):
                    balance_info = data['data'][0]
                    available = float(balance_info.get('available', 0))
                    
                    return {
                        'available_balance': available,
                        'total_equity': float(balance_info.get('equity', 0)),
                        'unrealized_pnl': float(balance_info.get('unrealizedPL', 0)),
                        'currency': 'USDT',
                        'source': 'bitget_api',
                        'last_updated': datetime.now().isoformat(),
                        'success': True
                    }
            
            return {'error': f'API Error: {response.status_code}', 'success': False}
            
        except Exception as e:
            logger.error(f"Erro ao obter saldo: {e}")
            return {'error': str(e), 'success': False}
# ========================================
# CLASSE TRADING BOT SIMPLIFICADA
# ========================================
class SimpleTradingBot:
    """Bot de trading simplificado para Railway"""
    
    def __init__(self, config, gemini_handler):
        self.config = config
        self.gemini_handler = gemini_handler
        self.bitget_api = None
        self.running = False
        self.emergency_stop = False
        self.consecutive_losses = 0
        self.last_balance_check = None
        self.cached_balance = None
        
        # Inicializar API se não for paper trading
        if not config.PAPER_TRADING and config.validate_api_keys():
            try:
                self.bitget_api = SimpleBitgetAPI(
                    config.BITGET_API_KEY,
                    config.BITGET_API_SECRET,
                    config.BITGET_PASSPHRASE
                )
                logger.info("API Bitget conectada")
            except Exception as e:
                logger.error(f"Erro ao conectar API Bitget: {e}")
        
        # Estado do bot
        self.bot_state = {
            'balance': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0,
            'consecutive_losses': 0,
            'emergency_stop': False,
            'last_updated': datetime.now().isoformat(),
            'current_positions': {},
            'trading_history': []
        }
        
        logger.info(f"Trading bot inicializado - {config.SYMBOL}")
    
    def get_balance_info(self):
        """Obter informações de saldo"""
        try:
            # Paper trading
            if self.config.PAPER_TRADING or not self.bitget_api:
                return {
                    'available_balance': 1000.0,  # Saldo simulado
                    'total_equity': 1000.0,
                    'unrealized_pnl': 0,
                    'currency': 'USDT',
                    'source': 'paper_trading',
                    'last_updated': datetime.now().isoformat(),
                    'paper_trading': True,
                    'sufficient_balance': True
                }
            
            # Trading real com cache
            now = time.time()
            if (self.cached_balance and self.last_balance_check and 
                now - self.last_balance_check < 30):  # Cache por 30s
                return self.cached_balance
            
            # Buscar saldo real
            balance_info = self.bitget_api.get_balance()
            
            if balance_info.get('success'):
                balance_info['paper_trading'] = False
                balance_info['sufficient_balance'] = balance_info.get('available_balance', 0) >= self.config.MIN_BALANCE_USDT
                
                # Cache do resultado
                self.cached_balance = balance_info
                self.last_balance_check = now
                
                return balance_info
            else:
                return {
                    'error': balance_info.get('error', 'Erro desconhecido'),
                    'available_balance': 0,
                    'currency': 'USDT',
                    'last_updated': datetime.now().isoformat(),
                    'paper_trading': False,
                    'sufficient_balance': False
                }
                
        except Exception as e:
            logger.error(f"Erro ao obter saldo: {e}")
            return {
                'error': str(e),
                'available_balance': 0,
                'currency': 'USDT',
                'last_updated': datetime.now().isoformat()
            }
    
    def get_state(self):
        """Obter estado do bot"""
        return self.bot_state.copy()
    
    def start(self):
        """Iniciar bot"""
        self.running = True
        logger.info("Bot iniciado")
    
    def stop(self):
        """Parar bot"""
        self.running = False
        logger.info("Bot parado")
    
    def execute_trading_cycle(self):
        """Ciclo básico de trading"""
        if not self.running or self.emergency_stop:
            return
        
        try:
            # Simular ciclo de trading
            balance_info = self.get_balance_info()
            
            if balance_info.get('error'):
                logger.warning(f"Erro no saldo: {balance_info['error']}")
                return
            
            # Atualizar estado
            self.bot_state['balance'] = balance_info.get('available_balance', 0)
            self.bot_state['last_updated'] = datetime.now().isoformat()
            
            # Análise de mercado básica
            market_analysis = self.gemini_handler.analyze_market_conditions()
            
            # Log do status
            if balance_info.get('paper_trading'):
                logger.debug("Paper trading - ciclo executado")
            else:
                logger.info(f"Saldo: ${balance_info.get('available_balance', 0):.2f} USDT")
            
     ...
[truncated]
