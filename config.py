import os
import logging

logger = logging.getLogger(__name__)

class TradingConfig:
    """Configuração do bot de trading ETH/USDT com alavancagem fixa 10x"""
    
    def __init__(self):
        # Configurações do ambiente
        self.PORT = int(os.getenv("PORT", 5000))
        self.ENVIRONMENT = os.getenv("RAILWAY_ENVIRONMENT", "development")
        
        # TRADING REAL - SEM SIMULAÇÃO
        self.PAPER_TRADING = os.getenv("PAPER_TRADING", "false").lower() == "true"
        self.SYMBOL = "ETHUSDT_UMCBL"
        
        # ALAVANCAGEM FIXA EM 10X
        self.FIXED_LEVERAGE = 10
        self.MIN_LEVERAGE = 10
        self.MAX_LEVERAGE = 10
        
        # Configurações de trading para REAL
        self.MIN_MARGIN_USAGE_PERCENT = float(os.getenv("MIN_MARGIN_USAGE_PERCENT", "75.0"))
        self.POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "5.0"))  # Mais lento para real
        self.DRAWDOWN_CLOSE_PCT = float(os.getenv("DRAWDOWN_CLOSE_PCT", "0.025"))  # 2.5%
        self.LIQ_DIST_THRESHOLD = float(os.getenv("LIQ_DIST_THRESHOLD", "0.04"))  # 4%
        self.MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3"))  # Mais conservador
        
        # API Keys do Bitget
        self.BITGET_API_KEY = os.getenv("BITGET_API_KEY", "")
        self.BITGET_API_SECRET = os.getenv("BITGET_API_SECRET", "")
        self.BITGET_PASSPHRASE = os.getenv("BITGET_PASSPHRASE", "")
        
        # Configurações mais conservadoras para REAL
        self.MIN_BALANCE_USDT = float(os.getenv("MIN_BALANCE_USDT", "50.0"))  # Mínimo $50
        self.MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
        self.RETRY_DELAY = float(os.getenv("RETRY_DELAY", "3.0"))
        
        # URLs da API
        self.BITGET_BASE_URL = "https://api.bitget.com"
        
        # Configurações de risco para TRADING REAL
        self.POSITION_SIZE_PERCENT = float(os.getenv("POSITION_SIZE_PERCENT", "0.08"))  # 8% mais conservador
        self.STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "0.02"))  # 2%
        self.TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "0.05"))  # 5%
        self.MAX_DAILY_TRADES = int(os.getenv("MAX_DAILY_TRADES", "15"))  # Máximo 15 trades/dia
        self.DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "100.0"))  # $100 máximo perda
        
        self._log_configuration()
    
    def _log_configuration(self):
        """Log das configurações principais"""
        mode = "Trading REAL" if not self.PAPER_TRADING else "Paper Trading"
        logger.info(f"🔧 CONFIGURAÇÃO DO BOT CARREGADA:")
        logger.info(f"📊 Modo: {mode}")
        logger.info(f"💱 Símbolo: {self.SYMBOL}")
        logger.info(f"⚡ Alavancagem FIXA: {self.FIXED_LEVERAGE}x")
        logger.info(f"⏱️ Intervalo de polling: {self.POLL_INTERVAL}s")
        logger.info(f"💰 Saldo mínimo: ${self.MIN_BALANCE_USDT}")
        logger.info(f"📈 Tamanho da posição: {self.POSITION_SIZE_PERCENT*100}%")
        logger.info(f"🛡️ Stop Loss: {self.STOP_LOSS_PERCENT*100}%")
        logger.info(f"🎯 Take Profit: {self.TAKE_PROFIT_PERCENT*100}%")
        logger.info(f"🌍 Ambiente: {self.ENVIRONMENT}")
    
    def validate_configuration(self) -> bool:
        """Validar configuração completa"""
        try:
            logger.info("🔍 Validando configuração...")
            
            # Validar API keys se não estiver em paper trading
            if not self.PAPER_TRADING:
                if not self.validate_api_keys():
                    logger.error("❌ Chaves da API não validadas")
                    return False
            
            # Validar configurações numéricas
            if self.POLL_INTERVAL < 3.0:
                logger.error("❌ POLL_INTERVAL deve ser pelo menos 3.0 segundos para trading real")
                return False
            
            if self.POSITION_SIZE_PERCENT > 0.15:
                logger.warning("⚠️ Tamanho de posição alto para trading real, ajustando para 15%")
                self.POSITION_SIZE_PERCENT = 0.15
            
            # Validar alavancagem fixa
            if self.FIXED_LEVERAGE != 10:
                logger.warning(f"⚠️ Alavancagem deve ser fixa em 10x")
                self.FIXED_LEVERAGE = 10
                self.MIN_LEVERAGE = 10
                self.MAX_LEVERAGE = 10
            
            logger.info("✅ Configuração validada com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao validar configuração: {e}")
            return False
    
    def validate_api_keys(self) -> bool:
        """Validar se as chaves da API estão presentes"""
        if self.PAPER_TRADING:
            logger.info("📝 Paper trading ativo - chaves da API não necessárias")
            return True
        
        required_keys = [
            self.BITGET_API_KEY,
            self.BITGET_API_SECRET, 
            self.BITGET_PASSPHRASE
        ]
        
        missing_keys = [key for key in required_keys if not key or key.strip() == ""]
        
        if missing_keys:
            logger.error(f"❌ Faltam {len(missing_keys)} chaves da API Bitget")
            logger.error("💡 Configure as variáveis: BITGET_API_KEY, BITGET_API_SECRET, BITGET_PASSPHRASE")
            return False
        
        logger.info("✅ Todas as chaves da API estão presentes")
        return True
    
    def get_display_config(self):
        """Obter configuração para exibição"""
        return {
            'environment': self.ENVIRONMENT,
            'paper_trading': self.PAPER_TRADING,
            'real_trading': not self.PAPER_TRADING,
            'symbol': self.SYMBOL,
            'fixed_leverage': self.FIXED_LEVERAGE,
            'poll_interval': self.POLL_INTERVAL,
            'min_balance': self.MIN_BALANCE_USDT,
            'position_size_percent': self.POSITION_SIZE_PERCENT * 100,
            'stop_loss_percent': self.STOP_LOSS_PERCENT * 100,
            'take_profit_percent': self.TAKE_PROFIT_PERCENT * 100,
            'max_daily_trades': self.MAX_DAILY_TRADES,
            'daily_loss_limit': self.DAILY_LOSS_LIMIT,
            'api_keys_configured': self.validate_api_keys(),
            'max_consecutive_losses': self.MAX_CONSECUTIVE_LOSSES
        }
