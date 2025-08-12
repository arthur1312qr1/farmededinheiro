import os
import logging

logger = logging.getLogger(__name__)

class TradingConfig:
    """Configuração do bot de trading otimizada para Railway"""
    
    def __init__(self):
        # Configurações do Railway
        self.PORT = int(os.getenv("PORT", 5000))
        self.ENVIRONMENT = os.getenv("RAILWAY_ENVIRONMENT", "production")
        
        # Configurações de Trading (conforme solicitado pelo usuário)
        self.PAPER_TRADING = os.getenv("PAPER_TRADING", "false").lower() == "true"  # false = trading real
        self.SYMBOL = os.getenv("SYMBOL", "ethusdt_UMCBL")  # ETH futures
        self.MIN_LEVERAGE = int(os.getenv("MIN_LEVERAGE", "9"))
        self.MAX_LEVERAGE = int(os.getenv("MAX_LEVERAGE", "60"))
        self.MIN_MARGIN_USAGE_PERCENT = float(os.getenv("MIN_MARGIN_USAGE_PERCENT", "80.0"))
        self.POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "1.0"))
        self.DRAWDOWN_CLOSE_PCT = float(os.getenv("DRAWDOWN_CLOSE_PCT", "0.03"))
        self.LIQ_DIST_THRESHOLD = float(os.getenv("LIQ_DIST_THRESHOLD", "0.03"))
        self.MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "5"))
        
        # API Keys - ficam no Railway
        self.BITGET_API_KEY = os.getenv("BITGET_API_KEY", "")
        self.BITGET_API_SECRET = os.getenv("BITGET_API_SECRET", "")
        self.BITGET_PASSPHRASE = os.getenv("BITGET_PASSPHRASE", "")
        
        # Configurações adicionais
        self.MIN_BALANCE_USDT = float(os.getenv("MIN_BALANCE_USDT", "50.0"))
        self.MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
        self.RETRY_DELAY = float(os.getenv("RETRY_DELAY", "5.0"))
        
        # URLs da API
        self.BITGET_BASE_URL = "https://api.bitget.com"
        
        logger.info(f"Configuração carregada:")
        logger.info(f"- Trading Real: {not self.PAPER_TRADING}")
        logger.info(f"- Symbol: {self.SYMBOL}")
        logger.info(f"- Leverage: {self.MIN_LEVERAGE}-{self.MAX_LEVERAGE}")
        logger.info(f"- Poll Interval: {self.POLL_INTERVAL}s")
        logger.info(f"- Environment: {self.ENVIRONMENT}")
    
    def validate_configuration(self) -> bool:
        """Validar configuração completa"""
        try:
            # Validar API keys se não estiver em paper trading
            if not self.PAPER_TRADING:
                if not self.validate_api_keys():
                    return False
            
            # Validar configurações numéricas
            if self.MIN_LEVERAGE >= self.MAX_LEVERAGE:
                logger.error("MIN_LEVERAGE deve ser menor que MAX_LEVERAGE")
                return False
            
            if self.POLL_INTERVAL < 0.5:
                logger.error("POLL_INTERVAL deve ser pelo menos 0.5 segundos")
                return False
            
            if self.MIN_MARGIN_USAGE_PERCENT < 10 or self.MIN_MARGIN_USAGE_PERCENT > 95:
                logger.error("MIN_MARGIN_USAGE_PERCENT deve estar entre 10 e 95")
                return False
            
            logger.info("Configuração validada com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao validar configuração: {e}")
            return False
    
    def validate_api_keys(self) -> bool:
        """Validar se as chaves da API estão presentes"""
        if self.PAPER_TRADING:
            logger.info("Paper trading ativo - chaves da API não necessárias")
            return True
        
        required_keys = [
            self.BITGET_API_KEY,
            self.BITGET_API_SECRET,
            self.BITGET_PASSPHRASE
        ]
        
        missing_keys = [key for key in required_keys if not key or key.strip() == ""]
        
        if missing_keys:
            logger.error(f"Faltam {len(missing_keys)} chaves da API Bitget")
            return False
        
        logger.info("Todas as chaves da API estão presentes")
        return True
    
    def get_display_config(self):
        """Obter configuração para exibição (sem dados sensíveis)"""
        return {
            'environment': self.ENVIRONMENT,
            'paper_trading': self.PAPER_TRADING,  # Mostra se está em paper ou real
            'real_trading': not self.PAPER_TRADING,  # Para mostrar se é trading real
            'symbol': self.SYMBOL,
            'poll_interval': self.POLL_INTERVAL,
            'min_leverage': self.MIN_LEVERAGE,
            'max_leverage': self.MAX_LEVERAGE,
            'min_margin_usage': self.MIN_MARGIN_USAGE_PERCENT,
            'min_balance': self.MIN_BALANCE_USDT,
            'api_keys_configured': self.validate_api_keys(),
            'max_consecutive_losses': self.MAX_CONSECUTIVE_LOSSES,
            'drawdown_close_pct': self.DRAWDOWN_CLOSE_PCT,
            'liq_dist_threshold': self.LIQ_DIST_THRESHOLD
        }
