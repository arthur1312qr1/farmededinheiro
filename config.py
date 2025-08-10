import os

class Config:
    def __init__(self):
        # Chaves da API
        self.API_KEY = os.getenv("API_KEY")
        self.API_SECRET = os.getenv("API_SECRET")
        self.API_PASSPHRASE = os.getenv("API_PASSPHRASE")

        # Mercado alvo - Futures USDT-M
        # Corrigido: productType correto para a API da Bitget
        self.PRODUCT_TYPE = "umcbl"  # antes: "USDT-FUTURES"
        
        # Corrigido: symbol correto para futures perpétuos USDT-M
        self.SYMBOL = os.getenv("SYMBOL", "ETHUSDT_UMCBL")  # antes: "ETHUSDT"

        # Configurações de negociação
        self.BASE_URL = "https://api.bitget.com"
        self.INTERVAL = os.getenv("INTERVAL", "1m")
        self.QUANTITY = float(os.getenv("QUANTITY", "0.01"))
        self.LEVERAGE = int(os.getenv("LEVERAGE", "50"))

        # Configurações do servidor
        self.HOST = "0.0.0.0"
        self.PORT = int(os.getenv("PORT", 5000))

        # IA
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
