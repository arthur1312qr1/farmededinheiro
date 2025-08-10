import os
import json
import logging
import requests
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class NewsAPIClient:
    """Cliente para integração com NewsAPI para notícias financeiras"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        
    def get_crypto_news(self, symbol: str = "ethereum", language: str = "pt") -> List[Dict]:
        """Busca notícias relacionadas a criptomoedas"""
        try:
            endpoint = f"{self.base_url}/everything"
            params = {
                'q': f'{symbol} OR cryptocurrency OR crypto OR bitcoin',
                'language': language,
                'sortBy': 'publishedAt',
                'pageSize': 10,
                'apiKey': self.api_key
            }
            
            response = requests.get(endpoint, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                articles = []
                
                for article in data.get('articles', []):
                    articles.append({
                        'titulo': article.get('title'),
                        'descricao': article.get('description'),
                        'url': article.get('url'),
                        'fonte': article.get('source', {}).get('name'),
                        'data_publicacao': article.get('publishedAt'),
                        'relevancia': self._calculate_relevance(article.get('title', ''))
                    })
                
                logger.info(f"Obtidas {len(articles)} notícias sobre {symbol}")
                return articles
            else:
                logger.error(f"Erro NewsAPI: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Erro ao buscar notícias: {e}")
            return []
    
    def _calculate_relevance(self, title: str) -> float:
        """Calcula relevância da notícia baseada no título"""
        keywords_high = ['ethereum', 'eth', 'pump', 'rally', 'surge', 'breakout']
        keywords_medium = ['crypto', 'blockchain', 'defi', 'trading']
        keywords_low = ['bitcoin', 'btc', 'market', 'price']
        
        title_lower = title.lower()
        score = 0.0
        
        for keyword in keywords_high:
            if keyword in title_lower:
                score += 1.0
                
        for keyword in keywords_medium:
            if keyword in title_lower:
                score += 0.6
                
        for keyword in keywords_low:
            if keyword in title_lower:
                score += 0.3
                
        return min(score, 1.0)

class EtherscanClient:
    """Cliente para integração com Etherscan API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.etherscan.io/api"
        
    def get_eth_price(self) -> Optional[float]:
        """Obtém preço atual do ETH"""
        try:
            params = {
                'module': 'stats',
                'action': 'ethprice',
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == '1':
                    price = float(data.get('result', {}).get('ethusd', 0))
                    logger.info(f"Preço ETH obtido via Etherscan: ${price}")
                    return price
            
            return None
            
        except Exception as e:
            logger.error(f"Erro ao obter preço ETH via Etherscan: {e}")
            return None
    
    def get_gas_tracker(self) -> Dict[str, Any]:
        """Obtém informações sobre gas fees"""
        try:
            params = {
                'module': 'gastracker',
                'action': 'gasoracle',
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == '1':
                    result = data.get('result', {})
                    return {
                        'gas_seguro': int(result.get('SafeGasPrice', 0)),
                        'gas_padrao': int(result.get('StandardGasPrice', 0)),
                        'gas_rapido': int(result.get('FastGasPrice', 0)),
                        'ultima_atualizacao': datetime.now().isoformat()
                    }
            
            return {}
            
        except Exception as e:
            logger.error(f"Erro ao obter gas tracker: {e}")
            return {}

class CoinGeckoClient:
    """Cliente para integração com CoinGecko API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.coingecko.com/api/v3"
        self.headers = {}
        
        if api_key:
            self.headers['x-cg-demo-api-key'] = api_key
    
    def get_ethereum_data(self) -> Dict[str, Any]:
        """Obtém dados completos do Ethereum"""
        try:
            endpoint = f"{self.base_url}/simple/price"
            params = {
                'ids': 'ethereum',
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true',
                'include_market_cap': 'true'
            }
            
            response = requests.get(endpoint, params=params, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                eth_data = data.get('ethereum', {})
                
                result = {
                    'preco_atual': float(eth_data.get('usd', 0)),
                    'mudanca_24h': float(eth_data.get('usd_24h_change', 0)),
                    'volume_24h': float(eth_data.get('usd_24h_vol', 0)),
                    'market_cap': float(eth_data.get('usd_market_cap', 0)),
                    'timestamp': datetime.now().isoformat(),
                    'fonte': 'coingecko'
                }
                
                logger.info(f"Dados ETH obtidos via CoinGecko: ${result['preco_atual']}")
                return result
            else:
                logger.error(f"Erro CoinGecko: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Erro ao obter dados CoinGecko: {e}")
            return {}
    
    def get_market_sentiment(self) -> Dict[str, Any]:
        """Obtém dados de sentimento do mercado"""
        try:
            endpoint = f"{self.base_url}/global"
            
            response = requests.get(endpoint, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json().get('data', {})
                
                return {
                    'dominancia_btc': float(data.get('market_cap_percentage', {}).get('btc', 0)),
                    'dominancia_eth': float(data.get('market_cap_percentage', {}).get('eth', 0)),
                    'total_market_cap': float(data.get('total_market_cap', {}).get('usd', 0)),
                    'volume_24h': float(data.get('total_volume', {}).get('usd', 0)),
                    'mercados_ativos': int(data.get('active_cryptocurrencies', 0)),
                    'indice_medo_ganancia': self._get_fear_greed_index(),
                    'timestamp': datetime.now().isoformat()
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Erro ao obter sentimento do mercado: {e}")
            return {}
    
    def _get_fear_greed_index(self) -> int:
        """Obtém índice de medo e ganância (simulado)"""
        try:
            # API alternativa para Fear & Greed Index
            response = requests.get("https://api.alternative.me/fng/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return int(data.get('data', [{}])[0].get('value', 50))
        except:
            pass
        
        # Valor padrão se não conseguir obter
        return 50

class MarketDataAggregator:
    """Agregador de dados de mercado de múltiplas fontes"""
    
    def __init__(self, config):
        self.config = config
        self.news_client = NewsAPIClient(config.NEWSAPI_KEY) if config.NEWSAPI_KEY else None
        self.etherscan_client = EtherscanClient(config.ETHERSCAN_API_KEY) if config.ETHERSCAN_API_KEY else None
        self.coingecko_client = CoinGeckoClient(config.COINGECKO_API_KEY)
        
    def get_complete_market_analysis(self) -> Dict[str, Any]:
        """Obtém análise completa do mercado de todas as fontes"""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'dados_preco': {},
            'noticias': [],
            'sentimento': {},
            'gas_fees': {},
            'resumo': {}
        }
        
        # Dados de preço do CoinGecko
        if self.coingecko_client:
            analysis['dados_preco'] = self.coingecko_client.get_ethereum_data()
            analysis['sentimento'] = self.coingecko_client.get_market_sentiment()
        
        # Preço do Etherscan como backup
        if self.etherscan_client and not analysis['dados_preco']:
            eth_price = self.etherscan_client.get_eth_price()
            if eth_price:
                analysis['dados_preco'] = {
                    'preco_atual': eth_price,
                    'fonte': 'etherscan',
                    'timestamp': datetime.now().isoformat()
                }
            
            analysis['gas_fees'] = self.etherscan_client.get_gas_tracker()
        
        # Notícias relevantes
        if self.news_client:
            analysis['noticias'] = self.news_client.get_crypto_news('ethereum')
        
        # Gerar resumo
        analysis['resumo'] = self._generate_summary(analysis)
        
        logger.info("Análise completa do mercado gerada")
        return analysis
    
    def _generate_summary(self, analysis: Dict) -> Dict[str, Any]:
        """Gera resumo da análise de mercado"""
        resumo = {
            'preco_disponivel': bool(analysis['dados_preco']),
            'noticias_count': len(analysis['noticias']),
            'sentimento_disponivel': bool(analysis['sentimento']),
            'gas_disponivel': bool(analysis['gas_fees'])
        }
        
        # Análise de tendência básica
        dados_preco = analysis.get('dados_preco', {})
        if dados_preco:
            mudanca_24h = dados_preco.get('mudanca_24h', 0)
            
            if mudanca_24h > 5:
                resumo['tendencia'] = 'ALTA_FORTE'
            elif mudanca_24h > 2:
                resumo['tendencia'] = 'ALTA'
            elif mudanca_24h > -2:
                resumo['tendencia'] = 'LATERAL'
            elif mudanca_24h > -5:
                resumo['tendencia'] = 'BAIXA'
            else:
                resumo['tendencia'] = 'BAIXA_FORTE'
        else:
            resumo['tendencia'] = 'DESCONHECIDA'
        
        return resumo
