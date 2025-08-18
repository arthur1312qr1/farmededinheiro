import  os
import logging
import json
from datetime import datetime
from  flask import Flask, render_template, request, jsonify, send_from_directory, make_response 
from flask_cors import CORS
import threading
import time

# Importar o bot de trading e API da Bitget
try:
    from trading_bot import TradingBot
    from bitget_api import BitgetAPI
    from config import Config
except ImportError as e:
    print(f"❌ Erro ao importar módulos: {e}")
    print("Certifique-se de que os arquivos trading_bot.py, bitget_api.py e config.py estão presentes")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

#  Criar aplicação Flask
app = Flask(__name__, static_folder='client/dist', static_url_path='/')
CORS(app, origins=['*'], allow_headers=['*'], methods=['*'], supports_credentials=True) 

# Configurações
app.config.from_object(Config)

# Instâncias globais
bitget_api = None
trading_bot = None
bot_thread = None

def initialize_bot():
    """Inicializar bot de trading"""
    global bitget_api, trading_bot
    
    try:
        # Inicializar API da Bitget
        bitget_api = BitgetAPI(
            api_key=os.getenv('BITGET_API_KEY'),
            secret_key=os.getenv('BITGET_SECRET'),
            passphrase=os.getenv('BITGET_PASSPHRASE'),
            sandbox=False  # Usar ambiente de produção
        )
        
        # Inicializar Trading Bot
        trading_bot = TradingBot(
            bitget_api=bitget_api,
            symbol='ETH/USDT:USDT',
            leverage=10,
            balance_percentage=100.0,
            daily_target=350,
            scalping_interval=0.3,
            paper_trading=True  # Iniciar em modo paper trading por segurança
        )
        
        logger.info("✅ Bot inicializado com sucesso")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao inicializar bot: {e}")
        return False

# Rotas da API

@app.route('/')
def  serve_frontend():
    """Servir o frontend React"""
    return send_from_directory(app.static_folder, 'index.html')

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response 

@app.route('/api/status',  methods=['GET', 'OPTIONS'])
def get_bot_status():
    """Obter status atual do bot"""
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'OK'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', '*')
        response.headers.add('Access-Control-Allow-Methods', '*')
        return response
    
    try: 
        if trading_bot is None:
            return jsonify({
                'error': 'Bot não inicializado',
                'is_running': False,
                'initialized': False
            }), 400
        
        status = trading_bot.get_status()
        
        # Adicionar informações extras
        status['initialized'] = True
        status['server_time'] = datetime.now().isoformat()
        
               response = jsonify(status)
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', '*')
        response.headers.add('Access-Control-Allow-Methods', '*')
        return response 
        
    except Exception as e:
        logger.error(f"❌ Erro ao obter status: {e}")
               response = jsonify({
            'error': str(e),
            'is_running': False,
            'initialized': False
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', '*')
        response.headers.add('Access-Control-Allow-Methods', '*')
        response.status_code = 500
        return response 

@app.route('/api/start',  methods=['POST', 'OPTIONS'])
def start_bot():
    """Iniciar o bot de trading"""
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'OK'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', '*')
        response.headers.add('Access-Control-Allow-Methods', '*')
        return response
    
    try: 
        if trading_bot is None:
            if not initialize_bot():
                return jsonify({
                    'success': False,
                    'message': 'Falha ao inicializar bot'
                }), 500
        
        if trading_bot.is_running:
            return jsonify({
                'success': False,
                'message': 'Bot já está rodando'
            }), 400
        
        # Obter parâmetros opcionais do request
        data = request.get_json() or {}
        
        # Atualizar configurações se fornecidas
        if 'symbol' in data:
            trading_bot.symbol = data['symbol']
        if 'leverage' in data:
            trading_bot.leverage = int(data['leverage'])
        if 'paper_trading' in data:
            trading_bot.paper_trading = bool(data['paper_trading'])
        
        # Iniciar bot
        success = trading_bot.start()
        
        if success:
            logger.info("🚀 Bot iniciado via API")
            return jsonify({
                'success': True,
                'message': f'Bot iniciado - Meta: {trading_bot.min_trades_per_day}+ trades por dia',
                'config': {
                    'symbol': trading_bot.symbol,
                    'leverage': trading_bot.leverage,
                    'paper_trading': trading_bot.paper_trading,
                    'min_trades_target': trading_bot.min_trades_per_day
                }
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Falha ao iniciar bot'
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Erro ao iniciar bot: {e}")
        return jsonify({
            'success': False,
            'message': f'Erro: {str(e)}'
        }), 500

@app.route('/api/stop',  methods=['POST', 'OPTIONS'])
def stop_bot():
    """Parar o bot de trading"""
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'OK'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', '*')
        response.headers.add('Access-Control-Allow-Methods', '*')
        return response
    
    try: 
        if trading_bot is None:
            return jsonify({
                'success': False,
                'message': 'Bot não inicializado'
            }), 400
        
        if not trading_bot.is_running:
            return jsonify({
                'success': False,
                'message': 'Bot não está rodando'
            }), 400
        
        # Parar bot
        success = trading_bot.stop()
        
        if success:
            # Obter estatísticas finais
            final_stats = {
                'trades_executed': trading_bot.trades_today,
                'target_achieved': trading_bot.trades_today >= trading_bot.min_trades_per_day,
                'win_rate': (trading_bot.profitable_trades / max(1, trading_bot.total_trades)) * 100,
                'total_profit': trading_bot.total_profit,
                'profitable_trades': trading_bot.profitable_trades,
                'total_trades': trading_bot.total_trades
            }
            
            logger.info("🛑 Bot parado via API")
            return jsonify({
                'success': True,
                'message': 'Bot parado com sucesso',
                'final_stats': final_stats
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Falha ao parar bot'
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Erro ao parar bot: {e}")
        return jsonify({
            'success': False,
            'message': f'Erro: {str(e)}'
        }), 500

@app.route('/api/emergency-stop', methods=['POST'])
def emergency_stop():
    """Parada de emergência"""
    try:
        if trading_bot is None:
            return jsonify({
                'success': False,
                'message': 'Bot não inicializado'
            }), 400
        
        success = trading_bot.emergency_stop()
        
        logger.warning("🚨 Parada de emergência executada via API")
        return jsonify({
            'success': success,
            'message': 'Parada de emergência executada'
        })
        
    except Exception as e:
        logger.error(f"❌ Erro na parada de emergência: {e}")
        return jsonify({
            'success': False,
            'message': f'Erro: {str(e)}'
        }), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    """Obter configuração atual do bot"""
    try:
        if trading_bot is None:
            return jsonify({
                'error': 'Bot não inicializado'
            }), 400
        
        config = {
            'symbol': trading_bot.symbol,
            'leverage': trading_bot.leverage,
            'balance_percentage': trading_bot.balance_percentage,
            'paper_trading': trading_bot.paper_trading,
            'min_trades_per_day': trading_bot.min_trades_per_day,
            'target_trades_per_day': trading_bot.target_trades_per_day,
            'profit_target': trading_bot.profit_target,
            'stop_loss_target': trading_bot.stop_loss_target,
            'max_position_time': trading_bot.max_position_time,
            'min_confidence_to_trade': trading_bot.min_confidence_to_trade
        }
        
        return jsonify(config)
        
    except Exception as e:
        logger.error(f"❌ Erro ao obter configuração: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/config', methods=['POST'])
def update_config():
    """Atualizar configuração do bot"""
    try:
        if trading_bot is None:
            return jsonify({
                'success': False,
                'message': 'Bot não inicializado'
            }), 400
        
        if trading_bot.is_running:
            return jsonify({
                'success': False,
                'message': 'Não é possível alterar configuração com bot rodando'
            }), 400
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'Dados não fornecidos'
            }), 400
        
        # Atualizar configurações permitidas
        allowed_configs = [
            'symbol', 'leverage', 'balance_percentage', 'paper_trading',
            'profit_target', 'stop_loss_target', 'max_position_time'
        ]
        
        updated = []
        for key, value in data.items():
            if key in allowed_configs and hasattr(trading_bot, key):
                setattr(trading_bot, key, value)
                updated.append(key)
        
        logger.info(f"⚙️ Configuração atualizada: {updated}")
        return jsonify({
            'success': True,
            'message': f'Configuração atualizada: {", ".join(updated)}',
            'updated_fields': updated
        })
        
    except Exception as e:
        logger.error(f"❌ Erro ao atualizar configuração: {e}")
        return jsonify({
            'success': False,
            'message': f'Erro: {str(e)}'
        }), 500

@app.route('/api/stats')
def get_daily_stats():
    """Obter estatísticas detalhadas do dia"""
    try:
        if trading_bot is None:
            return jsonify({
                'error': 'Bot não inicializado'
            }), 400
        
        stats = trading_bot.get_daily_stats()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"❌ Erro ao obter estatísticas: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/balance')
def get_account_balance():
    """Obter saldo da conta"""
    try:
        if trading_bot is None:
            return jsonify({
                'error': 'Bot não inicializado'
            }), 400
        
        balance = trading_bot.get_account_balance()
        return jsonify({
            'balance': balance,
            'currency': 'USDT'
        })
        
    except Exception as e:
        logger.error(f"❌ Erro ao obter saldo: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/health')
def health_check():
    """Health check do servidor e bot"""
    try:
        health_status = {
            'server': 'OK',
            'timestamp': datetime.now().isoformat(),
            'bot_initialized': trading_bot is not None,
            'bot_running': trading_bot.is_running if trading_bot else False
        }
        
        # Verificar conexão com API se bot estiver inicializado
        if trading_bot and bitget_api:
            try:
                # Teste rápido de conectividade
                market_data = bitget_api.get_market_data('ETH/USDT:USDT')
                health_status['api_connection'] = 'OK' if market_data else 'FAILED'
            except:
                health_status['api_connection'] = 'FAILED'
        else:
            health_status['api_connection'] = 'NOT_INITIALIZED'
        
        return jsonify(health_status)
        
    except Exception as e:
        logger.error(f"❌ Erro no health check: {e}")
        return jsonify({
            'server': 'ERROR',
            'error': str(e)
        }), 500

# Manipuladores de erro
@app.errorhandler(404)
def not_found(error):
    """Redirecionar 404 para o frontend"""
    return send_from_directory(app.static_folder, 'index.html')

@app.errorhandler(500)
def internal_error(error):
    """Erro interno do servidor"""
    logger.error(f"Erro interno: {error}")
    return jsonify({
        'error': 'Erro interno do servidor',
        'message': str(error)
    }), 500

# Inicialização
if __name__ == '__main__':
    logger.info("🚀 Iniciando servidor Flask Trading Bot")
    
    # Verificar variáveis de ambiente necessárias
    required_env_vars = ['BITGET_API_KEY', 'BITGET_SECRET', 'BITGET_PASSPHRASE']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Variáveis de ambiente obrigatórias não definidas: {missing_vars}")
        logger.error("Defina as variáveis no Render ou arquivo .env")
    else:
        logger.info("✅ Variáveis de ambiente verificadas")
    
    # Tentar inicializar bot na inicialização
    try:
        if not missing_vars:
            initialize_bot()
    except Exception as e:
        logger.warning(f"⚠️ Falha na inicialização automática do bot: {e}")
        logger.info("Bot pode ser inicializado via API /api/start")
    
    # Iniciar servidor
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"🌐 Servidor rodando na porta {port}")
    logger.info("📱 Interface disponível em: http://localhost:{}/")
    logger.info("🔌 API endpoints:")
    logger.info("   GET  /api/status - Status do bot")
    logger.info("   POST /api/start - Iniciar bot")
    logger.info("   POST /api/stop - Parar bot")
    logger.info("   GET  /api/config - Configuração")
    logger.info("   POST /api/config - Atualizar configuração")
    logger.info("   GET  /api/stats - Estatísticas")
    logger.info("   GET  /api/health - Health check")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
