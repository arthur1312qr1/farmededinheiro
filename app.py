import logging
import os
import time
import threading
from datetime import datetime
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from bitget_api import BitgetAPI
from trading_bot import TradingBot

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)

# Global variables for real-time monitoring
price_monitor = {
    'current_price': 0.0,
    'last_price': 0.0,
    'min_price': float('inf'),
    'max_price': 0.0,
    'price_change': 0.0,
    'price_change_percent': 0.0,
    'last_update': None,
    'monitoring': False,
    'rapid_changes': []
}

# Initialize APIs and Bot
def init_bot():
    try:
        # Get API credentials from environment
        api_key = os.getenv('BITGET_API_KEY')
        secret_key = os.getenv('BITGET_SECRET')
        passphrase = os.getenv('BITGET_PASSPHRASE')
        
        # Debug: Log se as variáveis estão sendo carregadas
        logging.info(f"🔍 Verificando credenciais:")
        logging.info(f"   API_KEY: {'✅ OK' if api_key else '❌ VAZIO'}")
        logging.info(f"   SECRET: {'✅ OK' if secret_key else '❌ VAZIO'}")
        logging.info(f"   PASSPHRASE: {'✅ OK' if passphrase else '❌ VAZIO'}")
        
        # FAIL HARD if no credentials
        if not all([api_key, secret_key, passphrase]):
            logging.error("❌ ERRO CRÍTICO: Credenciais obrigatórias não encontradas")
            logging.error("❌ Variáveis necessárias: BITGET_API_KEY, BITGET_SECRET, BITGET_PASSPHRASE")
            raise Exception("Credenciais não configuradas no ambiente")
        
        # Validate that credentials are not test values
        if api_key == "test_key" or secret_key == "test_secret" or passphrase == "test_pass":
            logging.error("❌ ERRO: Credenciais de teste detectadas - configure credenciais reais")
            raise Exception("Credenciais de teste não são permitidas")
        
        logging.info("✅ Credenciais reais encontradas - INICIANDO MODO PRODUÇÃO")
        
        # Initialize Bitget API - PRODUCTION MODE ONLY
        bitget_api = BitgetAPI(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            sandbox=False  # PRODUCTION MODE
        )
        
        # Initialize Trading Bot - REAL TRADING ONLY
        trading_bot = TradingBot(
            bitget_api=bitget_api,
            symbol='ethusdt_UMCBL',
            leverage=10,
            balance_percentage=100.0,
            daily_target=200,
            scalping_interval=2,
            paper_trading=False  # REAL TRADING
        )
        
        logging.info("🚀 Bot inicializado em MODO PRODUÇÃO - Trading Real")
        return trading_bot
        
    except Exception as e:
        logging.error(f"❌ Falha na inicialização: {e}")
        return None

def price_monitoring_thread():
    """Thread para monitoramento ultra-rápido de preços"""
    global price_monitor, bot
    
    logging.info("🎯 Iniciando monitoramento de preços em tempo real")
    
    while price_monitor['monitoring']:
        try:
            if not bot:
                time.sleep(1)
                continue
                
            # Obter preço atual - ALTA VELOCIDADE
            start_time = time.time()
            market_data = bot.bitget_api.get_market_data('ethusdt_UMCBL')
            
            if market_data and 'price' in market_data:
                current_price = float(market_data['price'])
                current_time = datetime.now()
                
                # Atualizar dados de preço
                price_monitor['last_price'] = price_monitor['current_price']
                price_monitor['current_price'] = current_price
                price_monitor['last_update'] = current_time
                
                # Calcular valor mínimo em tempo real
                if current_price < price_monitor['min_price']:
                    price_monitor['min_price'] = current_price
                    logging.info(f"🔻 NOVO MÍNIMO DETECTADO: ${current_price:.2f}")
                
                # Calcular valor máximo
                if current_price > price_monitor['max_price']:
                    price_monitor['max_price'] = current_price
                    logging.info(f"🔺 NOVO MÁXIMO DETECTADO: ${current_price:.2f}")
                
                # Calcular variação se temos preço anterior
                if price_monitor['last_price'] > 0:
                    price_change = current_price - price_monitor['last_price']
                    price_change_percent = (price_change / price_monitor['last_price']) * 100
                    
                    price_monitor['price_change'] = price_change
                    price_monitor['price_change_percent'] = price_change_percent
                    
                    # Detectar variações rápidas (>= 0.05%)
                    if abs(price_change_percent) >= 0.05:
                        rapid_change = {
                            'timestamp': current_time.isoformat(),
                            'from_price': price_monitor['last_price'],
                            'to_price': current_price,
                            'change': price_change,
                            'change_percent': price_change_percent,
                            'processing_time_ms': (time.time() - start_time) * 1000
                        }
                        
                        price_monitor['rapid_changes'].append(rapid_change)
                        
                        # Manter apenas últimas 100 mudanças rápidas
                        if len(price_monitor['rapid_changes']) > 100:
                            price_monitor['rapid_changes'].pop(0)
                        
                        logging.warning(f"⚡ VARIAÇÃO RÁPIDA: {price_change_percent:.4f}% em {(time.time() - start_time)*1000:.1f}ms")
                        logging.warning(f"💰 ${price_monitor['last_price']:.2f} → ${current_price:.2f}")
                
                # Log de velocidade (apenas para variações significativas)
                processing_time = (time.time() - start_time) * 1000
                if abs(price_monitor['price_change_percent']) >= 0.01:  # Log apenas para mudanças >= 0.01%
                    logging.info(f"🚀 Preço: ${current_price:.2f} | Variação: {price_monitor['price_change_percent']:.4f}% | Tempo: {processing_time:.1f}ms")
            
            # Intervalo ultra-rápido - 100ms para detectar mudanças de 0.05% por segundo
            time.sleep(0.1)
            
        except Exception as e:
            logging.error(f"❌ Erro no monitoramento: {e}")
            time.sleep(0.5)  # Espera um pouco mais em caso de erro

# Initialize bot globally
bot = init_bot()

@app.route('/')
def home():
    """Serve the dashboard website"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get real bot status with price monitoring"""
    try:
        if not bot:
            return jsonify({
                'error': 'Bot não inicializado - verifique credenciais no Render.com', 
                'status': 'error',
                'is_running': False,
                'is_paused': False,
                'daily_trades': 0,
                'win_rate': 0.0
            }), 500
        
        stats = bot.get_status()
        
        # Adicionar dados de monitoramento de preços
        stats.update({
            'price_monitoring': {
                'current_price': price_monitor['current_price'],
                'min_price': price_monitor['min_price'] if price_monitor['min_price'] != float('inf') else 0,
                'max_price': price_monitor['max_price'],
                'price_change': price_monitor['price_change'],
                'price_change_percent': price_monitor['price_change_percent'],
                'last_update': price_monitor['last_update'].isoformat() if price_monitor['last_update'] else None,
                'monitoring_active': price_monitor['monitoring'],
                'rapid_changes_count': len(price_monitor['rapid_changes'])
            }
        })
        
        return jsonify(stats)
        
    except Exception as e:
        logging.error(f"❌ Erro ao obter status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/start', methods=['POST'])
def start_bot():
    """Start the real trading bot with price monitoring"""
    try:
        if not bot:
            return jsonify({
                'error': 'Bot não inicializado - configure credenciais no Render.com',
                'success': False
            }), 500
        
        bot.start()
        
        # Iniciar monitoramento de preços
        if not price_monitor['monitoring']:
            price_monitor['monitoring'] = True
            monitor_thread = threading.Thread(target=price_monitoring_thread, daemon=True)
            monitor_thread.start()
            logging.info("🎯 Monitoramento de preços iniciado")
        
        logging.info("🟢 Bot INICIADO - Trading Real + Monitoramento Ativo")
        return jsonify({
            'message': 'Bot iniciado - Trading Real + Monitoramento de Preços Ativo',
            'status': 'running',
            'success': True
        })
        
    except Exception as e:
        logging.error(f"❌ Erro ao iniciar bot: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    """Stop the real trading bot and price monitoring"""
    try:
        if not bot:
            return jsonify({
                'error': 'Bot não inicializado',
                'success': False
            }), 500
        
        bot.stop()
        
        # Parar monitoramento de preços
        price_monitor['monitoring'] = False
        logging.info("🔴 Monitoramento de preços parado")
        
        logging.info("🔴 Bot PARADO")
        return jsonify({
            'message': 'Bot e monitoramento parados',
            'status': 'stopped',
            'success': True
        })
        
    except Exception as e:
        logging.error(f"❌ Erro ao parar bot: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/price_data')
def get_price_data():
    """Get detailed price monitoring data"""
    try:
        return jsonify({
            'current_price': price_monitor['current_price'],
            'min_price': price_monitor['min_price'] if price_monitor['min_price'] != float('inf') else 0,
            'max_price': price_monitor['max_price'],
            'price_change': price_monitor['price_change'],
            'price_change_percent': price_monitor['price_change_percent'],
            'last_update': price_monitor['last_update'].isoformat() if price_monitor['last_update'] else None,
            'monitoring_active': price_monitor['monitoring'],
            'rapid_changes': price_monitor['rapid_changes'][-10:],  # Últimas 10 mudanças rápidas
            'rapid_changes_count': len(price_monitor['rapid_changes']),
            'success': True
        })
    except Exception as e:
        logging.error(f"❌ Erro ao obter dados de preço: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/reset_price_monitor', methods=['POST'])
def reset_price_monitor():
    """Reset price monitoring data"""
    try:
        price_monitor['min_price'] = float('inf')
        price_monitor['max_price'] = 0.0
        price_monitor['rapid_changes'] = []
        
        logging.info("🔄 Dados de monitoramento resetados")
        return jsonify({
            'message': 'Dados de monitoramento resetados',
            'success': True
        })
    except Exception as e:
        logging.error(f"❌ Erro ao resetar monitoramento: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/pause', methods=['POST'])
def pause_bot():
    """Pause the real trading bot"""
    try:
        if not bot:
            return jsonify({
                'error': 'Bot não inicializado',
                'success': False
            }), 500
        
        if hasattr(bot, 'pause'):
            bot.pause()
            logging.info("⏸️ Bot PAUSADO")
            return jsonify({
                'message': 'Bot pausado',
                'status': 'paused',
                'success': True
            })
        else:
            return jsonify({
                'error': 'Método pause não disponível',
                'success': False
            }), 400
            
    except Exception as e:
        logging.error(f"❌ Erro ao pausar bot: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/emergency_stop', methods=['POST'])
def emergency_stop():
    """Emergency stop the bot and monitoring"""
    try:
        if bot:
            bot.stop()
        
        # Parar monitoramento imediatamente
        price_monitor['monitoring'] = False
        
        logging.warning("🚨 PARADA DE EMERGÊNCIA ATIVADA - Bot e Monitoramento Parados")
        return jsonify({
            'message': 'Parada de emergência - Bot e monitoramento parados',
            'status': 'emergency_stopped',
            'success': True
        })
        
    except Exception as e:
        logging.error(f"❌ Erro na parada de emergência: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/balance')
def get_balance():
    """Get real account balance"""
    try:
        if not bot:
            return jsonify({
                'error': 'Bot não inicializado',
                'success': False
            }), 500
        
        balance = bot.get_account_balance()
        return jsonify({
            'balance': balance,
            'currency': 'USDT',
            'leverage_power': balance * 10,
            'success': True
        })
        
    except Exception as e:
        logging.error(f"❌ Erro ao obter saldo: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/logs')
def get_logs():
    """Get logs from bot"""
    try:
        return jsonify({
            'message': 'Verifique os logs no console do servidor',
            'status': 'active',
            'logs': [],
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update bot configuration"""
    try:
        if not bot:
            return jsonify({
                'error': 'Bot não inicializado',
                'success': False
            }), 500
        
        data = request.get_json()
        bot.update_config(**data)
        logging.info(f"⚙️ Configuração atualizada: {data}")
        return jsonify({
            'message': 'Configuração atualizada',
            'config': data,
            'success': True
        })
        
    except Exception as e:
        logging.error(f"❌ Erro ao atualizar config: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    if bot:
        logging.info("🚀 Trading Bot API iniciada - MODO PRODUÇÃO")
        logging.info("💰 TRADING REAL ATIVO")
        logging.info("🎯 Monitoramento de preços configurado para detectar variações >= 0.05%")
    else:
        logging.error("❌ FALHA CRÍTICA: Bot não pôde ser inicializado")
        logging.error("❌ Configure as variáveis no Render.com:")
        logging.error("❌   BITGET_API_KEY")
        logging.error("❌   BITGET_SECRET") 
        logging.error("❌   BITGET_PASSPHRASE")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
