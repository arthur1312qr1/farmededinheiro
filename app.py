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

# Global variables for ultra-fast monitoring
price_monitor = {
    'current_price': 0.0,
    'last_price': 0.0,
    'min_price': float('inf'),
    'max_price': 0.0,
    'price_change': 0.0,
    'price_change_percent': 0.0,
    'last_update': None,
    'monitoring': False,
    'rapid_changes': [],
    'update_count': 0
}

def ultra_fast_price_monitoring():
    """Monitoramento ultra-rápido de preços"""
    global price_monitor, bot
    
    logging.warning("🚀 MONITORAMENTO ULTRA-RÁPIDO INICIADO")
    
    while price_monitor['monitoring']:
        try:
            if not bot:
                time.sleep(0.01)
                continue
            
            start_time = time.perf_counter()
            
            # Obter dados de mercado
            market_data = bot.bitget_api.get_market_data('ETH/USDT:USDT')
            
            if market_data and 'price' in market_data:
                current_price = float(market_data['price'])
                current_time = datetime.now()
                
                # Atualizar preços
                price_monitor['last_price'] = price_monitor['current_price']
                price_monitor['current_price'] = current_price
                price_monitor['last_update'] = current_time
                price_monitor['update_count'] += 1
                
                # Min/Max tracking
                if current_price < price_monitor['min_price']:
                    price_monitor['min_price'] = current_price
                    logging.warning(f"🔻 NOVO MÍNIMO: ${current_price:.2f}")
                
                if current_price > price_monitor['max_price']:
                    price_monitor['max_price'] = current_price
                    logging.warning(f"🔺 NOVO MÁXIMO: ${current_price:.2f}")
                
                # Calcular variação
                if price_monitor['last_price'] > 0:
                    price_change = current_price - price_monitor['last_price']
                    price_change_percent = (price_change / price_monitor['last_price']) * 100
                    
                    price_monitor['price_change'] = price_change
                    price_monitor['price_change_percent'] = price_change_percent
                    
                    # Detectar variações >= 0.05%
                    if abs(price_change_percent) >= 0.05:
                        processing_time = (time.perf_counter() - start_time) * 1000
                        logging.warning(f"⚡ VARIAÇÃO {price_change_percent:.4f}% | ${price_monitor['last_price']:.2f}→${current_price:.2f} | {processing_time:.1f}ms")
            
            # Intervalo ultra-rápido
            time.sleep(0.01)
            
        except Exception as e:
            if price_monitor['update_count'] % 100 == 0:
                logging.error(f"❌ Erro no monitoramento: {e}")
            time.sleep(0.02)

def calculate_100_percent_exact(usdt_balance, eth_price):
    """
    Calcula 100% EXATO do saldo dividido pelo preço atual
    JAMAIS modifica ou reduz o valor - usa EXATAMENTE 100%
    """
    try:
        # DIVISÃO EXATA: 100% do saldo ÷ preço ETH atual
        eth_quantity = usdt_balance / eth_price
        
        logging.warning(f"💎 CÁLCULO 100% EXATO:")
        logging.warning(f"   💰 USDT (100% COMPLETO): {usdt_balance}")
        logging.warning(f"   💎 Preço ETH ATUAL: {eth_price}")
        logging.warning(f"   🧮 Cálculo: {usdt_balance} ÷ {eth_price}")
        logging.warning(f"   📊 Resultado EXATO: {eth_quantity}")
        
        return eth_quantity
        
    except Exception as e:
        logging.error(f"❌ Erro no cálculo 100%: {e}")
        return None

# Initialize APIs and Bot
def init_bot():
    try:
        api_key = os.getenv('BITGET_API_KEY')
        secret_key = os.getenv('BITGET_SECRET')
        passphrase = os.getenv('BITGET_PASSPHRASE')
        
        logging.info(f"🔍 Verificando credenciais:")
        logging.info(f"   API_KEY: {'✅ OK' if api_key else '❌ VAZIO'}")
        logging.info(f"   SECRET: {'✅ OK' if secret_key else '❌ VAZIO'}")
        logging.info(f"   PASSPHRASE: {'✅ OK' if passphrase else '❌ VAZIO'}")
        
        if not all([api_key, secret_key, passphrase]):
            raise Exception("Credenciais não configuradas")
        
        logging.info("✅ MODO PRODUÇÃO - 100% ABSOLUTO DO SALDO")
        
        # Initialize Bitget API
        bitget_api = BitgetAPI(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            sandbox=False
        )
        
        # Patch do método place_order para usar 100% ABSOLUTO
        original_place_order = bitget_api.place_order
        
        def patched_place_order(symbol, side, size, price=None, leverage=10):
            try:
                logging.warning(f"🔧 ORDEM COM 100% ABSOLUTO DO SALDO:")
                
                # Obter saldo ATUAL e COMPLETO (sempre buscar valor atualizado)
                current_balance = bitget_api.get_account_balance()
                
                # Obter preço ATUAL do ETH (sempre buscar valor atualizado)
                if price is None:
                    market_data = bitget_api.get_market_data(symbol)
                    current_price = float(market_data['price'])
                else:
                    current_price = float(price)
                
                # CALCULAR 100% ABSOLUTO - SEM REDUÇÃO
                eth_quantity = calculate_100_percent_exact(current_balance, current_price)
                
                if eth_quantity is None:
                    return {'success': False, 'error': 'Erro no cálculo 100%'}
                
                logging.warning(f"🚀 ENVIANDO PARA BITGET:")
                logging.warning(f"   💰 Saldo Atual: {current_balance}")
                logging.warning(f"   💎 Preço Atual: {current_price}")
                logging.warning(f"   📊 Quantidade ETH: {eth_quantity}")
                
                # Chamar método original com quantidade 100% EXATA
                return original_place_order(symbol, side, eth_quantity, current_price, leverage)
                
            except Exception as e:
                logging.error(f"❌ Erro na ordem 100%: {e}")
                return {'success': False, 'error': str(e)}
        
        # Aplicar patch
        bitget_api.place_order = patched_place_order
        
        # Initialize Trading Bot
        trading_bot = TradingBot(
            bitget_api=bitget_api,
            symbol='ETH/USDT:USDT',
            leverage=10,
            balance_percentage=100.0,  # 100% ABSOLUTO
            daily_target=200,
            scalping_interval=2,
            paper_trading=False
        )
        
        logging.info("🚀 Bot inicializado - 100% ABSOLUTO DO SALDO")
        return trading_bot
        
    except Exception as e:
        logging.error(f"❌ Falha na inicialização: {e}")
        return None

# Initialize bot
bot = init_bot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    try:
        if not bot:
            return jsonify({
                'error': 'Bot não inicializado',
                'is_running': False
            }), 500
        
        stats = bot.get_status()
        
        # Adicionar dados de monitoramento
        stats.update({
            'ultra_monitoring': {
                'current_price': price_monitor['current_price'],
                'min_price': price_monitor['min_price'] if price_monitor['min_price'] != float('inf') else 0,
                'max_price': price_monitor['max_price'],
                'price_change_percent': price_monitor['price_change_percent'],
                'monitoring_active': price_monitor['monitoring'],
                'update_count': price_monitor['update_count']
            }
        })
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/start', methods=['POST'])
def start_bot():
    try:
        if not bot:
            return jsonify({'error': 'Bot não inicializado', 'success': False}), 500
        
        bot.start()
        
        # Iniciar monitoramento ultra-rápido
        if not price_monitor['monitoring']:
            price_monitor['monitoring'] = True
            monitor_thread = threading.Thread(target=ultra_fast_price_monitoring, daemon=True)
            monitor_thread.start()
            logging.warning("🎯 Monitoramento ultra-rápido iniciado")
        
        logging.warning("🟢 Bot iniciado - 100% ABSOLUTO + REINVESTIMENTO TOTAL")
        
        return jsonify({
            'message': 'Bot iniciado - 100% ABSOLUTO do saldo + reinvestimento total',
            'status': 'running',
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/test_100_percent')
def test_100_percent():
    """Testar cálculo de 100% ABSOLUTO"""
    try:
        if not bot:
            return jsonify({'error': 'Bot não inicializado'}, 500)
        
        # Obter valores ATUAIS (sempre buscar dados frescos)
        current_balance = bot.get_account_balance()
        market_data = bot.bitget_api.get_market_data('ETH/USDT:USDT')
        current_price = float(market_data['price'])
        
        # Calcular 100% ABSOLUTO
        eth_quantity = calculate_100_percent_exact(current_balance, current_price)
        
        return jsonify({
            'current_balance_exact': current_balance,
            'current_price_exact': current_price,
            'calculation': f"{current_balance} ÷ {current_price}",
            'eth_quantity_exact': eth_quantity,
            'percentage_used': 100.0,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/current_values')
def current_values():
    """Ver valores atuais em tempo real"""
    try:
        if not bot:
            return jsonify({'error': 'Bot não inicializado'}, 500)
        
        current_balance = bot.get_account_balance()
        market_data = bot.bitget_api.get_market_data('ETH/USDT:USDT')
        current_price = float(market_data['price'])
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'usdt_balance': current_balance,
            'eth_price': current_price,
            'eth_quantity_if_buy_now': current_balance / current_price,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    try:
        if bot:
            bot.stop()
            price_monitor['monitoring'] = False
            logging.warning("🔴 Bot e monitoramento parados")
        
        return jsonify({'message': 'Bot parado', 'status': 'stopped', 'success': True})
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/pause', methods=['POST'])
def pause_bot():
    try:
        if not bot:
            return jsonify({'error': 'Bot não inicializado', 'success': False}), 500
        
        if hasattr(bot, 'pause'):
            bot.pause()
            return jsonify({'message': 'Bot pausado', 'status': 'paused', 'success': True})
        else:
            return jsonify({'error': 'Método pause não disponível', 'success': False}), 400
            
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/emergency_stop', methods=['POST'])
def emergency_stop():
    try:
        if bot:
            bot.stop()
            price_monitor['monitoring'] = False
            logging.warning("🚨 PARADA DE EMERGÊNCIA")
        
        return jsonify({
            'message': 'Parada de emergência executada',
            'status': 'emergency_stopped',
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/balance')
def get_balance():
    try:
        if not bot:
            return jsonify({'error': 'Bot não inicializado', 'success': False}), 500
        
        # Usar o método get_balance diretamente
        balance_info = bot.bitget_api.get_balance()
        
        if balance_info:
            free_balance = balance_info.get('free', 0)
            return jsonify({
                'balance': free_balance,
                'currency': 'USDT',
                'leverage_power': free_balance * 10,
                'total': balance_info.get('total', 0),
                'used': balance_info.get('used', 0),
                'success': True
            })
        else:
            return jsonify({
                'balance': 0,
                'currency': 'USDT', 
                'leverage_power': 0,
                'success': False,
                'error': 'Não foi possível obter saldo'
            })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/logs')
def get_logs():
    return jsonify({
        'message': 'Bot ativo com 100% ABSOLUTO do saldo',
        'status': 'active',
        'logs': [],
        'success': True
    })

@app.route('/api/config', methods=['POST'])
def update_config():
    try:
        if not bot:
            return jsonify({'error': 'Bot não inicializado', 'success': False}), 500
        
        data = request.get_json()
        if data:
            bot.update_config(**data)
            return jsonify({'message': 'Configuração atualizada', 'success': True})
        else:
            return jsonify({'error': 'Dados não fornecidos', 'success': False}), 400
            
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    app.run(debug=True)
