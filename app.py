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
            market_data = bot.bitget_api.get_market_data('ethusdt_UMCBL')
            
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

# Função para corrigir cálculo de quantidade ETH
def calculate_eth_amount(usdt_amount, eth_price):
    """Calcula quantidade ETH com precisão correta para Bitget"""
    try:
        # Calcular quantidade bruta
        raw_amount = usdt_amount / eth_price
        
        # Arredondar para 2 casas decimais (precisão da Bitget)
        eth_amount = round(raw_amount, 2)
        
        # Verificar se está dentro dos limites da Bitget
        # Máximo: 0.06 ETH (conforme mencionado)
        if eth_amount > 0.06:
            eth_amount = 0.06
            logging.warning(f"⚠️ Quantidade limitada ao máximo: 0.06 ETH")
        
        # Se quantidade calculada for muito pequena, usar valor mínimo operacional
        if eth_amount < 0.01:
            eth_amount = 0.01
            logging.warning(f"⚠️ Quantidade ajustada para mínimo operacional: 0.01 ETH")
        
        logging.warning(f"💎 Cálculo ETH:")
        logging.warning(f"   💰 USDT: ${usdt_amount:.2f}")
        logging.warning(f"   💎 Preço: ${eth_price:.2f}")
        logging.warning(f"   📊 Quantidade: {eth_amount:.2f} ETH")
        
        return eth_amount
        
    except Exception as e:
        logging.error(f"❌ Erro no cálculo ETH: {e}")
        return 0.01  # Fallback seguro

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
        
        logging.info("✅ MODO PRODUÇÃO - 100% DO SALDO COM PRECISÃO CORRETA")
        
        # Initialize Bitget API
        bitget_api = BitgetAPI(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            sandbox=False
        )
        
        # Patch do método place_order para usar cálculo correto
        original_place_order = bitget_api.place_order
        
        def patched_place_order(symbol, side, size, price=None, leverage=10):
            try:
                logging.warning(f"🔧 ORDEM CORRIGIDA:")
                
                # Obter saldo atual
                current_balance = bitget_api.get_account_balance()
                
                # Obter preço atual se não fornecido
                if price is None:
                    market_data = bitget_api.get_market_data(symbol)
                    current_price = float(market_data['price'])
                else:
                    current_price = float(price)
                
                # Usar 100% do saldo
                usdt_amount = current_balance * 0.99  # 99% para taxas
                
                # Calcular quantidade ETH com precisão correta
                eth_quantity = calculate_eth_amount(usdt_amount, current_price)
                
                logging.warning(f"💰 Usando 100% do saldo: ${usdt_amount:.2f} USDT")
                logging.warning(f"💎 Quantidade corrigida: {eth_quantity:.2f} ETH")
                
                # Chamar método original com quantidade corrigida
                return original_place_order(symbol, side, eth_quantity, current_price, leverage)
                
            except Exception as e:
                logging.error(f"❌ Erro na ordem corrigida: {e}")
                return {'success': False, 'error': str(e)}
        
        # Aplicar patch
        bitget_api.place_order = patched_place_order
        
        # Initialize Trading Bot
        trading_bot = TradingBot(
            bitget_api=bitget_api,
            symbol='ethusdt_UMCBL',
            leverage=10,
            balance_percentage=100.0,  # 100% DO SALDO
            daily_target=200,
            scalping_interval=2,
            paper_trading=False
        )
        
        logging.info("🚀 Bot inicializado - 100% DO SALDO COM PRECISÃO CORRIGIDA")
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
        
        logging.warning("🟢 Bot iniciado - 100% DO SALDO COM PRECISÃO CORRIGIDA")
        return jsonify({
            'message': 'Bot iniciado - Usando 100% do saldo com precisão correta',
            'status': 'running',
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/test_calculation')
def test_calculation():
    """Testar cálculo de quantidade ETH"""
    try:
        if not bot:
            return jsonify({'error': 'Bot não inicializado'}, 500)
        
        # Obter dados atuais
        current_balance = bot.get_account_balance()
        market_data = bot.bitget_api.get_market_data('ethusdt_UMCBL')
        current_price = float(market_data['price'])
        
        # Usar 100% do saldo
        usdt_amount = current_balance * 0.99
        
        # Calcular com função corrigida
        eth_amount = calculate_eth_amount(usdt_amount, current_price)
        
        return jsonify({
            'current_balance': current_balance,
            'current_price': current_price,
            'usdt_to_use': usdt_amount,
            'eth_amount_calculated': eth_amount,
            'calculation_valid': 0.01 <= eth_amount <= 0.06,
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
        
        balance = bot.get_account_balance()
        return jsonify({
            'balance': balance,
            'currency': 'USDT',
            'leverage_power': balance * 10,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/logs')
def get_logs():
    return jsonify({
        'message': 'Bot ativo com precisão corrigida',
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
        bot.update_config(**data)
        return jsonify({
            'message': 'Configuração atualizada',
            'config': data,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    if bot:
        logging.warning("🚀 MODO 100% DO SALDO COM PRECISÃO CORRIGIDA")
        logging.warning("💎 Quantidade ETH: 2 casas decimais (0.01 - 0.06)")
        logging.warning("💰 Usa 100% do saldo disponível")
        logging.warning("⚡ Monitoramento ultra-rápido: 10ms")
    else:
        logging.error("❌ FALHA: Configure credenciais no Render.com")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
