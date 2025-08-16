import logging
import os
import time
import threading
import asyncio
import aiohttp
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from bitget_api import BitgetAPI
from trading_bot import TradingBot

# Load environment variables
load_dotenv()

# Configure logging (reduzido para não impactar performance)
logging.basicConfig(
    level=logging.WARNING,  # Apenas warnings e errors para reduzir I/O
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
    'price_cache': [],  # Cache para múltiplas leituras simultâneas
    'update_count': 0,
    'avg_response_time': 0.0
}

# Thread pool para operações paralelas
executor = ThreadPoolExecutor(max_workers=4)

def ultra_fast_price_monitoring():
    """Thread de monitoramento ULTRA-RÁPIDO com múltiplas otimizações"""
    global price_monitor, bot
    
    print("🚀 MONITORAMENTO ULTRA-RÁPIDO INICIADO - Detectando variações >= 0.05%")
    
    # Cache de conexão para reutilizar
    session = None
    
    while price_monitor['monitoring']:
        try:
            if not bot:
                time.sleep(0.01)
                continue
                
            # Timer para medir velocidade
            start_time = time.perf_counter()
            
            # Buscar preço usando múltiplas estratégias simultaneamente
            market_data = bot.bitget_api.get_market_data('ethusdt_UMCBL')
            
            if market_data and 'price' in market_data:
                current_price = float(market_data['price'])
                current_time = datetime.now()
                
                # Cálculos otimizados inline
                last_price = price_monitor['current_price']
                price_monitor['last_price'] = last_price
                price_monitor['current_price'] = current_price
                price_monitor['last_update'] = current_time
                price_monitor['update_count'] += 1
                
                # Atualizar min/max com verificação rápida
                if current_price < price_monitor['min_price']:
                    price_monitor['min_price'] = current_price
                    print(f"🔻 NOVO MÍNIMO: ${current_price:.4f}")
                
                if current_price > price_monitor['max_price']:
                    price_monitor['max_price'] = current_price
                    print(f"🔺 NOVO MÁXIMO: ${current_price:.4f}")
                
                # Cálculo ultra-rápido de variação
                if last_price > 0:
                    price_change = current_price - last_price
                    price_change_percent = (price_change / last_price) * 100
                    
                    price_monitor['price_change'] = price_change
                    price_monitor['price_change_percent'] = price_change_percent
                    
                    # Detectar variações críticas (>= 0.05%)
                    if abs(price_change_percent) >= 0.05:
                        processing_time = (time.perf_counter() - start_time) * 1000
                        
                        # Log otimizado apenas para mudanças importantes
                        print(f"⚡ VARIAÇÃO {price_change_percent:.4f}% | ${last_price:.4f}→${current_price:.4f} | {processing_time:.1f}ms")
                        
                        # Cache otimizado - só guarda últimas 50 para velocidade
                        rapid_change = {
                            'timestamp': current_time.isoformat(),
                            'from_price': last_price,
                            'to_price': current_price,
                            'change_percent': price_change_percent,
                            'processing_time_ms': processing_time
                        }
                        
                        price_monitor['rapid_changes'].append(rapid_change)
                        if len(price_monitor['rapid_changes']) > 50:
                            price_monitor['rapid_changes'].pop(0)
                
                # Calcular tempo médio de resposta
                total_time = (time.perf_counter() - start_time) * 1000
                price_monitor['avg_response_time'] = (
                    (price_monitor['avg_response_time'] * (price_monitor['update_count'] - 1) + total_time) 
                    / price_monitor['update_count']
                )
            
            # INTERVALO ULTRA-RÁPIDO - 10ms para detectar mudanças instantâneas
            time.sleep(0.01)
            
        except Exception as e:
            # Log minimal para errors
            if price_monitor['update_count'] % 100 == 0:  # Log apenas a cada 100 erros
                print(f"❌ Erro no monitoramento: {e}")
            time.sleep(0.02)  # Pausa mais curta em erro

# Função adicional para monitoramento paralelo (múltiplas threads)
def parallel_price_monitor():
    """Monitor adicional para redundância e velocidade"""
    global price_monitor
    
    while price_monitor['monitoring']:
        try:
            # Executar verificação paralela a cada 5ms offset
            time.sleep(0.005)
            
            if bot:
                # Quick price check alternativo
                market_data = bot.bitget_api.get_market_data('ethusdt_UMCBL')
                if market_data and 'price' in market_data:
                    # Cache de backup para comparação
                    backup_price = float(market_data['price'])
                    if abs(backup_price - price_monitor['current_price']) > 0.01:
                        print(f"🔄 VERIFICAÇÃO PARALELA: Diferença detectada ${backup_price:.4f}")
                        
        except:
            pass

# Initialize APIs and Bot
def init_bot():
    try:
        # Get API credentials from environment
        api_key = os.getenv('BITGET_API_KEY')
        secret_key = os.getenv('BITGET_SECRET')
        passphrase = os.getenv('BITGET_PASSPHRASE')
        
        # Debug: Log se as variáveis estão sendo carregadas
        print(f"🔍 Verificando credenciais:")
        print(f"   API_KEY: {'✅ OK' if api_key else '❌ VAZIO'}")
        print(f"   SECRET: {'✅ OK' if secret_key else '❌ VAZIO'}")
        print(f"   PASSPHRASE: {'✅ OK' if passphrase else '❌ VAZIO'}")
        
        # FAIL HARD if no credentials
        if not all([api_key, secret_key, passphrase]):
            raise Exception("Credenciais não configuradas no ambiente")
        
        if api_key == "test_key" or secret_key == "test_secret" or passphrase == "test_pass":
            raise Exception("Credenciais de teste não são permitidas")
        
        print("✅ MODO PRODUÇÃO - VELOCIDADE MÁXIMA")
        
        # Initialize Bitget API
        bitget_api = BitgetAPI(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            sandbox=False
        )
        
        # Initialize Trading Bot
        trading_bot = TradingBot(
            bitget_api=bitget_api,
            symbol='ethusdt_UMCBL',
            leverage=10,
            balance_percentage=100.0,
            daily_target=200,
            scalping_interval=2,
            paper_trading=False
        )
        
        print("🚀 Bot inicializado - MODO ULTRA-RÁPIDO")
        return trading_bot
        
    except Exception as e:
        print(f"❌ Falha na inicialização: {e}")
        return None

# Initialize bot globally
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
                'status': 'error',
                'is_running': False,
                'is_paused': False,
                'daily_trades': 0,
                'win_rate': 0.0
            }), 500
        
        stats = bot.get_status()
        
        # Adicionar dados de monitoramento ultra-rápido
        stats.update({
            'ultra_monitoring': {
                'current_price': price_monitor['current_price'],
                'min_price': price_monitor['min_price'] if price_monitor['min_price'] != float('inf') else 0,
                'max_price': price_monitor['max_price'],
                'price_change_percent': price_monitor['price_change_percent'],
                'last_update': price_monitor['last_update'].isoformat() if price_monitor['last_update'] else None,
                'monitoring_active': price_monitor['monitoring'],
                'update_count': price_monitor['update_count'],
                'avg_response_time_ms': round(price_monitor['avg_response_time'], 2),
                'rapid_changes_count': len(price_monitor['rapid_changes'])
            }
        })
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/start', methods=['POST'])
def start_bot():
    try:
        if not bot:
            return jsonify({
                'error': 'Bot não inicializado',
                'success': False
            }), 500
        
        bot.start()
        
        # Iniciar monitoramento ULTRA-RÁPIDO com múltiplas threads
        if not price_monitor['monitoring']:
            price_monitor['monitoring'] = True
            
            # Thread principal ultra-rápida
            main_thread = threading.Thread(target=ultra_fast_price_monitoring, daemon=True)
            main_thread.start()
            
            # Thread paralela para redundância
            parallel_thread = threading.Thread(target=parallel_price_monitor, daemon=True)
            parallel_thread.start()
            
            print("🚀 MONITORAMENTO ULTRA-RÁPIDO ATIVO - 10ms de intervalo")
        
        print("🟢 Bot + Monitoramento Ultra-Rápido ATIVO")
        return jsonify({
            'message': 'Bot iniciado - Monitoramento Ultra-Rápido (10ms)',
            'status': 'running',
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
        print("🔴 Bot e monitoramento ultra-rápido parados")
        
        return jsonify({
            'message': 'Bot e monitoramento parados',
            'status': 'stopped',
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/ultra_data')
def get_ultra_data():
    """Get ultra-fast monitoring data"""
    try:
        return jsonify({
            'current_price': price_monitor['current_price'],
            'min_price': price_monitor['min_price'] if price_monitor['min_price'] != float('inf') else 0,
            'max_price': price_monitor['max_price'],
            'price_change_percent': price_monitor['price_change_percent'],
            'last_update': price_monitor['last_update'].isoformat() if price_monitor['last_update'] else None,
            'monitoring_active': price_monitor['monitoring'],
            'update_count': price_monitor['update_count'],
            'avg_response_time_ms': round(price_monitor['avg_response_time'], 2),
            'rapid_changes': price_monitor['rapid_changes'][-5:],  # Últimas 5 mudanças
            'rapid_changes_count': len(price_monitor['rapid_changes']),
            'success': True
        })
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
        print("🚨 PARADA DE EMERGÊNCIA")
        
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
        'message': 'Monitoramento ultra-rápido ativo',
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
        print("🚀 MODO ULTRA-RÁPIDO ATIVO")
        print("⚡ Monitoramento: 10ms de intervalo")
        print("🎯 Detecta variações >= 0.05% instantaneamente")
        print("🔥 Múltiplas threads paralelas")
    else:
        print("❌ FALHA CRÍTICA: Configure credenciais no Render.com")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
