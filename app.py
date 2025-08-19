from flask import Flask, jsonify
from flask_cors import CORS
import os
import logging
import time
import threading
import traceback

# Importar arquivos existentes - CORREÇÃO
try:
    from bitget_api import BitgetAPI
    from config import get_config
    from trading_bot import TradingBot
    
    config = get_config()
    
    # Teste direto da API
    print("📡 Testando conexão BitgetAPI...")
    api = BitgetAPI(
        api_key=config.get('BITGET_API_KEY'),
        secret_key=config.get('BITGET_SECRET_KEY'), 
        passphrase=config.get('BITGET_PASSPHRASE')
    )
    
    # Testar saldo diretamente
    try:
        test_balance = api.get_balance()
        test_price = api.get_eth_price()
        print(f"✅ Teste API: Saldo={test_balance}, Preço={test_price}")
        API_CONNECTED = True
    except Exception as e:
        print(f"❌ Erro no teste API: {e}")
        API_CONNECTED = False
    
    # Inicializar TradingBot
    trading_bot = TradingBot(
        api=api,
        symbol=config.get('SYMBOL', 'ETH/USDT:USDT'),
        leverage=int(config.get('LEVERAGE', 10)),
        paper_trading=config.get('PAPER_TRADING', 'true').lower() == 'true'
    )
    
    print("✅ Sistema inicializado!")
    
except Exception as e:
    print(f"❌ ERRO CRÍTICO: {e}")
    traceback.print_exc()
    api = None
    trading_bot = None
    API_CONNECTED = False

app = Flask(__name__)
CORS(app)

# Estado global
bot_state = {
    'is_running': False,
    'is_paused': False,
    'trades_today': 0,
    'total_profit': 0.0,
    'thread': None,
    'last_error': None
}

def trading_loop():
    """Loop de trading que usa o trading_bot.py"""
    print("🚀 Iniciando loop de trading...")
    
    while bot_state['is_running'] and not bot_state['is_paused']:
        try:
            if trading_bot and API_CONNECTED:
                print("🔄 Executando trading_bot...")
                
                # Chamar métodos do trading_bot
                if hasattr(trading_bot, 'run'):
                    result = trading_bot.run()
                elif hasattr(trading_bot, 'execute'):
                    result = trading_bot.execute()
                elif hasattr(trading_bot, 'main_loop'):
                    result = trading_bot.main_loop()
                else:
                    # Tentar métodos individuais
                    try:
                        if hasattr(trading_bot, 'analyze_market'):
                            analysis = trading_bot.analyze_market()
                            print(f"📊 Análise: {analysis}")
                            
                        if hasattr(trading_bot, 'execute_trade'):
                            trade_result = trading_bot.execute_trade()
                            if trade_result:
                                bot_state['trades_today'] += 1
                                print(f"✅ Trade executado! Total: {bot_state['trades_today']}")
                    except Exception as e:
                        print(f"❌ Erro nos métodos individuais: {e}")
                
                bot_state['last_error'] = None
                
        except Exception as e:
            error_msg = f"Erro no trading loop: {e}"
            print(f"❌ {error_msg}")
            bot_state['last_error'] = error_msg
            
        time.sleep(3)  # Intervalo entre execuções
    
    print("🛑 Trading loop finalizado")

@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Trading Bot</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial; background: #1a1a1a; color: white; padding: 20px; }
        .card { background: #333; padding: 20px; margin: 10px; border-radius: 8px; }
        .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; }
        .metric { text-align: center; }
        .metric-value { font-size: 2em; margin-bottom: 10px; }
        .btn { padding: 15px 25px; margin: 5px; border: none; border-radius: 5px; color: white; font-size: 16px; cursor: pointer; }
        .btn-start { background: #4CAF50; }
        .btn-pause { background: #ff9800; }
        .btn-stop { background: #f44336; }
    </style>
</head>
<body>
    <h1>🤖 Trading Bot Dashboard</h1>
    
    <div class="card">
        <h2>Status: <span id="status">PARADO</span></h2>
        <button id="btn-start" class="btn btn-start">🚀 Iniciar</button>
        <button id="btn-pause" class="btn btn-pause">⏸️ Pausar</button>
        <button id="btn-stop" class="btn btn-stop">🛑 Parar</button>
    </div>
    
    <div class="grid">
        <div class="card metric">
            <div id="trades" class="metric-value" style="color: #2196F3;">0</div>
            <div>Trades Hoje</div>
        </div>
        <div class="card metric">
            <div id="balance" class="metric-value" style="color: #9C27B0;">$0.00</div>
            <div>Saldo USDT</div>
        </div>
        <div class="card metric">
            <div id="price" class="metric-value" style="color: #FF5722;">$0.00</div>
            <div>Preço ETH</div>
        </div>
        <div class="card metric">
            <div id="profit" class="metric-value" style="color: #FFC107;">$0.00</div>
            <div>Lucro Total</div>
        </div>
    </div>
    
    <div class="card">
        <h3>🔧 Sistema</h3>
        <div id="info">Carregando...</div>
        <div id="debug"></div>
    </div>
    
    <div class="card">
        <h3>⚠️ Status</h3>
        <div id="errors">Sistema funcionando</div>
    </div>

    <script>
        document.getElementById('btn-start').addEventListener('click', function() {
            fetch('/api/start', { method: 'POST' })
                .then(r => r.json())
                .then(data => console.log('Start:', data));
        });
        
        document.getElementById('btn-pause').addEventListener('click', function() {
            fetch('/api/pause', { method: 'POST' })
                .then(r => r.json())
                .then(data => console.log('Pause:', data));
        });
        
        document.getElementById('btn-stop').addEventListener('click', function() {
            fetch('/api/stop', { method: 'POST' })
                .then(r => r.json())
                .then(data => console.log('Stop:', data));
        });
        
        function updateData() {
            fetch('/api/data')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('trades').textContent = data.trades || 0;
                    document.getElementById('balance').textContent = '$' + (data.balance || 0).toFixed(4);
                    document.getElementById('price').textContent = '$' + (data.price || 0).toFixed(2);
                    document.getElementById('profit').textContent = '$' + (data.profit || 0).toFixed(4);
                    document.getElementById('status').textContent = data.status || 'PARADO';
                    document.getElementById('info').textContent = data.info || 'Sistema ativo';
                    document.getElementById('debug').textContent = 'Atualizado: ' + new Date().toLocaleTimeString();
                    document.getElementById('errors').textContent = data.error || '✅ Funcionando';
                })
                .catch(e => {
                    console.error('Erro:', e);
                    document.getElementById('errors').textContent = '❌ Erro: ' + e.message;
                });
        }
        
        updateData();
        setInterval(updateData, 3000);
    </script>
</body>
</html>
'''

@app.route('/api/data')
def get_data():
    """Endpoint que busca dados reais"""
    try:
        print("📊 Buscando dados...")
        
        # Buscar saldo DIRETAMENTE da API
        balance = 0
        try:
            if api and API_CONNECTED:
                balance_info = api.get_balance()
                print(f"💰 Balance info: {balance_info}")
                if balance_info and isinstance(balance_info, dict):
                    balance = balance_info.get('free', 0)
                    print(f"💰 Saldo extraído: {balance}")
        except Exception as e:
            print(f"❌ Erro ao buscar saldo: {e}")
        
        # Buscar preço DIRETAMENTE da API  
        price = 0
        try:
            if api and API_CONNECTED:
                price = api.get_eth_price()
                print(f"📈 Preço ETH: {price}")
        except Exception as e:
            print(f"❌ Erro ao buscar preço: {e}")
        
        # Status do bot
        if bot_state['is_running'] and not bot_state['is_paused']:
            status = 'RODANDO'
        elif bot_state['is_paused']:
            status = 'PAUSADO'
        else:
            status = 'PARADO'
        
        response = {
            'trades': bot_state['trades_today'],
            'balance': balance,
            'price': price,
            'profit': bot_state['total_profit'],
            'status': status,
            'info': f'API: {API_CONNECTED}, Trades: {bot_state["trades_today"]}',
            'error': bot_state['last_error']
        }
        
        print(f"✅ Resposta: {response}")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Erro em get_data: {e}")
        traceback.print_exc()
        return jsonify({
            'trades': 0, 'balance': 0, 'price': 0, 'profit': 0,
            'status': 'ERRO', 'error': str(e)
        })

@app.route('/api/start', methods=['POST'])
def start():
    """Iniciar bot"""
    try:
        bot_state['is_running'] = True
        bot_state['is_paused'] = False
        
        if not bot_state['thread'] or not bot_state['thread'].is_alive():
            bot_state['thread'] = threading.Thread(target=trading_loop, daemon=True)
            bot_state['thread'].start()
            print("🚀 Thread de trading iniciada!")
        
        return jsonify({'success': True, 'message': 'Bot iniciado'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/pause', methods=['POST'])
def pause():
    bot_state['is_paused'] = True
    return jsonify({'success': True, 'message': 'Bot pausado'})

@app.route('/api/stop', methods=['POST'])
def stop():
    bot_state['is_running'] = False
    bot_state['is_paused'] = False
    return jsonify({'success': True, 'message': 'Bot parado'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Iniciando na porta {port}")
    print(f"🔗 API: {'✅ Conectada' if API_CONNECTED else '❌ Desconectada'}")
    app.run(host='0.0.0.0', port=port, debug=True)
