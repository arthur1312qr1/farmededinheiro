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
    
    # CORREÇÃO: config já retorna bool, não string
    paper_trading = config.get('PAPER_TRADING', False)  # Já é bool!
    
    print("📡 Testando conexão BitgetAPI...")
    api = BitgetAPI(
        api_key=config.get('BITGET_API_KEY'),
        secret_key=config.get('BITGET_SECRET_KEY'), 
        passphrase=config.get('BITGET_PASSPHRASE')
    )
    
    # Testar API diretamente
    try:
        test_balance = api.get_balance()
        test_price = api.get_eth_price()
        print(f"✅ Teste API: Saldo={test_balance}, Preço={test_price}")
        API_CONNECTED = True
    except Exception as e:
        print(f"❌ Erro no teste API: {e}")
        API_CONNECTED = False
    
    # CORREÇÃO: Usar parâmetro correto para TradingBot
    trading_bot = TradingBot(
        bitget_api=api,  # Nome correto do parâmetro
        symbol=config.get('SYMBOL', 'ETH/USDT:USDT'),
        leverage=config.get('LEVERAGE', 10),
        paper_trading=paper_trading  # Já é bool
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
    """Loop que executa o trading_bot"""
    print("🚀 Iniciando loop de trading...")
    
    while bot_state['is_running'] and not bot_state['is_paused']:
        try:
            if trading_bot and API_CONNECTED:
                print("🔄 Executando trading_bot...")
                
                # Chamar start() do trading_bot uma vez
                if not trading_bot.is_running:
                    trading_bot.start()
                
                # Pegar status do trading_bot
                status = trading_bot.get_status()
                if status:
                    bot_state['trades_today'] = status.get('daily_progress', {}).get('trades_today', 0)
                    bot_state['total_profit'] = status.get('performance', {}).get('total_profit', 0)
                
                bot_state['last_error'] = None
                
        except Exception as e:
            error_msg = f"Erro no trading loop: {e}"
            print(f"❌ {error_msg}")
            bot_state['last_error'] = error_msg
            
        time.sleep(5)  # Intervalo maior
    
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
            console.log('🚀 Iniciando bot...');
            fetch('/api/start', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    console.log('Start:', data);
                    updateData();
                });
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
                    console.log('📊 Dados recebidos:', data);
                    
                    document.getElementById('trades').textContent = data.trades || 0;
                    document.getElementById('balance').textContent = '$' + (data.balance || 0).toFixed(4);
                    document.getElementById('price').textContent = '$' + (data.price || 0).toFixed(2);
                    document.getElementById('profit').textContent = '$' + (data.profit || 0).toFixed(4);
                    document.getElementById('status').textContent = data.status || 'PARADO';
                    document.getElementById('info').textContent = data.info || 'Sistema ativo';
                    document.getElementById('debug').textContent = 'Atualizado: ' + new Date().toLocaleTimeString();
                    
                    if (data.error) {
                        document.getElementById('errors').textContent = '❌ ' + data.error;
                    } else {
                        document.getElementById('errors').textContent = '✅ Funcionando';
                    }
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
    """Endpoint que busca dados reais da API"""
    try:
        print("📊 Buscando dados da API...")
        
        # Buscar saldo DIRETAMENTE
        balance = 0
        try:
            if api and API_CONNECTED:
                balance_info = api.get_balance()
                print(f"💰 Balance raw: {balance_info}")
                if balance_info and isinstance(balance_info, dict):
                    balance = balance_info.get('free', 0)
                    print(f"💰 Saldo extraído: ${balance:.4f}")
        except Exception as e:
            print(f"❌ Erro saldo: {e}")
        
        # Buscar preço DIRETAMENTE  
        price = 0
        try:
            if api and API_CONNECTED:
                price = api.get_eth_price()
                print(f"📈 Preço ETH: ${price:.2f}")
        except Exception as e:
            print(f"❌ Erro preço: {e}")
        
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
        
        print(f"✅ Enviando: {response}")
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
        bot_state['last_error'] = None
        
        if not bot_state['thread'] or not bot_state['thread'].is_alive():
            bot_state['thread'] = threading.Thread(target=trading_loop, daemon=True)
            bot_state['thread'].start()
            print("🚀 Thread iniciada!")
        
        return jsonify({'success': True, 'message': 'Bot iniciado'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/pause', methods=['POST'])
def pause():
    bot_state['is_paused'] = True
    if trading_bot:
        trading_bot.stop()
    return jsonify({'success': True, 'message': 'Bot pausado'})

@app.route('/api/stop', methods=['POST'])
def stop():
    bot_state['is_running'] = False
    bot_state['is_paused'] = False
    if trading_bot:
        trading_bot.stop()
    return jsonify({'success': True, 'message': 'Bot parado'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Iniciando na porta {port}")
    print(f"🔗 API: {'✅ Conectada' if API_CONNECTED else '❌ Desconectada'}")
    app.run(host='0.0.0.0', port=port, debug=True)
