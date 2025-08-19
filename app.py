from flask import Flask
from flask_cors import CORS
import os
import logging
from datetime import datetime
import sys
import time

# Importar a API da Bitget
try:
    from bitget_api import BitgetAPI
    from config import get_config
    config = get_config()
    bitget_api = BitgetAPI(
        api_key=config.get('BITGET_API_KEY'),
        secret_key=config.get('BITGET_SECRET_KEY'),
        passphrase=config.get('BITGET_PASSPHRASE')
    )
    API_CONNECTED = True
    print("✅ BitgetAPI inicializada com sucesso")
except Exception as e:
    print(f"❌ Erro na API: {e}")
    bitget_api = None
    API_CONNECTED = False

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Estado simples
bot_state = {
    'is_running': False,
    'is_paused': False,
    'trades_today': 0,
    'total_profit': 0.0
}

@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Trading Bot</title>
    <meta charset="UTF-8">
</head>
<body style="font-family: Arial; background: #1a1a1a; color: white; padding: 20px;">
    <h1>🤖 Trading Bot Dashboard</h1>
    
    <div style="background: #333; padding: 20px; margin: 10px; border-radius: 8px;">
        <h2>Status: <span id="status">PARADO</span></h2>
        <button onclick="start()" style="background: #4CAF50; color: white; padding: 10px; border: none; margin: 5px;">Iniciar</button>
        <button onclick="pause()" style="background: #ff9800; color: white; padding: 10px; border: none; margin: 5px;">Pausar</button>
        <button onclick="stop()" style="background: #f44336; color: white; padding: 10px; border: none; margin: 5px;">Parar</button>
    </div>
    
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
        <div style="background: #333; padding: 20px; border-radius: 8px; text-align: center;">
            <div style="font-size: 2em; color: #2196F3;" id="trades">0</div>
            <div>Trades Hoje</div>
        </div>
        <div style="background: #333; padding: 20px; border-radius: 8px; text-align: center;">
            <div style="font-size: 2em; color: #9C27B0;" id="balance">$0.00</div>
            <div>Saldo USDT</div>
        </div>
        <div style="background: #333; padding: 20px; border-radius: 8px; text-align: center;">
            <div style="font-size: 2em; color: #FF5722;" id="price">$0.00</div>
            <div>Preço ETH</div>
        </div>
    </div>
    
    <div style="background: #333; padding: 20px; margin: 10px 0; border-radius: 8px;">
        <h3>Sistema</h3>
        <div id="info">Carregando...</div>
        <div id="debug" style="font-size: 0.8em; margin-top: 10px; color: #888;"></div>
    </div>
    
    <script>
        let requestCount = 0;
        
        function update() {
            requestCount++;
            console.log('Fazendo requisição #' + requestCount);
            
            fetch('/api/data')
                .then(r => r.json())
                .then(data => {
                    console.log('Dados recebidos:', data);
                    
                    document.getElementById('trades').textContent = data.trades || 0;
                    document.getElementById('balance').textContent = '$' + (data.balance || 0).toFixed(4);
                    document.getElementById('price').textContent = '$' + (data.price || 0).toFixed(2);
                    document.getElementById('status').textContent = data.status || 'PARADO';
                    document.getElementById('info').textContent = data.info || 'Conectado';
                    document.getElementById('debug').textContent = 'Req #' + requestCount + ' - ' + new Date().toLocaleTimeString();
                })
                .catch(e => {
                    console.error('Erro na requisição:', e);
                    document.getElementById('info').textContent = 'Erro: ' + e.message;
                    document.getElementById('debug').textContent = 'ERRO Req #' + requestCount + ' - ' + e.message;
                });
        }
        
        function start() {
            fetch('/api/start', {method: 'POST'}).then(() => update());
        }
        function pause() {
            fetch('/api/pause', {method: 'POST'}).then(() => update());
        }
        function stop() {
            fetch('/api/stop', {method: 'POST'}).then(() => update());
        }
        
        update();
        setInterval(update, 5000);
    </script>
</body>
</html>
'''

@app.route('/api/data')
def get_data():
    """Endpoint único para todos os dados"""
    try:
        print("📊 Iniciando coleta de dados...")
        
        # Pegar saldo
        balance = 0
        if bitget_api and API_CONNECTED:
            try:
                print("💰 Buscando saldo...")
                balance_info = bitget_api.get_balance()
                print(f"💰 Balance info: {balance_info}")
                if balance_info and isinstance(balance_info, dict):
                    balance = balance_info.get('free', 0)
                    print(f"💰 Saldo encontrado: ${balance}")
                else:
                    print("⚠️ Balance info vazio ou inválido")
            except Exception as e:
                print(f"❌ Erro ao buscar saldo: {e}")
                balance = 0
        else:
            print("⚠️ API não conectada, usando saldo mock")
        
        # Pegar preço
        price = 0
        if bitget_api and API_CONNECTED:
            try:
                print("📈 Buscando preço ETH...")
                price = bitget_api.get_eth_price()
                print(f"📈 Preço encontrado: ${price}")
                if not price:
                    print("⚠️ Preço retornado é None")
                    price = 0
            except Exception as e:
                print(f"❌ Erro ao buscar preço: {e}")
                price = 0
        else:
            print("⚠️ API não conectada, usando preço mock")
        
        # Status do bot
        if bot_state['is_running'] and not bot_state['is_paused']:
            status = 'RODANDO'
        elif bot_state['is_paused']:
            status = 'PAUSADO'
        else:
            status = 'PARADO'
        
        response_data = {
            'trades': bot_state['trades_today'],
            'balance': balance,
            'price': price,
            'status': status,
            'info': f'API: {API_CONNECTED}, Saldo: ${balance:.4f}, Preço: ${price:.2f}'
        }
        
        print(f"✅ Dados finais: {response_data}")
        return response_data
        
    except Exception as e:
        error_msg = f"Erro em /api/data: {e}"
        print(f"❌ {error_msg}")
        logger.error(error_msg)
        return {
            'trades': 0,
            'balance': 0,
            'price': 0,
            'status': 'ERRO',
            'info': f'Erro: {str(e)}'
        }

@app.route('/api/start', methods=['POST'])
def start():
    bot_state['is_running'] = True
    bot_state['is_paused'] = False
    logger.info("🚀 Bot iniciado")
    return {'success': True}

@app.route('/api/pause', methods=['POST'])
def pause():
    bot_state['is_paused'] = True
    logger.info("⏸️ Bot pausado")
    return {'success': True}

@app.route('/api/stop', methods=['POST'])
def stop():
    bot_state['is_running'] = False
    bot_state['is_paused'] = False
    logger.info("🛑 Bot parado")
    return {'success': True}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Iniciando servidor na porta {port}")
    print(f"🔗 API Conectada: {API_CONNECTED}")
    app.run(host='0.0.0.0', port=port, debug=True)
