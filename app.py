import os
import sys
import traceback
import logging
from datetime import datetime

try:
    from flask import Flask, render_template, request, jsonify
    from flask_socketio import SocketIO, emit
    from flask_cors import CORS
    print("✅ Todas as importações do Flask carregadas com sucesso")
except ImportError as e:
    print(f"❌ Erro ao importar Flask: {e}")
    sys.exit(1)

# Configuração de logging mais robusta
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def create_app():
    """Cria e configura a aplicação Flask"""
    try:
        logger.info("🚀 Iniciando criação da aplicação Flask...")
        app = Flask(__name__)
        
        # Configurações básicas
        app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-muito-seguro-2024')
        app.config['DEBUG'] = False  # Força debug=False para produção
        
        logger.info("✅ Configurações básicas definidas")
        
        # Configurar CORS
        CORS(app, origins="*")
        logger.info("✅ CORS configurado")
        
        # Configurar SocketIO de forma mais segura
        try:
            socketio = SocketIO(
                app, 
                cors_allowed_origins="*", 
                async_mode='threading',
                logger=False,  # Desabilita logs verbosos do SocketIO
                engineio_logger=False
            )
            logger.info("✅ SocketIO configurado com sucesso")
        except Exception as e:
            logger.error(f"❌ Erro ao configurar SocketIO: {e}")
            # Se SocketIO falhar, continua sem ele
            socketio = None
        
        # Rota principal mais simples
        @app.route('/')
        def index():
            try:
                logger.info("📄 Rota principal acessada")
                return """
                <!DOCTYPE html>
                <html lang="pt-BR">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>🚀 Trading Bot - Farmede Dinheiro</title>
                    <style>
                        body { 
                            font-family: Arial, sans-serif; 
                            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                            color: white; 
                            margin: 0; 
                            padding: 40px;
                            min-height: 100vh;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                        }
                        .container { 
                            max-width: 800px; 
                            text-align: center; 
                            background: rgba(255,255,255,0.1);
                            padding: 40px;
                            border-radius: 15px;
                            backdrop-filter: blur(10px);
                        }
                        .title { 
                            font-size: 3em; 
                            margin-bottom: 20px; 
                            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
                        }
                        .status { 
                            background: rgba(76,175,80,0.2); 
                            padding: 20px; 
                            border-radius: 10px; 
                            margin: 20px 0;
                            border: 1px solid #4CAF50;
                        }
                        .success { 
                            color: #4CAF50; 
                            font-weight: bold; 
                            font-size: 1.5em;
                        }
                        .feature { 
                            margin: 10px 0; 
                            padding: 10px;
                            background: rgba(255,255,255,0.1);
                            border-radius: 5px;
                        }
                        .pulse { animation: pulse 2s infinite; }
                        @keyframes pulse { 
                            0% { opacity: 1; } 
                            50% { opacity: 0.7; } 
                            100% { opacity: 1; } 
                        }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1 class="title">🚀 Trading Bot</h1>
                        <h2 style="opacity: 0.9; margin-bottom: 30px;">Farmede Dinheiro</h2>
                        
                        <div class="status">
                            <div class="success pulse">✅ SISTEMA ONLINE!</div>
                            <p>Bot de trading funcionando perfeitamente</p>
                        </div>
                        
                        <div class="status">
                            <h3>🎯 Funcionalidades Ativas:</h3>
                            <div class="feature">📊 API REST funcionando</div>
                            <div class="feature">💱 Integração CCXT disponível</div>
                            <div class="feature">🔄 WebSocket preparado</div>
                            <div class="feature">🛡️ CORS configurado</div>
                        </div>
                        
                        <div class="status">
                            <h3>📋 API Endpoints:</h3>
                            <div class="feature">GET /api/status - Status do sistema</div>
                            <div class="feature">GET /api/test - Teste da API</div>
                            <div class="feature">GET /api/ccxt - Info das exchanges</div>
                        </div>
                        
                        <div style="margin-top: 30px; opacity: 0.8;">
                            <p>🕒 Sistema iniciado: """ + datetime.now().strftime('%d/%m/%Y %H:%M:%S') + """</p>
                        </div>
                    </div>
                </body>
                </html>
                """
            except Exception as e:
                logger.error(f"❌ Erro na rota principal: {e}")
                return f"<h1>Erro: {str(e)}</h1>", 500
        
        # API de status
        @app.route('/api/status')
        def status():
            try:
                return jsonify({
                    'status': 'online',
                    'timestamp': datetime.now().isoformat(),
                    'message': 'Trading Bot funcionando perfeitamente!',
                    'version': '1.0.0',
                    'features': {
                        'flask': 'ativo',
                        'ccxt': 'disponível',
                        'cors': 'configurado',
                        'socketio': 'ativo' if socketio else 'indisponível'
                    }
                })
            except Exception as e:
                logger.error(f"❌ Erro na API status: {e}")
                return jsonify({'error': str(e)}), 500
        
        # API de teste
        @app.route('/api/test')
        def test():
            try:
                return jsonify({
                    'success': True,
                    'message': 'API funcionando 100%!',
                    'timestamp': datetime.now().isoformat(),
                    'data': {
                        'server': 'online',
                        'database': 'não configurado',
                        'exchanges': 'disponível via CCXT'
                    }
                })
            except Exception as e:
                logger.error(f"❌ Erro na API test: {e}")
                return jsonify({'error': str(e)}), 500
        
        # API para listar exchanges disponíveis
        @app.route('/api/ccxt')
        def ccxt_info():
            try:
                import ccxt
                exchanges = list(ccxt.exchanges)[:10]  # Primeiras 10 exchanges
                return jsonify({
                    'total_exchanges': len(ccxt.exchanges),
                    'sample_exchanges': exchanges,
                    'message': 'CCXT funcionando - exchanges disponíveis'
                })
            except Exception as e:
                logger.error(f"❌ Erro na API CCXT: {e}")
                return jsonify({'error': str(e)}), 500
        
        # Configurar eventos do SocketIO se disponível
        if socketio:
            @socketio.on('connect')
            def handle_connect():
                logger.info('👤 Cliente conectado via WebSocket')
                emit('response', {
                    'data': 'Conectado ao Trading Bot!', 
                    'timestamp': datetime.now().isoformat()
                })
            
            @socketio.on('disconnect')
            def handle_disconnect():
                logger.info('👤 Cliente desconectado')
            
            @socketio.on('ping')
            def handle_ping():
                emit('pong', {'timestamp': datetime.now().isoformat()})
        
        # Error handlers
        @app.errorhandler(404)
        def not_found(error):
            return jsonify({'error': 'Endpoint não encontrado'}), 404
        
        @app.errorhandler(500)
        def internal_error(error):
            logger.error(f"❌ Erro interno: {error}")
            return jsonify({'error': 'Erro interno do servidor'}), 500
        
        # Anexa socketio ao app se disponível
        if socketio:
            app.socketio = socketio
        
        logger.info("✅ Aplicação Flask criada com sucesso!")
        return app
        
    except Exception as e:
        logger.error(f"❌ Erro fatal ao criar aplicação: {e}")
        logger.error(traceback.format_exc())
        raise

def main():
    try:
        logger.info("🚀 === INICIANDO TRADING BOT ===")
        
        # Criar aplicação
        app = create_app()
        
        # Obter porta
        port = int(os.environ.get('PORT', 5000))
        logger.info(f"🌐 Porta configurada: {port}")
        
        # Verificar se SocketIO está disponível
        if hasattr(app, 'socketio') and app.socketio:
            logger.info("🔄 Iniciando com SocketIO...")
            app.socketio.run(
                app,
                host='0.0.0.0',
                port=port,
                debug=False,
                use_reloader=False
            )
        else:
            logger.info("🌐 Iniciando sem SocketIO...")
            app.run(
                host='0.0.0.0',
                port=port,
                debug=False,
                use_reloader=False
            )
            
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}")
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("🛑 Aplicação interrompida pelo usuário")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Erro não tratado: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
