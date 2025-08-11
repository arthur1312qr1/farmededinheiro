# Trading Bot - CORRIGIDO PARA RAILWAY

## 🚀 Deploy Imediato no Railway

1. Copie todos os arquivos desta pasta
2. Faça upload no seu repositório GitHub
3. Conecte no Railway
4. Deploy automático funcionará

## 📁 Arquivos Corrigidos

### Core Python Files
- `main.py` - Entry point otimizado
- `app.py` - Flask app com tratamento de erros
- `config.py` - Configuração Railway
- `trading_bot.py` - Bot de trading
- `bitget_api.py` - API Bitget
- `gemini_handler.py` - Handler Gemini AI

### Railway Configuration
- `railway.json` - Configuração Railway
- `Procfile` - Processo de deploy
- `nixpacks.toml` - Build system
- `requirements.txt` - Dependências Python
- `pyproject.toml` - Projeto Python

### Frontend
- `templates/` - HTML templates
- `static/` - CSS e assets

### Environment
- `.env.example` - Template variáveis

## ✅ Funcionalidades

- ✅ Paper trading (modo seguro)
- ✅ Dashboard web responsivo
- ✅ API Bitget integrada
- ✅ Gemini AI para análise
- ✅ Sistema de logs
- ✅ Health check endpoint

## 🔧 Variáveis Railway (Opcional)

Para trading real, configure no Railway:
```
PAPER_TRADING=false
BITGET_API_KEY=sua_chave
BITGET_API_SECRET=seu_secret
BITGET_PASSPHRASE=sua_passphrase
GEMINI_API_KEY=sua_chave_gemini
```

## 🎯 Deploy Railway

1. Push no GitHub
2. Conectar Railway
3. Deploy automático
4. Bot funcionando!
