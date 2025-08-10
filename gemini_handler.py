import os
import json
import logging
import time
import random
from typing import Optional, Dict, Any

# Importação do Google Gemini
try:
    import google.generativeai as genai
    from google.api_core.exceptions import (
        ResourceExhausted,
        ServiceUnavailable, 
        InternalServerError,
        InvalidArgument
    )
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

logger = logging.getLogger(__name__)

class GeminiErrorHandler:
    """Manipula análise e correção de erros usando Google Gemini AI"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        Inicializa o manipulador de erros Gemini
        
        Args:
            api_key: Chave da API do Google Gemini
            model_name: Modelo Gemini a ser usado
        """
        self.api_key = api_key
        self.model_name = model_name
        self.model = None
        
        if api_key and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(model_name)
                logger.info(f"Cliente Gemini inicializado com modelo: {model_name}")
            except Exception as e:
                logger.error(f"Falha ao inicializar cliente Gemini: {e}")
                self.model = None
        else:
            if not GEMINI_AVAILABLE:
                logger.warning("Biblioteca google-generativeai não encontrada - análise de erro desabilitada")
            else:
                logger.warning("Nenhuma chave API Gemini fornecida - análise de erro desabilitada")
    
    def analyze_and_fix_error(self, error_message: str, context: Optional[Dict] = None) -> str:
        """
        Analisa erro e fornece sugestões de correção
        
        Args:
            error_message: A mensagem de erro para analisar
            context: Contexto adicional sobre o erro
            
        Returns:
            Sugestão de correção gerada pela IA
        """
        if not self.model:
            return "Gemini AI não disponível - nenhuma chave API fornecida"
        
        try:
            # Constrói prompt com contexto
            prompt = self._build_error_analysis_prompt(error_message, context)
            
            # Obtém análise da IA com lógica de retry
            response = self._generate_with_retry(prompt)
            
            if response:
                logger.info(f"Análise Gemini concluída para erro: {error_message[:100]}...")
                return response
            else:
                return "Incapaz de analisar erro - resposta Gemini vazia"
                
        except Exception as e:
            logger.error(f"Erro na análise Gemini: {e}")
            return f"Análise Gemini falhou: {str(e)}"
    
    def _build_error_analysis_prompt(self, error_message: str, context: Optional[Dict] = None) -> str:
        """Build comprehensive prompt for error analysis"""
        
        prompt = f"""
Você é um desenvolvedor Python especialista e especialista em bots de trading de criptomoedas.
Analise o seguinte erro e forneça soluções específicas e práticas.

MENSAGEM DE ERRO:
{error_message}

CONTEXTO:
- Aplicação: Bot de trading de criptomoedas usando API Bitget
- Framework: Aplicação web Flask
- Trading: Futuros USDT-M (símbolo ETHUSDT)
- Deploy: Plataforma Render.com

"""
        
        if context:
            prompt += f"\nADDITIONAL CONTEXT:\n{json.dumps(context, indent=2)}\n"
        
        prompt += """
Por favor forneça:

1. CAUSA RAIZ: Qual é a provável causa deste erro?

2. CORREÇÃO IMEDIATA: Solução passo a passo para resolver este erro específico

3. PREVENÇÃO: Como prevenir este erro no futuro

4. ALTERAÇÕES DE CÓDIGO: Se aplicável, forneça modificações específicas no código

5. MONITORAMENTO: O que deve ser monitorado para detectar problemas similares

6. SEVERIDADE: Classifique a severidade (BAIXA/MÉDIA/ALTA/CRÍTICA)

7. AÇÃO: O bot deve CONTINUAR, RESTART_REQUIRED ou STOP_TRADING?

Formate sua resposta como uma análise estruturada que pode guiar a recuperação automática de erros.
Foque em soluções práticas e implementáveis para um ambiente de trading em produção.
"""
        
        return prompt
    
    def _generate_with_retry(
        self, 
        prompt: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0
    ) -> Optional[str]:
        """
        Generate content with comprehensive retry logic
        
        Args:
            prompt: The prompt to send to Gemini
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries
            max_delay: Maximum delay between retries
            
        Returns:
            Generated content or None if failed
        """
        
        for attempt in range(max_retries + 1):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        'temperature': 0.3,  # Temperatura baixa para respostas mais consistentes
                        'max_output_tokens': 2048
                    }
                )
                
                if response.candidates:
                    candidate = response.candidates[0]
                    
                    # Verificar bloqueios de segurança
                    if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                        if candidate.finish_reason.name == "SAFETY":
                            logger.warning("Conteúdo bloqueado por configurações de segurança")
                            if attempt == 0:
                                neutral_prompt = f"Analise este erro técnico: {prompt[:200]}..."
                                return self._generate_with_retry(neutral_prompt, max_retries-1)
                            return "Conteúdo bloqueado por filtros de segurança"
                        
                        elif candidate.finish_reason.name == "RECITATION":
                            logger.warning("Conteúdo bloqueado por recitação")
                            return "Conteúdo bloqueado por recitação"
                
                return response.text
                
            except ResourceExhausted as e:
                # Limitação de taxa (429)
                if attempt < max_retries:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    delay += random.uniform(0, 1)  # Adicionar jitter
                    logger.warning(f"Limite de taxa atingido. Tentando novamente em {delay:.2f}s... (tentativa {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    logger.error("Limite de taxa excedido. Máximo de tentativas atingido.")
                    return "Limite de taxa excedido - incapaz de analisar erro"
            
            except Exception as gemini_exception:
                # Capturar todas as exceções do Gemini
                if attempt < max_retries:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    delay += random.uniform(0, 1)
                    logger.warning(f"Erro Gemini. Tentando novamente em {delay:.2f}s... (tentativa {attempt + 1}/{max_retries}): {gemini_exception}")
                    time.sleep(delay)
                else:
                    logger.error(f"Erro Gemini após {max_retries} tentativas: {gemini_exception}")
                    return f"Análise Gemini falhou: {str(gemini_exception)}"
        
        return None
    
    def analyze_trading_performance(self, trading_data: Dict) -> str:
        """
        Analisa performance de trading e sugere melhorias
        
        Args:
            trading_data: Dicionário contendo estatísticas e dados de performance
            
        Returns:
            Análise da IA sobre performance de trading com recomendações
        """
        if not self.model:
            return "Gemini AI não disponível para análise de performance"
        
        try:
            prompt = f"""
Analyze the following cryptocurrency trading bot performance data and provide optimization recommendations:

TRADING DATA:
{json.dumps(trading_data, indent=2, default=str)}

Please provide:

1. PERFORMANCE SUMMARY: Overall assessment of trading performance
2. STRENGTHS: What is working well in the current strategy
3. WEAKNESSES: Areas that need improvement
4. OPTIMIZATION SUGGESTIONS: Specific parameter adjustments
5. RISK ASSESSMENT: Current risk level and recommendations
6. MARKET CONDITIONS: How well the bot is adapting to current market
7. ACTION ITEMS: Priority improvements to implement

Focus on actionable insights for a Futures USDT-M trading bot on Bitget.
"""
            
            response = self._generate_with_retry(prompt)
            return response or "Unable to analyze trading performance"
            
        except Exception as e:
            logger.error(f"Error in performance analysis: {e}")
            return f"Performance analysis failed: {str(e)}"
    
    def suggest_market_strategy(self, market_data: Dict) -> str:
        """
        Suggest trading strategy based on current market conditions
        
        Args:
            market_data: Current market data and indicators
            
        Returns:
            AI-suggested trading strategy
        """
        if not self.client:
            return "Gemini AI not available for strategy suggestions"
        
        try:
            prompt = f"""
Based on the current market data, suggest an optimal trading strategy for a cryptocurrency futures bot:

MARKET DATA:
{json.dumps(market_data, indent=2, default=str)}

Provide strategy recommendations for:

1. MARKET SENTIMENT: Current market conditions assessment
2. ENTRY SIGNALS: When and how to enter positions
3. EXIT STRATEGY: Take profit and stop loss recommendations
4. LEVERAGE: Recommended leverage levels for current conditions
5. RISK MANAGEMENT: Position sizing and risk controls
6. TIMEFRAME: Optimal trading timeframes
7. INDICATORS: Most relevant technical indicators to monitor

Focus on ETHUSDT futures trading with practical, implementable strategies.
"""
            
            response = self._generate_with_retry(prompt)
            return response or "Unable to generate strategy suggestions"
            
        except Exception as e:
            logger.error(f"Error in strategy analysis: {e}")
            return f"Strategy analysis failed: {str(e)}"
