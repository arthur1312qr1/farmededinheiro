# Adicionar no final da classe BitgetAPI

def validate_order_params(self, symbol: str, side: str, size: float, **kwargs) -> Dict:
    """Valida parâmetros da ordem antes de enviar"""
    errors = []
    
    # Validar valor mínimo
    if size < 1.0:
        errors.append(f"Valor {size:.2f} USDT abaixo do mínimo 1 USDT")
    
    # Validar símbolo
    if not symbol:
        errors.append("Símbolo não informado")
    
    # Validar lado da operação
    if side not in ['buy', 'sell']:
        errors.append(f"Lado inválido: {side}")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def place_order(self, symbol: str, side: str, size: float, price: float = 0, leverage: int = 1) -> Optional[Dict]:
    """Place order with improved validation and error handling"""
    try:
        # Validar parâmetros
        validation = self.validate_order_params(symbol, side, size)
        if not validation['valid']:
            logger.error(f"❌ Validação falhou: {', '.join(validation['errors'])}")
            return {
                'success': False,
                'error': ', '.join(validation['errors'])
            }
        
        if not self.api_key:
            # Return mock result for demo
            logger.info(f"📝 MOCK ORDER: {side.upper()} {size:.2f} USDT of {symbol} at ${price:.2f}")
            return {
                'success': True,
                'order_id': f'mock_{int(time.time())}',
                'symbol': symbol,
                'side': side,
                'size': size,
                'price': price
            }
        
        endpoint = "/api/v2/mix/order/place-order"
        data = {
            'symbol': symbol,
            'productType': 'USDT-FUTURES',
            'marginMode': 'crossed',
            'marginCoin': 'USDT',
            'side': side,
            'orderType': 'market' if price == 0 else 'limit',
            'size': str(size),  # Valor em USDT
            'leverage': str(leverage)
        }
        
        if price > 0:
            data['price'] = str(price)
        
        response = self._make_request("POST", endpoint, data=data)
        
        if response and response.get('code') == '00000':
            order_data = response.get('data', {})
            logger.info(f"✅ Ordem executada: {order_data.get('orderId')}")
            return {
                'success': True,
                'order_id': order_data.get('orderId'),
                'symbol': symbol,
                'side': side,
                'size': size,
                'price': price
            }
        else:
            error_msg = response.get('msg', 'Erro desconhecido') if response else 'Falha na comunicação'
            logger.error(f"❌ Erro na API Bitget: {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }
            
    except Exception as e:
        logger.error(f"❌ Erro ao executar ordem: {e}")
        return {
            'success': False,
            'error': str(e)
        }
