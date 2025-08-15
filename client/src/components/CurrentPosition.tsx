import { useSocket } from '@/hooks/useSocket';
import { useTradingStatus } from '@/hooks/useTradingData';
import { TradingPosition } from '@/types/trading';
import { useToast } from '@/hooks/use-toast';

export default function CurrentPosition() {
  const { data } = useSocket();
  const { data: tradingStatus } = useTradingStatus();
  const { closePosition } = useSocket();
  const { toast } = useToast();

  // Get position from socket data or fallback to query data
  const position: TradingPosition | null = 
    data.trading_status?.position || 
    tradingStatus?.current_position || 
    null;

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(amount);
  };

  const formatCrypto = (amount: number) => {
    return amount.toFixed(4);
  };

  const handleClosePosition = () => {
    if (!position) return;

    if (window.confirm('Tem certeza que deseja fechar a posição atual?')) {
      closePosition();
      toast({
        title: "Posição fechada",
        description: "Ordem de fechamento enviada com sucesso",
      });
    }
  };

  return (
    <div className="bg-charcoal rounded-lg border border-dark-gray p-6">
      <h3 className="text-lg font-semibold mb-4 flex items-center">
        <i className="fas fa-crosshairs mr-2 text-warning"></i>
        Posição Atual
      </h3>
      
      {position ? (
        <div className="space-y-4">
          <div className="text-center p-4 bg-dark-gray rounded-lg">
            <p className="text-text-secondary text-sm mb-1">{position.symbol}</p>
            <p className={`text-2xl font-bold font-mono ${
              position.side === 'long' ? 'text-profit' : 'text-loss'
            }`} data-testid="text-position-size">
              {position.side.toUpperCase()} {formatCrypto(position.size)}
            </p>
            <p className="text-sm text-text-secondary mt-1">Alavancagem: 10x</p>
          </div>
          
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-text-secondary">Preço Entrada:</span>
              <span className="font-mono" data-testid="text-entry-price">
                {formatCurrency(position.entry_price)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-secondary">PnL Não Real:</span>
              <span className={`font-mono ${
                position.unrealized_pnl >= 0 ? 'text-profit' : 'text-loss'
              }`} data-testid="text-unrealized-pnl">
                {position.unrealized_pnl >= 0 ? '+' : ''}{formatCurrency(position.unrealized_pnl)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-secondary">Stop Loss:</span>
              <span className="font-mono text-loss" data-testid="text-stop-loss">
                {formatCurrency(position.stop_loss)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-secondary">Take Profit:</span>
              <span className="font-mono text-profit" data-testid="text-take-profit">
                {formatCurrency(position.take_profit)}
              </span>
            </div>
          </div>

          <button 
            onClick={handleClosePosition}
            className="w-full mt-4 bg-loss hover:bg-red-600 text-white px-4 py-2 rounded-lg transition-colors font-semibold"
            data-testid="button-close-position"
          >
            <i className="fas fa-times mr-2"></i>
            Fechar Posição
          </button>
        </div>
      ) : (
        <div className="text-center p-8">
          <i className="fas fa-chart-line text-4xl text-text-secondary mb-4"></i>
          <p className="text-text-secondary">Nenhuma posição aberta</p>
          <p className="text-sm text-text-secondary mt-2">
            O bot abrirá posições automaticamente quando identificar oportunidades
          </p>
        </div>
      )}
    </div>
  );
}
