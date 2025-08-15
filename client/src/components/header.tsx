import { useSocket } from '@/hooks/useSocket';
import { useTradingStatus } from '@/hooks/useTradingData';

export default function Header() {
  const { isConnected } = useSocket();
  const { data: tradingStatus } = useTradingStatus();

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(amount);
  };

  const formatPnL = (pnl: number) => {
    const formatted = formatCurrency(Math.abs(pnl));
    return pnl >= 0 ? `+${formatted}` : `-${formatted}`;
  };

  return (
    <header className="bg-charcoal border-b border-dark-gray px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <h1 className="text-2xl font-bold text-warning">
            <i className="fas fa-coins mr-2"></i>
            Farme de Dinheiro
          </h1>
          <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
            isConnected 
              ? 'bg-profit text-dark-blue' 
              : 'bg-loss text-white'
          }`}>
            <i className={`fas fa-circle text-xs mr-1 ${
              isConnected ? 'animate-pulse' : ''
            }`}></i>
            {isConnected ? 'ATIVO' : 'DESCONECTADO'}
          </span>
        </div>
        
        <div className="flex items-center space-x-6">
          <div className="text-right">
            <p className="text-text-secondary text-sm">Saldo Total</p>
            <p className="font-mono font-semibold text-lg">
              {formatCurrency(12847.32)}
            </p>
          </div>
          <div className="text-right">
            <p className="text-text-secondary text-sm">PnL Hoje</p>
            <p className={`font-mono font-semibold text-lg ${
              (tradingStatus?.daily_pnl || 0) >= 0 ? 'text-profit' : 'text-loss'
            }`}>
              {formatPnL(tradingStatus?.daily_pnl || 234.56)}
            </p>
          </div>
          <button 
            className="bg-dark-gray hover:bg-opacity-80 px-4 py-2 rounded-lg transition-colors"
            data-testid="button-settings"
          >
            <i className="fas fa-cog"></i>
          </button>
        </div>
      </div>
    </header>
  );
}
