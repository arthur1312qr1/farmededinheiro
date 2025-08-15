import { useSocket } from '@/hooks/useSocket';
import { useBotStatus } from '@/hooks/useTradingData';
import { useToast } from '@/hooks/use-toast';

export default function BotControls() {
  const { data, toggleBot, emergencyStop } = useSocket();
  const { data: staticBotStatus } = useBotStatus();
  const { toast } = useToast();

  // Get bot status from socket data or fallback to query data
  const botStatus = data.bot_status || staticBotStatus;

  const handleToggleBot = () => {
    const action = botStatus?.is_running 
      ? (botStatus.is_paused ? 'retomar' : 'pausar')
      : 'iniciar';
    
    if (window.confirm(`Tem certeza que deseja ${action} o bot?`)) {
      toggleBot();
      toast({
        title: "Bot controlado",
        description: `Bot ${action}do com sucesso`,
      });
    }
  };

  const handleEmergencyStop = () => {
    if (window.confirm('PARADA DE EMERGÊNCIA! Isso fechará todas as posições imediatamente. Confirma?')) {
      emergencyStop();
      toast({
        title: "Parada de emergência",
        description: "Todas as posições foram fechadas",
        variant: "destructive",
      });
    }
  };

  const getStatusColor = () => {
    if (!botStatus?.is_running) return 'text-text-secondary';
    if (botStatus.is_paused) return 'text-warning';
    return 'text-profit';
  };

  const getStatusText = () => {
    if (!botStatus?.is_running) return 'PARADO';
    if (botStatus.is_paused) return 'PAUSADO';
    return 'ATIVO';
  };

  const getButtonText = () => {
    if (!botStatus?.is_running) return 'Iniciar Bot';
    if (botStatus.is_paused) return 'Retomar Bot';
    return 'Pausar Bot';
  };

  const getButtonColor = () => {
    if (!botStatus?.is_running) return 'bg-profit hover:bg-green-600';
    if (botStatus.is_paused) return 'bg-profit hover:bg-green-600';
    return 'bg-warning hover:bg-yellow-600';
  };

  return (
    <div className="bg-charcoal rounded-lg border border-dark-gray p-6">
      <h3 className="text-lg font-semibold mb-4 flex items-center">
        <i className="fas fa-robot mr-2 text-warning"></i>
        Controle do Bot
      </h3>
      
      <div className="space-y-4">
        {/* Bot Status */}
        <div className="flex items-center justify-between p-3 bg-dark-gray rounded-lg">
          <span className="font-medium">Status do Bot</span>
          <div className="flex items-center space-x-2">
            <span className={`font-semibold ${getStatusColor()}`} data-testid="text-bot-status">
              {getStatusText()}
            </span>
            <div className={`w-3 h-3 rounded-full ${
              botStatus?.is_running && !botStatus?.is_paused 
                ? 'bg-profit animate-pulse' 
                : 'bg-text-secondary'
            }`}></div>
          </div>
        </div>
        
        {/* Toggle Bot Button */}
        <button 
          onClick={handleToggleBot}
          className={`w-full ${getButtonColor()} text-white px-4 py-3 rounded-lg transition-colors font-semibold`}
          data-testid="button-toggle-bot"
        >
          <i className={`fas ${
            botStatus?.is_running && !botStatus?.is_paused 
              ? 'fa-pause' 
              : 'fa-play'
          } mr-2`}></i>
          {getButtonText()}
        </button>
        
        {/* Emergency Stop Button */}
        <button 
          onClick={handleEmergencyStop}
          className="w-full bg-dark-gray hover:bg-opacity-80 text-text-primary px-4 py-3 rounded-lg transition-colors"
          data-testid="button-emergency-stop"
        >
          <i className="fas fa-exclamation-triangle mr-2"></i>
          Parada de Emergência
        </button>
      </div>
    </div>
  );
}
