import Header from './Header';
import Sidebar from './Sidebar';
import MainContent from './MainContent';
import { useSocket } from '@/hooks/useSocket';
import { useEffect } from 'react';
import { useToast } from '@/hooks/use-toast';

export default function TradingDashboard() {
  const { isConnected, data } = useSocket();
  const { toast } = useToast();

  // Handle socket event responses
  useEffect(() => {
    if (data.bot_action_result) {
      toast({
        title: data.bot_action_result.success ? "Sucesso" : "Erro",
        description: data.bot_action_result.message,
        variant: data.bot_action_result.success ? "default" : "destructive",
      });
    }
  }, [data.bot_action_result, toast]);

  useEffect(() => {
    if (data.position_action_result) {
      toast({
        title: data.position_action_result.success ? "Sucesso" : "Erro",
        description: data.position_action_result.message,
        variant: data.position_action_result.success ? "default" : "destructive",
      });
    }
  }, [data.position_action_result, toast]);

  return (
    <div className="bg-dark-blue text-text-primary font-inter min-h-screen">
      <Header />
      
      <div className="flex h-screen">
        <Sidebar />
        <MainContent />
      </div>

      {/* Real-time Connection Status Toast */}
      {isConnected && (
        <div className="fixed bottom-4 right-4 bg-charcoal border border-dark-gray rounded-lg p-4 shadow-lg">
          <div className="flex items-center space-x-3">
            <div className="w-3 h-3 bg-profit rounded-full animate-pulse"></div>
            <span className="text-sm font-medium">Conectado ao servidor</span>
          </div>
        </div>
      )}
    </div>
  );
}
