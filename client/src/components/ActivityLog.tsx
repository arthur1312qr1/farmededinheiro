import { useSocket } from '@/hooks/useSocket';
import { useActivityLogs } from '@/hooks/useTradingData';
import { useEffect, useState } from 'react';
import { ActivityLogEntry } from '@shared/schema';

export default function ActivityLog() {
  const { data: staticLogs } = useActivityLogs();
  const { data } = useSocket();
  const [logs, setLogs] = useState<ActivityLogEntry[]>([]);

  useEffect(() => {
    if (staticLogs) {
      setLogs(staticLogs);
    }
  }, [staticLogs]);

  // Handle real-time log updates
  useEffect(() => {
    if (data.new_log_entry) {
      setLogs(prev => [data.new_log_entry, ...prev.slice(0, 19)]); // Keep last 20 entries
    }
  }, [data.new_log_entry]);

  // Handle trading status updates for activity log
  useEffect(() => {
    if (data.trading_status?.activity_log) {
      setLogs(data.trading_status.activity_log);
    }
  }, [data.trading_status]);

  const getLogIcon = (type: string) => {
    switch (type) {
      case 'success':
        return 'w-2 h-2 bg-profit rounded-full mt-2';
      case 'error':
        return 'w-2 h-2 bg-loss rounded-full mt-2';
      case 'warning':
        return 'w-2 h-2 bg-warning rounded-full mt-2';
      default:
        return 'w-2 h-2 bg-blue-500 rounded-full mt-2';
    }
  };

  const getLogColor = (type: string) => {
    switch (type) {
      case 'success':
        return 'text-profit';
      case 'error':
        return 'text-loss';
      case 'warning':
        return 'text-warning';
      default:
        return 'text-blue-400';
    }
  };

  const formatTime = (timestamp: string | Date) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('pt-BR', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  return (
    <div className="lg:col-span-2 bg-charcoal rounded-lg border border-dark-gray">
      <div className="p-6 border-b border-dark-gray">
        <h2 className="text-xl font-semibold flex items-center">
          <i className="fas fa-list mr-3 text-warning"></i>
          Log de Atividades
          <span className="ml-auto bg-profit text-dark-blue px-2 py-1 rounded text-sm font-medium">
            LIVE
          </span>
        </h2>
      </div>
      
      <div className="p-6">
        <div className="space-y-4 max-h-96 overflow-y-auto" data-testid="activity-log-container">
          {logs.length === 0 ? (
            <div className="text-center text-text-secondary py-8">
              <i className="fas fa-clock text-4xl mb-4"></i>
              <p>Aguardando atividades do trading bot...</p>
            </div>
          ) : (
            logs.map((log, index) => (
              <div 
                key={`${log.timestamp}-${index}`}
                className="flex items-start space-x-4 p-4 bg-dark-gray rounded-lg"
                data-testid={`log-entry-${index}`}
              >
                <div className={getLogIcon(log.type)}></div>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <span className={`font-medium ${getLogColor(log.type)}`}>
                      {log.action}
                    </span>
                    <span className="text-text-secondary text-sm font-mono">
                      {formatTime(log.timestamp)}
                    </span>
                  </div>
                  <p className="text-sm text-text-secondary">{log.message}</p>
                  {log.details && (
                    <p className={`text-xs mt-1 ${getLogColor(log.type)}`}>
                      {log.details}
                    </p>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
