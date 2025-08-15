import { useEffect, useRef, useState } from 'react';
import { io, Socket } from 'socket.io-client';

interface SocketData {
  trading_status?: any;
  bot_status?: any;
  price_update?: any;
  analysis_update?: any;
  new_log_entry?: any;
  connection_status?: any;
}

export function useSocket() {
  const [isConnected, setIsConnected] = useState(false);
  const [data, setData] = useState<SocketData>({});
  const socketRef = useRef<Socket | null>(null);

  useEffect(() => {
    // Initialize socket connection
    const socket = io({
      autoConnect: true,
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });

    socketRef.current = socket;

    // Connection event handlers
    socket.on('connect', () => {
      console.log('🔗 Connected to trading server');
      setIsConnected(true);
    });

    socket.on('disconnect', () => {
      console.log('🔌 Disconnected from trading server');
      setIsConnected(false);
    });

    socket.on('connect_error', (error) => {
      console.error('❌ Connection error:', error);
      setIsConnected(false);
    });

    // Trading data event handlers
    socket.on('trading_status_update', (statusData) => {
      setData(prev => ({
        ...prev,
        trading_status: statusData
      }));
    });

    socket.on('bot_status_update', (botData) => {
      setData(prev => ({
        ...prev,
        bot_status: botData
      }));
    });

    socket.on('price_update', (priceData) => {
      setData(prev => ({
        ...prev,
        price_update: priceData
      }));
    });

    socket.on('analysis_update', (analysisData) => {
      setData(prev => ({
        ...prev,
        analysis_update: analysisData
      }));
    });

    socket.on('new_log_entry', (logEntry) => {
      setData(prev => ({
        ...prev,
        new_log_entry: logEntry
      }));
    });

    socket.on('connection_status', (status) => {
      setData(prev => ({
        ...prev,
        connection_status: status
      }));
    });

    // Cleanup on unmount
    return () => {
      socket.disconnect();
    };
  }, []);

  // Socket actions
  const toggleBot = () => {
    socketRef.current?.emit('toggle_bot');
  };

  const emergencyStop = () => {
    socketRef.current?.emit('emergency_stop');
  };

  const closePosition = () => {
    socketRef.current?.emit('close_position');
  };

  const getTradingStatus = () => {
    socketRef.current?.emit('get_trading_status');
  };

  const getBotStatus = () => {
    socketRef.current?.emit('get_bot_status');
  };

  return {
    isConnected,
    data,
    toggleBot,
    emergencyStop,
    closePosition,
    getTradingStatus,
    getBotStatus,
  };
}
