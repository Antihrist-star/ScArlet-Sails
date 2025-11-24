# ai_modules/live_dashboard.py
"""
Интегрируется с BacktestEngine для real-time мониторинга
"""

from comet_ml import Experiment
import asyncio
import websocket
import json

class LiveTradingDashboard:
    """
    Real-time мониторинг через CometML + WebSocket
    """
    
    def __init__(self, backtest_engine):
        self.backtest = backtest_engine
        self.experiment = Experiment(
            project_name="scarlet-sails-live",
            auto_histogram_weight_logging=True
        )
        
        # WebSocket для Binance данных
        self.ws_url = "wss://stream.binance.com:9443/ws/btcusdt@trade"
        
    async def start_monitoring(self):
        """Запуск real-time мониторинга"""
        
        # Parallel tasks
        await asyncio.gather(
            self._monitor_trades(),
            self._monitor_market(),
            self._monitor_performance()
        )
    
    async def _monitor_trades(self):
        """Мониторинг сделок из BacktestEngine"""
        while True:
            if hasattr(self.backtest, 'last_trade'):
                trade = self.backtest.last_trade
                
                # Log to CometML
                self.experiment.log_metrics({
                    "trade_entry": trade['entry_price'],
                    "trade_exit": trade['exit_price'],
                    "trade_pnl": trade['pnl'],
                    "trade_duration": trade['duration_bars']
                })
                
                # Alert on big losses
                if trade['pnl'] < -500:
                    self._send_alert(f"⚠️ Large loss: ${trade['pnl']:.2f}")
            
            await asyncio.sleep(1)
    
    async def _monitor_market(self):
        """Мониторинг рыночных данных"""
        ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=self._on_market_update
        )
        ws.run_forever()
    
    def _on_market_update(self, ws, message):
        """Обработка market data"""
        data = json.loads(message)
        price = float(data['p'])
        
        # Check for volatility spikes
        if hasattr(self, 'last_price'):
            change = abs(price - self.last_price) / self.last_price
            if change > 0.02:  # 2% spike
                self._send_alert(f"🔥 Volatility spike: {change*100:.1f}%")
        
        self.last_price = price
        
    async def _monitor_performance(self):
        """Мониторинг метрик производительности"""
        while True:
            metrics = self.backtest.calculate_metrics()
            
            # Log performance
            self.experiment.log_metrics({
                "profit_factor": metrics.get('profit_factor', 0),
                "win_rate": metrics.get('win_rate', 0),
                "sharpe_ratio": metrics.get('sharpe_ratio', 0),
                "max_drawdown": metrics.get('max_drawdown', 0),
                "total_return": metrics.get('total_return', 0)
            })
            
            # Проверка критических метрик
            if metrics.get('max_drawdown', 0) > 15:
                self._send_alert("🚨 CRITICAL: Drawdown > 15%")
                self.backtest.emergency_stop()
            
            await asyncio.sleep(60)  # Every minute