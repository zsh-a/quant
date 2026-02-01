import logging
from typing import Dict, List, Optional
from .base import DataStream, Broker, Strategy, Bar, Order
from .risk_manager import RiskManager

logger = logging.getLogger(__name__)

class TradingEngine:
    def __init__(self, strategy: Strategy, broker: Broker, data_stream: DataStream, 
                 on_step=None, risk_manager: Optional[RiskManager] = None):
        self.strategy = strategy
        self.broker = broker
        self.data_stream = data_stream
        self.on_step = on_step
        self.risk_manager = risk_manager
        self.last_bars = None
        
        self.strategy.set_engine(self)
        self.running = False
        
        logger.info(f"TradingEngine initialized with risk_manager={risk_manager is not None}")

    def submit_order(self, order: Order):
        """Submit order with optional risk check"""
        # Risk check if risk manager is enabled
        if self.risk_manager and self.risk_manager.enabled:
            # Get current price from last bars
            current_price = 0.0
            if self.last_bars and order.symbol in self.last_bars:
                current_price = self.last_bars[order.symbol].close
            
            # Check position limit for buy orders
            if order.type == 'buy':
                allowed, reason = self.risk_manager.check_position_limit(
                    order.symbol, order.quantity, current_price
                )
                if not allowed:
                    logger.warning(f"Order rejected by risk manager: {reason}")
                    order.status = "REJECTED"
                    order.id = "RISK_REJECTED"
                    return order.id
        
        return self.broker.submit_order(order)

    def cancel_order(self, order_id: str):
        return self.broker.cancel_order(order_id)
    
    def get_risk_status(self) -> Optional[Dict]:
        """Get current risk status"""
        if self.risk_manager:
            return self.risk_manager.get_status()
        return None

    def run(self):
        self.running = True
        logger.info("Trading engine started.")
        
        while self.running:
            bars = self.data_stream.next_bar()
            if bars is None:
                logger.info("End of data stream.")
                break
            
            # 1. Update broker with new price data (fill orders)
            self.broker.step(bars)
            
            # 2. Auto-update strategy's current date for logging
            if hasattr(self.strategy, '_update_current_date'):
                self.strategy._update_current_date(bars)
            
            # 3. Strategy process bars
            self.strategy.on_bar(bars)
            
            # 3. Process immediate orders (generated in on_bar)
            self.broker.process_same_bar_orders(bars, 'IMMEDIATE_OPEN')
            self.broker.process_same_bar_orders(bars, 'IMMEDIATE_CLOSE')
            
            # 4. Trigger callback for progress tracking
            if self.on_step:
                self.on_step(bars)
            
            self.last_bars = bars

        # Do NOT run broker.step(last_bars) again here. Orders submitted on the last bar
        # (e.g. JSG rebalance on last trading day of month) are NEXT_OPEN and must fill
        # at the *next* bar's open. If the stream has ended, there is no next bar, so those
        # orders should remain unfilled; filling them at last bar's open would wrongly
        # show execution on the last day (e.g. 2026-01-30) instead of next Monday.

        self.running = False
        logger.info("Trading engine stopped.")

    def stop(self):
        self.running = False
