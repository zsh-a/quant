import logging
from typing import Dict, List, Optional
from .base import DataStream, Broker, Strategy, Bar, Order

logger = logging.getLogger(__name__)

class TradingEngine:
    def __init__(self, strategy: Strategy, broker: Broker, data_stream: DataStream, on_step=None):
        self.strategy = strategy
        self.broker = broker
        self.data_stream = data_stream
        self.on_step = on_step
        self.last_bars = None
        
        self.strategy.set_engine(self)
        self.running = False

    def submit_order(self, order: Order):
        return self.broker.submit_order(order)

    def cancel_order(self, order_id: str):
        return self.broker.cancel_order(order_id)

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
            
            # 2. Strategy process bars
            self.strategy.on_bar(bars)
            
            # 3. Trigger callback for progress tracking
            if self.on_step:
                self.on_step(bars)
            
            self.last_bars = bars
            
        # Final settlement: if there are pending orders and we have last known bars
        # we can attempt to fill them at the last close for accuracy in backtest metrics
        if self.last_bars:
            logger.info("Final settlement: Processing remaining orders at last available close.")
            self.broker.step(self.last_bars) # One last step to match orders from last bar
            if self.on_step:
                self.on_step(self.last_bars)

        self.running = False
        logger.info("Trading engine stopped.")

    def stop(self):
        self.running = False
