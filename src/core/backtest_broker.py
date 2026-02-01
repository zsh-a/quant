from typing import Dict, Any, List, Optional
from .base import Broker, Order, Bar
from datetime import datetime
from loguru import logger
import uuid

class BacktestBroker(Broker):
    def __init__(self, initial_cash: float = 1000000.0, commission: float = 0.0001, 
                 db_client=None, risk_manager=None):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.commission = commission
        self.db_client = db_client
        self.risk_manager = risk_manager
        
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.orders: Dict[str, Order] = {}
        self.history: List[Order] = []
        self.current_bars: Dict[str, Bar] = {}
        self.equity_history: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        
        # New: Track latest known prices for every symbol as fallback
        self.last_prices: Dict[str, float] = {} 
        
        # New: Track cost basis per position
        self.position_costs: Dict[str, float] = {} # symbol -> avg_price
        
        # Optimization: Track last equity for PnL calc even if history is cleared
        self._last_equity = initial_cash
        
        # Stock name mapping
        self.stock_names: Dict[str, str] = {}
        self._load_stock_names()
        
        # Initialize risk manager if provided
        if self.risk_manager:
            self.risk_manager.current_capital = initial_cash
            self.risk_manager.daily_start_capital = initial_cash
            logger.info(f"BacktestBroker initialized with risk manager")

    def _load_stock_names(self):
        import os
        import pandas as pd
        if os.path.exists("all_stock.csv"):
            try:
                df = pd.read_csv("all_stock.csv")
                self.stock_names = dict(zip(df['code'], df['code_name']))
            except Exception as e:
                logger.error(f"Failed to load stock names: {e}")

    def submit_order(self, order: Order) -> str:
        order.id = str(uuid.uuid4())
        order.status = "SUBMITTED"
        self.orders[order.id] = order
        return order.id

    def cancel_order(self, order_id: str):
        if order_id in self.orders:
            self.orders[order_id].status = "CANCELLED"
            self.history.append(self.orders.pop(order_id))

    def get_account_info(self) -> Dict[str, Any]:
        detailed_positions = {}
        for symbol, qty in self.positions.items():
            price = 0.0
            if symbol in self.current_bars:
                price = self.current_bars[symbol].close
            elif symbol in self.last_prices:
                price = self.last_prices[symbol]
            elif self.db_client and self.current_bars:
                # Attempt fallback fetch
                try:
                    ts = next(iter(self.current_bars.values())).timestamp
                    df = self.db_client.get_price(symbol, str(ts.date()), ["close"], 1)
                    if not df.empty:
                        price = df.iloc[0]["close"]
                        self.last_prices[symbol] = price
                except Exception:
                    pass
            
            avg_cost = self.position_costs.get(symbol, 0.0)
            market_value = qty * price
            cost_value = qty * avg_cost
            unrealized_pnl = market_value - cost_value
            pnl_pct = (unrealized_pnl / cost_value) if cost_value != 0 else 0.0
            
            detailed_positions[symbol] = {
                "qty": qty,
                "name": self.stock_names.get(symbol, "Unknown"),
                "price": price,
                "value": market_value,
                "avg_cost": avg_cost,
                "unrealized_pnl": unrealized_pnl,
                "pnl_pct": pnl_pct
            }
            
        return {
            "cash": self.cash,
            "positions": self.positions,
            "detailed_positions": detailed_positions,
            "total_equity": self.get_total_equity(),
            "equity_history": self.equity_history,
            "trades": self.trades,
            "pending_orders": [vars(o) for o in self.orders.values()]
        }

    def get_total_equity(self) -> float:
        equity = self.cash
        for symbol, qty in self.positions.items():
            price = 0.0
            if symbol in self.current_bars:
                price = self.current_bars[symbol].close
            elif self.db_client and self.current_bars:
                ts = next(iter(self.current_bars.values())).timestamp
                try:
                    df = self.db_client.get_price(symbol, str(ts.date()), ["close"], 1)
                    if not df.empty:
                        price = df.iloc[0]["close"]
                        self.last_prices[symbol] = price
                except Exception:
                    pass
            
            if price == 0 and symbol in self.last_prices:
                price = self.last_prices[symbol]
                
            equity += qty * price
        return equity

    def process_same_bar_orders(self, bars: Dict[str, Bar], timing: str):
        """Process orders that should be filled immediately (SAME_BAR) at OPEN or CLOSE"""
        if not bars: return
        
        # Update last known prices
        for sym, bar in bars.items():
            self.last_prices[sym] = bar.close
        
        current_ts = next(iter(bars.values())).timestamp
        
        for order_id in list(self.orders.keys()):
            order = self.orders[order_id]
            
            # Filter by timing (e.g., 'IMMEDIATE_OPEN' or 'IMMEDIATE_CLOSE')
            if order.execution_type != timing:
                continue
                
            bar = bars.get(order.symbol)
            execution_price = None
            
            if bar:
                if timing == 'IMMEDIATE_OPEN':
                    execution_price = bar.open
                elif timing == 'IMMEDIATE_CLOSE':
                    execution_price = bar.close
            elif self.db_client:
                # Fallback DB fetch if bar missing
                try:
                    field = "open" if timing == 'IMMEDIATE_OPEN' else "close"
                    df = self.db_client.get_price(order.symbol, str(current_ts.date()), [field], 1)
                    if not df.empty:
                        execution_price = df.iloc[0][field]
                except Exception as e:
                    logger.error(f"Failed to fetch price for {order.symbol}: {e}")
            
            if execution_price is None:
                continue
            
            self._execute_order(order, execution_price, current_ts)

    def _execute_order(self, order, execution_price, current_ts):
        amount = execution_price * order.quantity
        fee = amount * self.commission
        
        if order.type == 'buy':
            if self.cash >= amount + fee:
                self.cash -= (amount + fee)
                curr_qty = self.positions.get(order.symbol, 0)
                curr_cost = self.position_costs.get(order.symbol, 0.0)
                total_shares = curr_qty + order.quantity
                
                # Update average cost (including commissions)
                total_spent = (curr_qty * curr_cost) + amount + fee
                new_cost = total_spent / total_shares
                
                self.position_costs[order.symbol] = new_cost
                self.positions[order.symbol] = total_shares
                order.status = "FILLED"
                order.avg_fill_price = execution_price
                order.filled_quantity = order.quantity
                
                # Update risk manager
                if self.risk_manager:
                    self.risk_manager.add_position(order.symbol, order.quantity, execution_price)
                    
            else:
                order.status = "REJECTED"
                logger.warning(f"Order REJECTED (Insufficient cash): {order.symbol}")
        elif order.type == 'sell':
            curr_qty = self.positions.get(order.symbol, 0)
            if curr_qty >= order.quantity:
                self.cash += (amount - fee)
                new_qty = curr_qty - order.quantity
                
                # Update risk manager before modifying positions
                if self.risk_manager and order.symbol in self.risk_manager.positions:
                    self.risk_manager.close_position(order.symbol, execution_price)
                
                if new_qty > 0:
                    self.positions[order.symbol] = new_qty
                else:
                    self.positions.pop(order.symbol, None)
                    self.position_costs.pop(order.symbol, None)
                    
                order.status = "FILLED"
                order.avg_fill_price = execution_price
                order.filled_quantity = order.quantity
            else:
                order.status = "REJECTED"
                logger.warning(f"Order REJECTED (Insufficient qty): {order.symbol}")
        
        if order.status == "FILLED":
            logger.info(f"ORDER FILLED ({order.execution_type}): {order.type} {order.quantity} {order.symbol} at {execution_price} on {current_ts}")
            self.trades.append({
                "timestamp": str(current_ts),
                "symbol": order.symbol,
                "name": self.stock_names.get(order.symbol, "Unknown"),
                "type": order.type,
                "price": float(execution_price),
                "quantity": float(order.quantity),
                "amount": float(amount),
                "commission": float(fee)
            })
        
        if order.status in ["FILLED", "REJECTED"]:
            self.history.append(self.orders.pop(order.id))

    def step(self, bars: Dict[str, Bar]):
        self.current_bars = bars
        if not bars: return
        
        current_ts = next(iter(bars.values())).timestamp
        
        # Update risk manager positions with current prices
        if self.risk_manager:
            for symbol in self.positions.keys():
                if symbol in bars:
                    current_price = bars[symbol].close
                    self.risk_manager.update_position(symbol, current_price)
                    
                    # Check stop loss and take profit
                    stop_triggered, stop_reason = self.risk_manager.check_stop_loss(symbol, current_price)
                    if stop_triggered:
                        logger.warning(f"Stop loss triggered for {symbol}: {stop_reason}")
                        # Auto-create sell order
                        qty = self.positions.get(symbol, 0)
                        if qty > 0:
                            sell_order = Order(
                                symbol=symbol,
                                type='sell',
                                quantity=qty,
                                execution_type='IMMEDIATE_CLOSE'
                            )
                            self.submit_order(sell_order)
                    
                    profit_triggered, profit_reason = self.risk_manager.check_take_profit(symbol, current_price)
                    if profit_triggered:
                        logger.info(f"Take profit triggered for {symbol}: {profit_reason}")
                        # Auto-create sell order
                        qty = self.positions.get(symbol, 0)
                        if qty > 0:
                            sell_order = Order(
                                symbol=symbol,
                                type='sell',
                                quantity=qty,
                                execution_type='IMMEDIATE_CLOSE'
                            )
                            self.submit_order(sell_order)
            
            # Update capital
            self.risk_manager.current_capital = self.get_total_equity()
        
        # Process NEXT_OPEN orders (Standard Backtest behavior)
        for order_id in list(self.orders.keys()):
            order = self.orders[order_id]
            if order.execution_type != "NEXT_OPEN":
                continue
                
            bar = bars.get(order.symbol)
            execution_price = None
            
            if bar:
                execution_price = bar.open
            elif self.db_client:
                try:
                    df = self.db_client.get_price(order.symbol, str(current_ts.date()), ["open"], 1)
                    if not df.empty:
                        execution_price = df.iloc[0]["open"]
                except Exception as e:
                    logger.error(f"Failed to fetch price for {order.symbol}: {e}")
            
            if execution_price is None:
                continue
            
            self._execute_order(order, execution_price, current_ts)
        
        # Record equity daily
        current_equity = float(self.get_total_equity())
        daily_pnl = 0.0
        daily_return = 0.0
        
        # Use _last_equity for calculation
        prev_equity = self._last_equity
        daily_pnl = current_equity - prev_equity
        daily_return = daily_pnl / prev_equity if prev_equity != 0 else 0.0
        
        # Update _last_equity
        self._last_equity = current_equity
        
        # Get positions info reusing the logic (simplified)
        pos_snapshot = {}
        for k, v in self.positions.items():
            # Robust price fetching: Current Bar -> DB (Current Day) -> Last Known Price
            price = 0.0
            if k in bars:
                price = float(bars[k].close)
            elif self.db_client:
                 try:
                    df = self.db_client.get_price(k, str(current_ts.date()), ["close"], 1)
                    if not df.empty:
                        price = float(df.iloc[0]["close"])
                        self.last_prices[k] = price
                 except: pass
            
            if price == 0 and k in self.last_prices:
                price = float(self.last_prices[k])
            
            # If still 0, warn
            if price == 0:
                logger.warning(f"Could not find price for {k} at {current_ts}")
            
            avg_cost = round(float(self.position_costs.get(k, 0.0)), 2)
            market_value = round(float(v * price), 2)
            cost_value = round(float(v * avg_cost), 2)
            unrealized_pnl = round(float(market_value - cost_value), 2)
            pnl_pct = round(float(unrealized_pnl / cost_value), 4) if cost_value != 0 else 0.0

            pos_snapshot[k] = {
                "qty": float(v),
                "name": self.stock_names.get(k, "Unknown"),
                "price": round(float(price), 2),
                "value": market_value,
                "avg_cost": avg_cost,
                "unrealized_pnl": unrealized_pnl,
                "pnl_pct": pnl_pct
            }

        self.equity_history.append({
            "timestamp": str(current_ts),
            "total_equity": round(current_equity, 2),
            "daily_pnl": round(daily_pnl, 2),
            "daily_return": round(daily_return, 4),
            "cash": round(float(self.cash), 2),
            "positions": pos_snapshot
        })

    def get_report(self):
        if not self.equity_history:
            return "No backtest data available."
            
        initial_equity = self.equity_history[0]['total_equity']
        final_equity = self.equity_history[-1]['total_equity']
        total_ret = (final_equity - initial_equity) / initial_equity
        
        report = []
        report.append("="*40)
        report.append("BACKTEST REPORT summary")
        report.append("="*40)
        report.append(f"Initial Equity: {initial_equity:,.2f}")
        report.append(f"Final Equity:   {final_equity:,.2f}")
        report.append(f"Total Return:   {total_ret:.2%}")
        report.append(f"Total Trades:   {len(self.trades)}")
        report.append("\n" + "="*40)
        report.append("DAILY HOLDINGS & EQUITY")
        report.append("="*40)
        
        for entry in self.equity_history:
            report.append(f"\nDate: {entry['timestamp']}")
            report.append(f"Equity: {entry['total_equity']:,.2f} | Cash: {entry['cash']:,.2f}")
            if entry['positions']:
                report.append("Holdings:")
                for symbol, info in entry['positions'].items():
                    report.append(f"  - {symbol} ({info['name']}): {info['qty']}")
            else:
                report.append("Holdings: None")
        
        report.append("\n" + "="*40)
        report.append("TRADE DETAILS")
        report.append("="*40)
        for t in self.trades:
            report.append(f"[{t['timestamp']}] {t['type'].upper()} {t['quantity']} {t['symbol']} ({t['name']}) @ {t['price']:.2f} | Amt: {t['amount']:,.2f}")
            
        return "\n".join(report)
