from typing import Dict, Any, List
from .base import Broker, Order, Bar
from datetime import datetime
from loguru import logger
import uuid

class BacktestBroker(Broker):
    def __init__(self, initial_cash: float = 1000000.0, commission: float = 0.0001, db_client=None):
        self.cash = initial_cash
        self.commission = commission
        self.db_client = db_client
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.orders: Dict[str, Order] = {}
        self.history: List[Order] = []
        self.current_bars: Dict[str, Bar] = {}
        self.equity_history: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        
        # New: Track cost basis per position
        self.position_costs: Dict[str, float] = {} # symbol -> avg_price
        
        # Optimization: Track last equity for PnL calc even if history is cleared
        self._last_equity = initial_cash
        
        # Stock name mapping
        self.stock_names: Dict[str, str] = {}
        self._load_stock_names()

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
            elif self.db_client and self.current_bars:
                # Attempt fallback fetch
                try:
                    ts = next(iter(self.current_bars.values())).timestamp
                    df = self.db_client.get_price(symbol, str(ts.date()), ["close"], 1)
                    if not df.empty:
                        price = df.iloc[0]["close"]
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
                except Exception:
                    pass
            equity += qty * price
        return equity

    def step(self, bars: Dict[str, Bar]):
        self.current_bars = bars
        if not bars: return
        
        current_ts = next(iter(bars.values())).timestamp
        
        for order_id in list(self.orders.keys()):
            order = self.orders[order_id]
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
            
            amount = execution_price * order.quantity
            fee = amount * self.commission
            
            if order.type == 'buy':
                if self.cash >= amount + fee:
                    self.cash -= (amount + fee)
                    
                    # Update average cost
                    curr_qty = self.positions.get(order.symbol, 0)
                    curr_cost = self.position_costs.get(order.symbol, 0.0)
                    total_shares = curr_qty + order.quantity
                    new_cost = ((curr_qty * curr_cost) + (order.quantity * execution_price)) / total_shares
                    self.position_costs[order.symbol] = new_cost
                    
                    self.positions[order.symbol] = total_shares
                    order.status = "FILLED"
                    order.avg_fill_price = execution_price
                    order.filled_quantity = order.quantity
                else:
                    order.status = "REJECTED"
                    logger.warning(f"Order REJECTED (Insufficient cash): {order.symbol} need {amount+fee}, have {self.cash}")
            elif order.type == 'sell':
                curr_qty = self.positions.get(order.symbol, 0)
                if curr_qty >= order.quantity:
                    self.cash += (amount - fee)
                    new_qty = curr_qty - order.quantity
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
                    logger.warning(f"Order REJECTED (Insufficient quantity): {order.symbol} need {order.quantity}, have {curr_qty}")
            
            if order.status == "FILLED":
                logger.info(f"ORDER FILLED: {order.type} {order.quantity} {order.symbol} at {execution_price} on {current_ts}")
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
                self.history.append(self.orders.pop(order_id))
        
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
            price = float(bars[k].close) if k in bars else 0.0
            # Fallback if price is 0 (missing bar) - try to keep last known or 0
            if price == 0 and self.current_bars and k in self.current_bars:
                 price = self.current_bars[k].close
            
            avg_cost = self.position_costs.get(k, 0.0)
            market_value = v * price
            cost_value = v * avg_cost
            unrealized_pnl = market_value - cost_value
            pnl_pct = (unrealized_pnl / cost_value) if cost_value != 0 else 0.0

            pos_snapshot[k] = {
                "qty": v,
                "name": self.stock_names.get(k, "Unknown"),
                "price": price,
                "value": market_value,
                "avg_cost": avg_cost,
                "unrealized_pnl": unrealized_pnl,
                "pnl_pct": pnl_pct
            }

        self.equity_history.append({
            "timestamp": str(current_ts),
            "total_equity": current_equity,
            "daily_pnl": daily_pnl,
            "daily_return": daily_return,
            "cash": float(self.cash),
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