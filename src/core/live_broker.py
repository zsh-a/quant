from typing import Dict, Any, List
from .base import Broker, Order, Bar
from loguru import logger
from datetime import datetime
import requests
import uuid
import json

class LiveBroker(Broker):
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.orders: Dict[str, Order] = {}
        self.stock_names: Dict[str, str] = {}
        self.positions: Dict[str, float] = {}
        self._load_stock_names()
        self.sync_state()

    def sync_state(self):
        """Sync local state with remote server."""
        info = self.get_account_info()
        self.positions = info.get("positions", {})
        # We could also sync cash, etc. if needed
        logger.info(f"LiveBroker synced. Positions: {self.positions}")

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
        
        try:
            endpoint = "/buy" if order.type == "buy" else "/sell"
            params = {
                "code": order.symbol,
                "amount": int(order.quantity)
            }
            if order.price:
                params["price"] = order.price
                
            url = f"{self.server_url}{endpoint}"
            logger.info(f"LiveBroker submitting: {url} {params}")
            
            resp = requests.get(url, params=params)
            data = resp.json()
            
            if data.get("status") == -1:
                order.status = "REJECTED"
                logger.error(f"Order rejected: {data}")
            else:
                order.status = "FILLED"
                # Assume filled at ordered price or current market price?
                # Without real-time callback, we just mark it filled.
                # In real life, we would query 'deal' endpoint.
                pass
                
        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            order.status = "ERROR"
            
        return order.id

    def cancel_order(self, order_id: str):
        logger.warning("Cancel order not implemented in underlying client")
        pass

    def get_account_info(self) -> Dict[str, Any]:
        try:
            # Fetch balance
            bal_resp = requests.get(f"{self.server_url}/balance")
            bal_data = bal_resp.json()
            
            cash = 0.0
            if isinstance(bal_data, dict):
                 cash = float(bal_data.get("zj", 0) if isinstance(bal_data, dict) else 0)

            # Fetch positions
            pos_resp = requests.get(f"{self.server_url}/position")
            pos_data = pos_resp.json()
            
            positions = {}
            detailed = {}
            
            if isinstance(pos_data, dict) and "data" in pos_data:
                for p in pos_data["data"]:
                    code = p.get("证券代码")
                    qty = float(p.get("股票余额", 0))
                    name = p.get("证券名称", "Unknown")
                    price = float(p.get("市价", 0))
                    
                    cost_price = float(p.get("成本价", 0))
                    market_value = float(p.get("市值", qty * price))
                    pnl = float(p.get("浮动盈亏", 0))
                    pnl_ratio = float(p.get("盈亏比例(%)", 0)) / 100.0 if "盈亏比例(%)" in p else 0.0
                    
                    if code and qty > 0:
                        positions[code] = qty
                        detailed[code] = {
                            "qty": qty,
                            "name": name,
                            "price": price,
                            "value": market_value,
                            "avg_cost": cost_price,
                            "unrealized_pnl": pnl,
                            "pnl_pct": pnl_ratio
                        }
            
            # Construct local trades list from filled orders
            trades_list = []
            for order in self.orders.values():
                if order.status == "FILLED":
                    trades_list.append({
                        "timestamp": str(order.created_at), # Use order creation time as trade time approx
                        "symbol": order.symbol,
                        "name": self.stock_names.get(order.symbol, "Unknown"),
                        "type": order.type,
                        "price": order.price if order.price else 0.0, # We don't have fill price easily without another query
                        "quantity": order.quantity,
                        "amount": order.quantity * (order.price if order.price else 0.0),
                        "commission": 0.0 # Unknown
                    })

            return {
                "cash": cash,
                "positions": positions,
                "detailed_positions": detailed,
                "total_equity": cash + sum(p['value'] for p in detailed.values()),
                "equity_history": [],
                "trades": trades_list,
                "pending_orders": [vars(o) for o in self.orders.values() if o.status == "SUBMITTED"]
            }
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            return {"cash": 0, "positions": {}, "detailed_positions": {}, "total_equity": 0}

    def step(self, bars: Dict[str, Bar]):
        pass
