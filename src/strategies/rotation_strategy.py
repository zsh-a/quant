import pandas as pd
import numpy as np
import talib as ta
from src.core.base import Strategy, Bar

PRICE_CHANGE_LIMIT = 0.098
MAX_POSITION = 99999999

# Shenwan Level 1 Industry Mapping (Partial, as in original)
SW1 = {
    '801010': '农林牧渔I', '801020': '采掘I', '801030': '化工I', '801040': '钢铁I',
    '801050': '有色金属I', '801060': '建筑建材I', '801070': '机械设备I', '801080': '电子I',
    '801090': '交运设备I', '801100': '信息设备I', '801110': '家用电器I', '801120': '食品饮料I',
    '801130': '纺织服装I', '801140': '轻工制造I', '801150': '医药生物I', '801160': '公用事业I',
    '801170': '交通运输I', '801180': '房地产I', '801190': '金融服务I', '801200': '商业贸易I',
    '801210': '休闲服务I', '801220': '信息服务I', '801230': '综合I', '801710': '建筑材料I',
    '801720': '建筑装饰I', '801730': '电气设备I', '801740': '国防军工I', '801750': '计算机I',
    '801760': '传媒I', '801770': '通信I', '801780': '银行I', '801790': '非银金融I',
    '801880': '汽车I', '801890': '机械设备I', '801950': '煤炭I', '801960': '石油石化I',
    '801970': '环保I', '801980': '美容护理I', '999998': '小市值200', '999999': '创业板50',
}

class RotationStrategy(Strategy):
    def __init__(self, db_client, session_id: str = None, **kwargs):
        super().__init__(session_id=session_id)  # Pass session_id to base class
        self.db_client = db_client
        
        # Load parameters
        params = self.get_parameters()
        self.stock_sum = kwargs.get('stock_sum', params['stock_sum']['default'])
        self.timing = kwargs.get('timing', params['timing']['default'])
        
        rebalance_dates_str = kwargs.get('rebalance_dates', None)
        self.rebalance_dates = None
        if rebalance_dates_str:
            self.rebalance_dates = [d.strip() for d in rebalance_dates_str.split(',') if d.strip()]

        self.JSG_group = {'银行I', '有色金属I', '钢铁I', '煤炭I'}
        self.XSZ_group = {'小市值200'}
        self.CYB_group = {'创业板50'}
        self.black_industry_name = {"银行I", "煤炭I", "采掘I", "钢铁I"}
        
        self.trad_days = pd.read_csv(
            "marked_trade_datas.csv", index_col="calendar_date", parse_dates=True
        )
        
        self._log(f"RotationStrategy initialized: stock_sum={self.stock_sum}, timing={self.timing}")
        
        # Calculate trigger dates for Backtest and Live
        self.trigger_dates_backtest = set()
        self.trigger_dates_live = set()
        
        if self.rebalance_dates:
            # Filter trad_days to only trading days
            trading_days_df = self.trad_days[self.trad_days['is_trading_day'] == 1].index
            
            for d_str in self.rebalance_dates:
                try:
                    target_date = pd.Timestamp(d_str)
                    
                    # Backtest: Shift D-1 for OPEN
                    if self.timing == 'OPEN':
                        prev_days = trading_days_df[trading_days_df < target_date]
                        if not prev_days.empty:
                            self.trigger_dates_backtest.add(prev_days[-1].strftime('%Y-%m-%d'))
                        else:
                            logger.warning(f"No previous trading day for {d_str}")
                            
                        # Live: Trigger on D for OPEN (at 09:30)
                        curr_days = trading_days_df[trading_days_df <= target_date]
                        if not curr_days.empty and curr_days[-1] == target_date: # Ensure target is trading day
                             self.trigger_dates_live.add(target_date.strftime('%Y-%m-%d'))
                    else: # CLOSE
                        # Both trigger on D
                        curr_days = trading_days_df[trading_days_df <= target_date]
                        if not curr_days.empty:
                            d_fmt = curr_days[-1].strftime('%Y-%m-%d')
                            self.trigger_dates_backtest.add(d_fmt)
                            self.trigger_dates_live.add(d_fmt)
                            
                except Exception as e:
                    logger.error(f"Error parsing date {d_str}: {e}")

    @classmethod
    def get_parameters(cls) -> dict:
        return {
            "stock_sum": {
                "type": "int",
                "default": 10,
                "description": "Number of stocks to hold",
                "min": 1,
                "max": 20
            },
            "timing": {
                "type": "str",
                "default": "OPEN",
                "description": "Rebalance execution timing (OPEN/CLOSE)",
                "options": ["OPEN", "CLOSE"]
            },
            "rebalance_dates": {
                "type": "str",
                "default": "",
                "description": "Specific rebalance dates (comma separated, YYYY-MM-DD)",
            }
        }

    def on_bar(self, bars: dict[str, Bar]):
        if not bars: return
        
        today_str = self._current_date  # Auto-updated by engine

        should_run = False
        is_live = self.engine and self.engine.broker.__class__.__name__ == 'LiveBroker'
        
        trigger_dates = self.trigger_dates_live if is_live else self.trigger_dates_backtest
        
        # 1. Check Date
        if self.rebalance_dates:
            if today_str in trigger_dates:
                should_run = True
        else:
            # Fallback to original logic (Last trading day of month)
            if today_str in self.trad_days.index.strftime('%Y-%m-%d'):
                if self.trad_days.loc[today_str, "is_last_trading_day"] != 0:
                    should_run = True

        if not should_run:
            return
            
        # 2. Check Time (for Live Mode alignment)
        if is_live:
             # OPEN: Run at 09:30
             if self.timing == 'OPEN':
                 if ts.hour < 9 or (ts.hour == 9 and ts.minute < 30): return
             # CLOSE: Run at 14:50
             else:
                 if ts.hour < 14 or (ts.hour == 14 and ts.minute < 50): return
        
        # For Backtest 'OPEN' timing, we are running on D-1. We don't check time (00:00).

        self._log(f"========== 轮动调仓日 (Timing: {self.timing}, Live: {is_live}) ==========")

        # 1. Market Breath Analysis
        df_ratio = self.get_market_breadth(today_str)
        if df_ratio.empty: return

        means = self.calculate_group_means(df_ratio)
        max_group = max(means, key=means.get)
        max_mean = means[max_group]
        
        # Extract top industries and market environment for defensive check
        I_top = df_ratio.nlargest(1, 'ratio')['name'].tolist()
        market_env = self.judge_market_env(today_str)

        if any(item in self.black_industry_name for item in I_top) and market_env == '存量':
            self.adjust([], today_str)
            return

        final_list = []
        if max_group == 'JSG' and max_mean > 90:
            L2 = self.get_L2(today_str)
            max_ind_code = df_ratio[df_ratio['name'].isin(self.JSG_group)]['ratio'].idxmax()
            L1_stocks = self.db_client.get_swindustry_stocks(max_ind_code, today_str)
            L1 = self.get_L1(L1_stocks, today_str)
            final_list = L1[:1] + L2[:9]
        elif max_group == 'XSZ':
            final_list = self.get_L2(today_str)[:10]
        elif max_group == 'CYB':
            L2 = self.get_L2(today_str)
            final_list = L2[:9] + ['159915'] 
        else:
            final_list = self.get_L2(today_str)[:10]

        # 3. Adjust Positions
        self.adjust(final_list, today_str)

    def get_market_breadth(self, end_date):
        all_stocks = self.db_client.get_index_stocks('000985', end_date)
        h1 = self.db_client.get_price(all_stocks, end_date, ['close'], 21)
        if h1.empty: return pd.DataFrame()
        
        h1["ma20"] = h1.groupby(level="code")["close"].transform(lambda x: ta.MA(x, timeperiod=20))
        h1 = h1.groupby(level=0).tail(1).copy()
        h1["bias"] = h1["close"] > h1["ma20"]
        
        industry_df = self.db_client.get_stock_industry_sw(h1.index.get_level_values('code').to_list(), end_date)
        h1["industry_code"] = industry_df.reindex(h1.index.get_level_values(0))["industry_code"].values
        df_ratio1 = (h1.groupby("industry_code")["bias"].mean() * 100.0).round()

        # Simplified for brevity (implement XSZ/CYB logic if needed)
        df_ratio = df_ratio1 
        res_df = pd.DataFrame({'ratio': df_ratio})
        res_df['name'] = [SW1.get(c, c) for c in res_df.index]
        return res_df

    def calculate_group_means(self, df_ratio):
        means = {
            'JSG': df_ratio[df_ratio['name'].isin(self.JSG_group)]['ratio'].mean() or 0,
            'XSZ': df_ratio[df_ratio['name'].isin(self.XSZ_group)]['ratio'].mean() or 0,
            'CYB': df_ratio[df_ratio['name'].isin(self.CYB_group)]['ratio'].mean() or 0,
            'OTHER': df_ratio[~df_ratio['name'].isin(self.JSG_group | self.XSZ_group | self.CYB_group)]['ratio'].mean() or 0
        }
        return {k: (v if not np.isnan(v) else 0) for k, v in means.items()}

    def judge_market_env(self, date_str):
        df_sh = self.db_client.get_price(['sh.000001'], date_str, ['amount'], 25)
        df_sz = self.db_client.get_price(['sz.399001'], date_str, ['amount'], 25)
        if df_sh.empty or df_sz.empty: return None
        
        total_money = df_sh['amount'].values + df_sz['amount'].values
        ma_total = pd.Series(total_money).rolling(20).mean().dropna()
        if len(ma_total) < 6: return None
        change_total = (ma_total.iloc[-1] - ma_total.iloc[-6]) / ma_total.iloc[-6]
        return '存量' if change_total <= 0.1 else None

    def get_L2(self, date_str):
        S_stocks = self.db_client.get_index_stocks('399101', date_str)
        # Add basic filters here (ST etc.) if needed as in original
        fin_db = self.db_client.get_stock_fincial(S_stocks, fields=["roe", "roa", "total_shares"], date=date_str)
        if fin_db.empty: return []
        
        df_prc = self.db_client.get_price(fin_db.index.to_list(), date_str, ["close"], 1, price_adj=False)
        if not df_prc.empty:
            fin_db["close"] = df_prc["close"]
            fin_db["market_cap"] = fin_db["close"] * fin_db["total_shares"]
        
        mask = (fin_db["roe"] > 0.15) & (fin_db["roa"] > 0.10)
        return fin_db[mask].sort_values(by="market_cap", ascending=True).index.to_list()

    def get_L1(self, stocks, date_str):
        if not stocks: return []
        fin_db = self.db_client.get_stock_fincial(stocks, fields=["roa", "pb_ratio"], date=date_str)
        mask = (fin_db["pb_ratio"] < 1.0) & (fin_db["roa"] > 0.15)
        return fin_db[mask].sort_values(by="roa", ascending=False).index.to_list()

    def adjust(self, target_stocks, date_str):
        target = target_stocks[: min(len(target_stocks), self.stock_sum)]
        account = self.engine.broker.get_account_info()
        hold_list = list(account['positions'].keys())
        
        self._log(f"调仓: 目标={target}, 当前持仓={hold_list}")
        
        # Determine execution type
        is_live = self.engine and self.engine.broker.__class__.__name__ == 'LiveBroker'
        
        exec_type = 'NEXT_OPEN'
        if self.timing == 'CLOSE':
            exec_type = 'IMMEDIATE_CLOSE'
        elif self.timing == 'OPEN' and is_live:
             # In Live OPEN, we run at 09:30, so we want IMMEDIATE filling
             exec_type = 'IMMEDIATE_OPEN'
        
        # 1. Sell stocks not in target
        for stock in hold_list:
            if stock not in target:
                qty = account['positions'][stock]
                if qty > 0:
                    self._log(f"清仓 {stock}: qty={qty} (不在目标中)", stock=stock)
                    self.sell(stock, qty, execution_type=exec_type)

        # 2. Buy/Rebalance target stocks
        total_equity = account['total_equity']
        if target:
            val_per_stock = (total_equity * 0.95) / len(target) # 5% cash buffer
            for code in target:
                price_df = self.db_client.get_price(code, date_str, ["close"], 1)
                if price_df.empty:
                    self._log(f"跳过 {code}: 无价格数据", level="WARNING", stock=code)
                    continue
                price = price_df.iloc[0]["close"]
                target_qty = int(val_per_stock / price // 100 * 100)
                
                curr_qty = account['positions'].get(code, 0)
                if target_qty > curr_qty:
                    self._log(f"买入 {code}: {target_qty - curr_qty}股, 当前={curr_qty}, 目标={target_qty}", stock=code)
                    self.buy(code, target_qty - curr_qty, execution_type=exec_type)
                elif target_qty < curr_qty:
                    self._log(f"卖出 {code}: {curr_qty - target_qty}股, 当前={curr_qty}, 目标={target_qty}", stock=code)
                    self.sell(code, curr_qty - target_qty, execution_type=exec_type)
