import numpy as np
import pandas as pd
import talib as ta
from loguru import logger
from policy.base_policy import OrderPolicy
from market_env import MultiMarketEnv

# Constants from original JoinQuant strategy
PRICE_CHANGE_LIMIT = 0.098
MAX_POSITION = 99999999

NUM_STOCKS = 10

# Shenwan Level 1 Industry Mapping
SW1 = {
    '801010': '农林牧渔I',
    '801020': '采掘I',
    '801030': '化工I',
    '801040': '钢铁I',
    '801050': '有色金属I',
    '801060': '建筑建材I',
    '801070': '机械设备I',
    '801080': '电子I',
    '801090': '交运设备I',
    '801100': '信息设备I',
    '801110': '家用电器I',
    '801120': '食品饮料I',
    '801130': '纺织服装I',
    '801140': '轻工制造I',
    '801150': '医药生物I',
    '801160': '公用事业I',
    '801170': '交通运输I',
    '801180': '房地产I',
    '801190': '金融服务I',
    '801200': '商业贸易I',
    '801210': '休闲服务I',
    '801220': '信息服务I',
    '801230': '综合I',
    '801710': '建筑材料I',
    '801720': '建筑装饰I',
    '801730': '电气设备I',
    '801740': '国防军工I',
    '801750': '计算机I',
    '801760': '传媒I',
    '801770': '通信I',
    '801780': '银行I',
    '801790': '非银金融I',
    '801880': '汽车I',
    '801890': '机械设备I',
    '801950': '煤炭I',
    '801960': '石油石化I',
    '801970': '环保I',
    '801980': '美容护理I',
    '999998': '小市值200',
    '999999': '创业板50',
}

class AdvancedAgent:
    def __init__(self, market_env: MultiMarketEnv, **args) -> None:
        self.market_env = market_env
        self.db_client = args["db_client"]
        self.current_date = None
        self.pass_month = []
        self.pool_size = 20
        self.stock_sum = 10  # G.stock_num from original strategy

        self.JSG_group = {'银行I', '有色金属I', '钢铁I', '煤炭I'}
        self.XSZ_group = {'小市值200'}
        self.CYB_group = {'创业板50'}
        self.black_industry_name = {"银行I", "煤炭I", "采掘I", "钢铁I"}

        self.trad_days = pd.read_csv(
            "marked_trade_datas.csv", index_col="calendar_date", parse_dates=True
        )
        self.yesterday_HL_list = []

    def get_current_date_str(self):
        return str(self.current_date.date())

    def get_next_trading_day(self):
        date = pd.to_datetime(self.current_date)
        next_date = date + pd.DateOffset(days=1)
        while self.trad_days.loc[str(next_date.date()), "is_trading_day"] == 0:
            next_date += pd.DateOffset(days=1)
        return next_date

    def judge_market_env(self, ma_window=20, slope_window=5):
        yesterday = self.get_current_date_str()
        
        # Combined amount for market sentiment logic
        df_sh = self.db_client.get_price(['sh.000001'], yesterday, ['amount'], ma_window + slope_window)
        df_sz = self.db_client.get_price(['sz.399001'], yesterday, ['amount'], ma_window + slope_window)
        
        if df_sh.empty or df_sz.empty:
            return None
            
        total_money = df_sh['amount'].values + df_sz['amount'].values
        ma_total = pd.Series(total_money).rolling(ma_window).mean().dropna()
        if len(ma_total) < slope_window + 1:
            return None
            
        change_total = (ma_total.iloc[-1] - ma_total.iloc[-slope_window - 1]) / ma_total.iloc[-slope_window - 1]
        
        # Bank Index return check
        price_df = self.db_client.get_price(['sz.399986'], yesterday, ['close'], ma_window)
        if price_df.empty:
            return None
            
        close_prices = price_df['close']
        bank_return = close_prices.iloc[-1] / close_prices.iloc[0]
        
        return '存量' if change_total <= 0.1 or bank_return <= 0.9 else None

    def get_market_breadth(self, end_date):
        # Group 1: All SW Level 1 Industries
        all_stocks = self.db_client.get_index_stocks('000985', end_date)
        logger.debug(f"get_market_breadth: all_stocks count: {len(all_stocks)}")
        h1 = self.db_client.get_price(all_stocks, end_date, ['close'], 21)
        if h1.empty: 
            logger.warning(f"get_market_breadth: h1 empty for {end_date}")
            return pd.DataFrame()
        
        h1["ma20"] = h1.groupby(level="code")["close"].transform(lambda x: ta.MA(x, timeperiod=20))
        h1 = h1.groupby(level=0).tail(1).copy()
        h1["bias"] = h1["close"] > h1["ma20"]
        
        industry_df = self.db_client.get_stock_industry_sw(h1.index.get_level_values('code').to_list(), end_date)
        h1["industry_code"] = industry_df.reindex(h1.index.get_level_values(0))["industry_code"].values
        df_ratio1 = (h1.groupby("industry_code")["bias"].mean() * 100.0).round()

        # Group 2: SmallCap 200 simulation
        S_stocks = self.db_client.get_index_stocks('399101', end_date)
        S_stocks = self.filter_basic(S_stocks)
        fin_date = (self.get_next_trading_day() - pd.DateOffset(days=1)).date()
        df_shrs = self.db_client.get_stock_shares_info(S_stocks, fin_date)
        df_prc = self.db_client.get_price(S_stocks, end_date, ['close'], 1, price_adj=False)
        if not df_shrs.empty and not df_prc.empty:
            if "date" in df_prc.index.names:
                df_prc = df_prc.reset_index(level="date", drop=True)
            merged = pd.concat([df_shrs['total_shares'], df_prc['close']], axis=1).dropna()
            mkt_cap = (merged['total_shares'] * merged['close']).sort_values()
            Slst = mkt_cap.index[:200].to_list()
        else:
            Slst = []
        
        # Group 3: GEM 50 simulation
        Blst = self.db_client.get_index_stocks('399673', end_date)

        def calc_pseudo_bias(stocks, code):
            if not stocks: return pd.Series()
            h = self.db_client.get_price(stocks, end_date, ['close'], 21)
            if h.empty: return pd.Series()
            h["ma20"] = h.groupby(level="code")["close"].transform(lambda x: ta.MA(x, timeperiod=20))
            h = h.groupby(level=0).tail(1)
            bias = (h["close"] > h["ma20"]).mean() * 100.0
            return pd.Series([round(bias)], index=[code])

        df_ratio2 = calc_pseudo_bias(Slst, '999998')
        df_ratio3 = calc_pseudo_bias(Blst, '999999')

        df_ratio = pd.concat([df_ratio1, df_ratio2, df_ratio3])
        res_df = pd.DataFrame({'ratio': df_ratio})
        res_df['name'] = [SW1.get(c, c) for c in res_df.index]
        return res_df

    def filter_basic(self, stocks):
        if not stocks: return []
        # Ensure unique stocks to avoid duplicate index issues
        stocks = list(set(stocks))
        df = self.db_client.get_price(stocks, self.get_current_date_str(), ["isST"], 1)
        if df.empty: return []
        
        if "date" in df.index.names:
            df = df.reset_index(level="date", drop=True)
            
        # Ensure index is unique
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='first')]
            
        # Use .values to avoid alignment issues with duplicate labels if any remain
        mask = (df["tradestatus"].values == 1) & (df["isST"].values == 0)
        df = df[mask]
        return df.index.to_list()

    def get_L1(self, initial_list):
        # Value selection: PB < 1, ROA > 0.15, Profit Growth > 0, Cash Inflow > 1M, Adjusted Profit > 1M
        if not initial_list: return []
        fin_date = (self.get_next_trading_day() - pd.DateOffset(days=1)).date()
        fin_db = self.db_client.get_stock_fincial(
            initial_list, 
            fields=["roa", "pb_ratio", "inc_net_profit_year_on_year", "subtotal_operate_cash_inflow", "adjusted_profit"], 
            date=fin_date
        )
        if fin_db.empty: return []
        
        mask = (fin_db["pb_ratio"] < 1.0) & \
               (fin_db["roa"] > 0.15) & \
               (fin_db["inc_net_profit_year_on_year"] > 0) & \
               (fin_db["subtotal_operate_cash_inflow"] > 1e6) & \
               (fin_db["adjusted_profit"] > 1e6)
        
        L1_stocks = fin_db[mask].sort_values(by="roa", ascending=False).index.to_list()
        logger.info(f"get_L1: found {len(L1_stocks)} stocks")
        return self.filter_basic(L1_stocks)

    def get_L2(self, today):
        # Growth selection: ROE > 0.15, ROA > 0.10, Smallest Market Cap
        S_stocks = self.db_client.get_index_stocks('399101', today)
        S_stocks = self.filter_basic(S_stocks)
        if not S_stocks: return []
        
        fin_date = (self.get_next_trading_day() - pd.DateOffset(days=1)).date()
        fin_db = self.db_client.get_stock_fincial(S_stocks, fields=["roe", "roa", "total_shares"], date=fin_date)
        if fin_db.empty: return []
        
        df_prc = self.db_client.get_price(fin_db.index.to_list(), today, ["close"], 1, price_adj=False)
        if not df_prc.empty:
            if "date" in df_prc.index.names:
                df_prc.reset_index(level="date", drop=True, inplace=True)
            fin_db["close"] = df_prc["close"]
            fin_db["market_cap"] = fin_db["close"] * fin_db["total_shares"]
        else:
            fin_db["market_cap"] = 9e15

        mask = (fin_db["roe"] > 0.15) & (fin_db["roa"] > 0.10)
        L2_stocks = fin_db[mask].sort_values(by="market_cap", ascending=True).index.to_list()
        logger.info(f"get_L2: found {len(L2_stocks)} stocks")
        return L2_stocks

    def prepare_yesterday_high_limit(self):
        hold_list = list(self.market_env.account.positions[-1].keys())
        if not hold_list:
            self.yesterday_HL_list = []
            return
        
        # 'high_limit' is not in DB, using pctChg > 9.5% as proxy for limit up
        df = self.db_client.get_price(hold_list, self.get_current_date_str(), ["close", "pctChg"], 1)
        if df.empty:
            self.yesterday_HL_list = []
            return
        if "date" in df.index.names:
            df.reset_index(level="date", drop=True, inplace=True)
            
        # Approximation: if pctChg > 9.5 (assuming 10% limit), consider it a limit-up candidate
        # This is not perfect but unblocks execution without the explicit high_limit column
        hl_df = df[df['pctChg'] > 9.5]
        self.yesterday_HL_list = hl_df.index.to_list()

    def action_decider(self, stocks_obs):
        ts = self.market_env.cur_date
        logger.info(f"action_decider called for {ts}")
        if ts is None: return
        self.current_date = ts
        today_str = str(ts.date())

        if today_str not in self.trad_days.index.strftime('%Y-%m-%d'):
            logger.debug(f"action_decider: {today_str} not in trad_days index")
            return

        if self.trad_days.loc[today_str, "is_last_trading_day"] == 0:
            return
            
        logger.info(f"action_decider: processing rebalance for {today_str}")

        df_ratio = self.get_market_breadth(today_str)
        if df_ratio.empty: 
            logger.warning("action_decider: df_ratio empty")
            return
        
        # Group Mean Calculations
        JSG_rows = df_ratio[df_ratio['name'].isin(self.JSG_group)]
        XSZ_rows = df_ratio[df_ratio['name'].isin(self.XSZ_group)]
        CYB_rows = df_ratio[df_ratio['name'].isin(self.CYB_group)]
        OTHER_rows = df_ratio[~df_ratio['name'].isin(self.JSG_group | self.XSZ_group | self.CYB_group)]
        
        means = {
            'JSG': JSG_rows['ratio'].mean() if not JSG_rows.empty else 0,
            'XSZ': XSZ_rows['ratio'].mean() if not XSZ_rows.empty else 0,
            'CYB': CYB_rows['ratio'].mean() if not CYB_rows.empty else 0,
            'OTHER': OTHER_rows['ratio'].mean() if not OTHER_rows.empty else 0
        }
        max_group = max(means, key=means.get)
        max_mean = means[max_group]
        
        logger.info(f"Group Means: {means} | Max: {max_group}({max_mean})")
        
        # Environment Check
        I_top = df_ratio.nlargest(self.stock_sum, 'ratio')['name'].tolist()
        if any(item in self.black_industry_name for item in I_top) and self.judge_market_env() == '存量':
            logger.info("Market is '存量' and JSG dominating, clearing positions.")
            self.adjust([])
            return

        final_list = []
        if max_group == 'JSG' and max_mean > 90:
            L2 = self.get_L2(today_str)
            max_ind_code = JSG_rows['ratio'].idxmax() if not JSG_rows.empty else None
            L1_stocks = self.db_client.get_swindustry_stocks(max_ind_code, today_str) if max_ind_code else []
            L1 = self.get_L1(L1_stocks)
            final_list = L1[:1] + L2[:9]
        elif max_group == 'XSZ':
            final_list = self.get_L2(today_str)[:10]
        elif max_group == 'CYB':
            L2 = self.get_L2(today_str)
            final_list = L2[:9] + ['sz.159915'] 
        else:
            final_list = self.get_L2(today_str)[:10]

        self.adjust(final_list)
        logger.info(f"action_decider: final_list size: {len(final_list)}")

    def process_order_value(self, target_value):
        # Precise rebalancing logic
        pos_df = self.market_env.account.get_position_price(self.get_current_date_str())
        if not pos_df.empty:
            pos_df["value"] = pos_df["position"] * pos_df["close"]

        # Get latest prices for all targets
        new_price = self.db_client.get_price(
            list(target_value.keys()), self.get_current_date_str(), ["close"], 1
        )
        if not new_price.empty and "date" in (new_price.index.names if hasattr(new_price.index, 'names') else []):
            new_price.reset_index(level="date", drop=True, inplace=True)
            
        action_pos = {}
        for code, value in target_value.items():
            if not pos_df.empty and code in pos_df.index:
                new_pos = int(value / pos_df.loc[code, "close"])
                delta = new_pos - pos_df.loc[code, "position"]
                if delta != 0:
                    action_pos[code] = delta
            elif not new_price.empty and code in new_price.index:
                action_pos[code] = int(value / new_price.loc[code, "close"])

        # Execute SELL orders first to free up cash
        sell_orders = {k: v for k, v in action_pos.items() if v < 0}
        buy_orders = {k: v for k, v in action_pos.items() if v > 0}
        
        for code, action in sell_orders.items():
            self.create_order(code, action)
        for code, action in buy_orders.items():
            self.create_order(code, action)

    def adjust(self, stocks):
        target = stocks[: min(len(stocks), self.stock_sum)]
        hold_list = list(self.market_env.account.positions[-1].keys())
        target_value = {}
        
        # Sell stocks not in target (unless they are today's limit-up stocks, which we check in run_end)
        # Note: here we follow the original logic: close position if not in target
        for stock in hold_list:
            if stock not in target:
                target_value[stock] = 0

        total_value = self.market_env.account.get_total_value()
        if target:
            val_per_stock = total_value / len(target)
            for code in target:
                # Removed 'if code not in hold_list' check to allow rebalancing of existing positions
                target_value[code] = val_per_stock

        if target_value:
            self.process_order_value(target_value)

    def step(self):
        self.current_date = self.market_env.cur_date
        self.prepare_yesterday_high_limit()

    def run_end(self):
        # Limit-up break sell logic
        if not self.yesterday_HL_list: return
        # Using pctChg again for consistency check or simple price drop
        # Without high_limit, we can check if price dropped significantly from yesterday's close
        # But for now, let's skip the precise limit-break check or implement a simple threshold
        today_df = self.db_client.get_price(self.yesterday_HL_list, self.get_current_date_str(), ["close", "pctChg"], 1)
        if today_df.empty: return
        if "date" in today_df.index.names:
            today_df.reset_index(level="date", drop=True, inplace=True)
            
        for stock in self.yesterday_HL_list:
            if stock in today_df.index:
                # If it was limit up yesterday, and today pctChg is negative or low, we might sell
                # Original logic: close < high_limit. Since we don't have high_limit, 
                # we can't perfectly replicate "limit opened". 
                # Alternative: Check if pctChg < 9.5 implies it's not limit-up anymore
                if today_df.loc[stock, "pctChg"] < 9.0: # Arbitrary buffer below 10%
                    logger.info(f"[{stock}] Limit-up broke (pctChg < 9%), selling.")
                    self.create_order(stock, -MAX_POSITION, exec_time="close")

    def create_order(self, code, action, exec_time="open"):
        if action < 0:
            self.market_env.order_manager.create_order(code, "sell", abs(action), None, exec_time)
        else:
            self.market_env.order_manager.create_order(code, "buy", abs(action), None, exec_time)

if __name__ == "__main__":
    from db import DB
    from market_env import MultiMarketEnv
    db_client = DB()
    # Simple verification code...
