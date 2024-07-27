import pandas as pd
import numpy as np
import talib as ta
from loguru import logger
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
    def __init__(self, db_client, stock_sum=10):
        super().__init__()
        self.db_client = db_client
        self.stock_sum = stock_sum
        
        self.JSG_group = {'银行I', '有色金属I', '钢铁I', '煤炭I'}
        self.XSZ_group = {'小市值200'}
        self.CYB_group = {'创业板50'}
        self.black_industry_name = {"银行I", "煤炭I", "采掘I", "钢铁I"}
        
        self.trad_days = pd.read_csv(
            "marked_trade_datas.csv", index_col="calendar_date", parse_dates=True
        )

    def on_bar(self, bars: dict[str, Bar]):
        if not bars: return
        ts = next(iter(bars.values())).timestamp
        today_str = str(ts.date())

        if today_str not in self.trad_days.index.strftime('%Y-%m-%d'):
            return

        if self.trad_days.loc[today_str, "is_last_trading_day"] == 0:
            return

        logger.info(f"RotationStrategy: Rebalancing on {today_str}")

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
            final_list = L2[:9] + ['sz.159915'] 
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
        
        logger.info(f"Rebalancing: Target {target}, Holding {hold_list}")
        
        # 1. Sell stocks not in target
        for stock in hold_list:
            if stock not in target:
                qty = account['positions'][stock]
                if qty > 0:
                    logger.info(f"Selling {stock} (qty: {qty}) because not in target")
                    self.sell(stock, qty)

        # 2. Buy/Rebalance target stocks
        total_equity = account['total_equity']
        if target:
            val_per_stock = (total_equity * 0.95) / len(target) # 5% cash buffer
            for code in target:
                price_df = self.db_client.get_price(code, date_str, ["close"], 1)
                if price_df.empty:
                    logger.warning(f"Could not get price for {code} on {date_str}")
                    continue
                price = price_df.iloc[0]["close"]
                target_qty = int(val_per_stock / price // 100 * 100)
                
                curr_qty = account['positions'].get(code, 0)
                if target_qty > curr_qty:
                    logger.info(f"Submitting BUY for {code}: {target_qty - curr_qty} shares")
                    self.buy(code, target_qty - curr_qty)
                elif target_qty < curr_qty:
                    logger.info(f"Submitting SELL for {code}: {curr_qty - target_qty} shares")
                    self.sell(code, curr_qty - target_qty)
