import pandas as pd
import time
from typing import Dict, Optional, List
from .base import DataStream, Bar
from datetime import datetime

class CSVDataStream(DataStream):
    def __init__(self, csv_files: Dict[str, str], start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        csv_files: Dict mapping symbol to filepath
        """
        self.data: Dict[str, pd.DataFrame] = {}
        for symbol, path in csv_files.items():
            # Check if file has header or not. Based on inspection, it doesn't.
            df = pd.read_csv(path, header=None)
            if len(df.columns) >= 6:
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'amount'][:len(df.columns)]
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            df = df.sort_values('timestamp')
            if start_date:
                df = df[df['timestamp'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['timestamp'] <= pd.to_datetime(end_date)]
            
            self.data[symbol] = df.reset_index(drop=True)
            
        self.idx = 0
        # Determine the union of all timestamps (or just use the first symbol if they are aligned)
        # Simplified for now: assume they are aligned by index
        self.max_idx = max(len(df) for df in self.data.values()) if self.data else 0

    def next_bar(self) -> Optional[Dict[str, Bar]]:
        if self.idx >= self.max_idx:
            return None
        
        bars = {}
        for symbol, df in self.data.items():
            if self.idx < len(df):
                row = df.iloc[self.idx]
                bars[symbol] = Bar(
                    symbol=symbol,
                    timestamp=row['timestamp'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row.get('volume', 0.0),
                    amount=row.get('amount', 0.0)
                )
        
        self.idx += 1
        return bars

    def reset(self):
        self.idx = 0

class DBDataStream(DataStream):
    def __init__(self, db_client, symbols: List[str], start_date: str, end_date: Optional[str] = None):
        self.db_client = db_client
        self.symbols = symbols
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()
        
        # Load master timeline (using the first symbol as reference or a market index)
        # This is lightweight compared to loading all columns for all stocks
        ref_symbol = symbols[0] if symbols else 'sh.000001'
        try:
            # We fetch just dates if possible, but get_kline fetches all. 
            # Optimization: In a real scenario, we'd add a get_trading_days method to DB.
            # For now, we assume fetching one symbol's full history is acceptable overhead 
            # compared to fetching ALL symbols' full history.
            ref_df = self.db_client.get_kline(ref_symbol, start_date, end_date)
            if 'date' in ref_df.columns:
                self.timestamps = pd.to_datetime(ref_df['date']).sort_values().unique().tolist()
            elif 'datetime' in ref_df.columns:
                self.timestamps = pd.to_datetime(ref_df['datetime']).sort_values().unique().tolist()
            else:
                self.timestamps = pd.to_datetime(ref_df.index).sort_values().unique().tolist()
        except Exception:
            # Fallback if reference symbol fails
            self.timestamps = pd.date_range(start=self.start_date, end=self.end_date, freq='B').tolist()

        self.total_bars = len(self.timestamps)
        self.global_idx = 0
        
        # Chunking
        self.chunk_size_years = 1
        self.current_chunk_data: Dict[str, pd.DataFrame] = {}
        self.current_chunk_start_idx = 0
        self.current_chunk_end_idx = 0
        
        self._load_next_chunk()

    def _load_next_chunk(self):
        if self.global_idx >= self.total_bars:
            self.current_chunk_data = {}
            return

        chunk_start_ts = self.timestamps[self.global_idx]
        # Determine chunk end date
        next_year = chunk_start_ts.year + self.chunk_size_years
        chunk_end_date_limit = chunk_start_ts.replace(year=next_year)
        
        # Find the index in self.timestamps that corresponds to this limit
        # We want to load enough data to cover [chunk_start_ts, chunk_end_date_limit)
        
        # Filter timestamps for this chunk
        chunk_timestamps = [t for t in self.timestamps if t >= chunk_start_ts and t < chunk_end_date_limit]
        
        if not chunk_timestamps:
            # Should not happen unless global_idx is at end
            return

        chunk_end_ts = chunk_timestamps[-1]
        
        # Update chunk indices relative to global timestamps
        self.current_chunk_start_idx = self.global_idx
        self.current_chunk_end_idx = self.global_idx + len(chunk_timestamps)
        
        start_str = chunk_start_ts.strftime("%Y-%m-%d")
        end_str = chunk_end_ts.strftime("%Y-%m-%d")
        
        # Load data for all symbols in this range
        self.current_chunk_data = {}
        for symbol in self.symbols:
            df = self.db_client.get_kline(symbol, start_str, end_str)
            if df.empty:
                continue
                
            df.columns = [c.lower() for c in df.columns]
            if 'datetime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['datetime'])
            elif 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
            else:
                df['timestamp'] = pd.to_datetime(df.index)
            
            if 'adjfactor' in df.columns:
                for col in ['open', 'high', 'low', 'close']:
                    if col in df.columns:
                        df[col] = df[col] * df['adjfactor']
            
            # Index by timestamp for faster lookup in next_bar
            self.current_chunk_data[symbol] = df.set_index('timestamp').sort_index()

    def next_bar(self) -> Optional[Dict[str, Bar]]:
        if self.global_idx >= self.total_bars:
            return None
            
        # Check if we need to load next chunk
        if self.global_idx >= self.current_chunk_end_idx:
            self._load_next_chunk()
            if self.global_idx >= self.total_bars: # Double check
                return None

        current_ts = self.timestamps[self.global_idx]
        bars = {}
        
        for symbol, df in self.current_chunk_data.items():
            if current_ts in df.index:
                row = df.loc[current_ts]
                # row might be a Series (single row) or DataFrame (duplicate timestamps)
                # handle duplicate timestamps if necessary, assume Series
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                    
                bars[symbol] = Bar(
                    symbol=symbol,
                    timestamp=current_ts,
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row.get('volume', 0.0),
                    amount=row.get('amount', 0.0),
                    extra={k: v for k, v in row.items() if k not in ['open', 'high', 'low', 'close', 'volume', 'amount']}
                )
        
        self.global_idx += 1
        # expose idx for progress tracking (mimicking old interface)
        self.idx = self.global_idx 
        return bars

    def reset(self):
        self.global_idx = 0
        self.idx = 0
        self._load_next_chunk()

class RealtimeDataStream(DataStream):
    def __init__(self, symbols: List[str], interval_seconds: int = 60):
        self.symbols = symbols
        self.interval_seconds = interval_seconds
        self.last_fetch_time = time.time()
        import akshare as ak
        self.ak = ak

    def next_bar(self) -> Optional[Dict[str, Bar]]:
        # Block until interval passes
        time_to_wait = self.interval_seconds - (time.time() - self.last_fetch_time)
        if time_to_wait > 0:
            time.sleep(time_to_wait)
        
        self.last_fetch_time = time.time()
        current_ts = datetime.now()
        
        # Simple trading hours check (China A-share)
        # 09:30 - 11:30, 13:00 - 15:00
        # If outside, we might still return data or wait?
        # For simplicity, we just fetch.
        
        bars = {}
        try:
            # Efficient: fetch all spot data once and filter
            # ak.fund_etf_spot_em() is for ETFs. 
            # We need to support both stocks and ETFs? 
            # Assuming ETFs for now based on '510880' in examples.
            # Or use stock_zh_a_spot_em() for stocks.
            
            # Using stock_zh_a_spot_em for broader coverage or fund_etf_spot_em depending on symbol
            # This part is tricky without knowing exact symbol types.
            # We'll try fund_etf_spot_em first as in market_env.py
            
            df = self.ak.fund_etf_spot_em()
            # Columns: 代码, 名称, 最新价, ...
            # Map to symbols
            
            for symbol in self.symbols:
                # Symbol format expected: '510880' or 'sh.510880'?
                # DBDataStream uses raw code often.
                # Remove prefix if present
                code = symbol.split('.')[-1]
                
                row = df[df['代码'] == code]
                if not row.empty:
                    data = row.iloc[0]
                    bars[symbol] = Bar(
                        symbol=symbol,
                        timestamp=current_ts,
                        open=float(data['开盘价']),
                        high=float(data['最高价']),
                        low=float(data['最低价']),
                        close=float(data['最新价']),
                        volume=float(data['成交量']),
                        amount=float(data['成交额']),
                        extra={"name": data['名称']}
                    )
        except Exception as e:
            print(f"Realtime fetch error: {e}")
            # Don't crash, just return empty or retry?
            # Return empty dict means no new data this step
            pass
            
        return bars if bars else {}

    def reset(self):
        pass
