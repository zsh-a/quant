import pandas as pd
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
        self.start_date = start_date
        self.end_date = end_date
        
        # Load all data for symbols upfront for performance in backtest
        self.data: Dict[str, pd.DataFrame] = {}
        all_timestamps = set()
        
        for symbol in symbols:
            df = self.db_client.get_kline(symbol, start_date, end_date)
            # Standardize columns
            df.columns = [c.lower() for c in df.columns]
            if 'datetime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['datetime'])
            elif 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
            else:
                df['timestamp'] = pd.to_datetime(df.index)
            
            # Apply adjustment factor if present (from DBDataSource logic)
            if 'adjfactor' in df.columns:
                for col in ['open', 'high', 'low', 'close']:
                    if col in df.columns:
                        df[col] = df[col] * df['adjfactor']
            
            df = df.sort_values('timestamp')
            self.data[symbol] = df.reset_index(drop=True)
            all_timestamps.update(df['timestamp'].tolist())
            
        self.timestamps = sorted(list(all_timestamps))
        self.idx = 0

    def next_bar(self) -> Optional[Dict[str, Bar]]:
        if self.idx >= len(self.timestamps):
            return None
            
        current_ts = self.timestamps[self.idx]
        bars = {}
        
        for symbol, df in self.data.items():
            # Find the row with current timestamp
            # This can be optimized with a pointer if data is perfectly aligned
            mask = df['timestamp'] == current_ts
            if mask.any():
                row = df[mask].iloc[0]
                bars[symbol] = Bar(
                    symbol=symbol,
                    timestamp=row['timestamp'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row.get('volume', 0.0),
                    amount=row.get('amount', 0.0),
                    extra={k: row[k] for k in df.columns if k not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'amount']}
                )
        
        self.idx += 1
        return bars

    def reset(self):
        self.idx = 0
