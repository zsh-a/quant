import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from src.alpha_mining.operators import SAFE_LOCALS
from loguru import logger

class AlphaEvaluator:
    def __init__(self, data: pd.DataFrame, train_ratio: float = 0.6, val_ratio: float = 0.2):
        """
        Args:
            data: DataFrame with columns [date, symbol, open, high, low, close, volume, amount]
            train_ratio: Ratio of training data (default 0.6)
            val_ratio: Ratio of validation data (default 0.2), test = 1 - train - val
        """
        # Ensure MultiIndex for internal storage
        self.raw_data = data.copy()
        if 'symbol' in self.raw_data.columns and 'date' in self.raw_data.columns:
            self.raw_data = self.raw_data.set_index(['date', 'symbol']).sort_index()

        self._cache = {}

        # Pre-calculate matrix format data for fast vectorized eval
        self.context = {}
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            if col in data.columns:
                self.context[col] = data.pivot(index='date', columns='symbol', values=col)

        # Derived matrix fields
        if 'amount' in self.context and 'volume' in self.context:
            self.context['vwap'] = self.context['amount'] / (self.context['volume'] + 1e-9)

        self.context.update(SAFE_LOCALS)

        # Pre-calculate Forward Returns (Target) for multiple periods
        close_matrix = self.context['close']
        self.forward_returns_1d = close_matrix.shift(-1) / close_matrix - 1.0
        self.forward_returns_5d = close_matrix.shift(-5) / close_matrix - 1.0  # 1周后收益（主要）
        self.forward_returns_10d = close_matrix.shift(-10) / close_matrix - 1.0

        # Time-series split for train/val/test
        dates = sorted(self.context['close'].index)
        n_dates = len(dates)
        train_end_idx = int(n_dates * train_ratio)
        val_end_idx = int(n_dates * (train_ratio + val_ratio))

        self.train_dates = dates[:train_end_idx]
        self.val_dates = dates[train_end_idx:val_end_idx]
        self.test_dates = dates[val_end_idx:]

        logger.info(f"Evaluator initialized: {len(self.context['close'].columns)} symbols")
        logger.info(f"Train: {len(self.train_dates)} days, Val: {len(self.val_dates)} days, Test: {len(self.test_dates)} days")

        # Warning if data is insufficient
        if len(dates) < 1250:  # Less than 5 years
            logger.warning(f"Only {len(dates)} trading days available. Recommend at least 1250 days (5 years) to reduce overfitting risk.")

    def evaluate(self, formula: str, mode: str = 'train') -> Dict[str, Any]:
        """
        Evaluate alpha factor on specified dataset split

        Args:
            formula: Alpha formula string
            mode: 'train', 'val', 'test', or 'all'
        """
        cache_key = f"{formula}_{mode}"
        if cache_key in self._cache:
            logger.debug(f"      [Eval] Cache hit for formula (mode={mode})")
            return self._cache[cache_key]

        logger.info(f"      [Eval] Evaluating formula on {mode} set: {formula[:60]}...")
        try:
            # 1. Evaluate Formula (Vectorized across all stocks and dates)
            logger.info(f"      [Eval] Executing formula evaluation...")
            factor_matrix = eval(formula, {"__builtins__": {}}, self.context)
            logger.info(f"      [Eval] Formula execution complete. Shape: {factor_matrix.shape}")

            # 2. Handle extreme values and NaN
            logger.info(f"      [Eval] Cleaning extreme values and NaN...")
            factor_matrix = factor_matrix.replace([np.inf, -np.inf], np.nan)
            # Winsorize at 1% and 99% percentiles
            lower = factor_matrix.quantile(0.01)
            upper = factor_matrix.quantile(0.99)
            factor_matrix = factor_matrix.clip(lower=lower, upper=upper, axis=1)

            # 3. Select date range based on mode
            if mode == 'train':
                dates = self.train_dates
            elif mode == 'val':
                dates = self.val_dates
            elif mode == 'test':
                dates = self.test_dates
            else:  # 'all'
                dates = factor_matrix.index

            logger.info(f"      [Eval] Selected {len(dates)} dates for {mode} mode")

            # 4. Align with target (use 5-day forward return as primary target - 一周收益)
            common_dates = factor_matrix.index.intersection(self.forward_returns_5d.index).intersection(dates)
            logger.info(f"      [Eval] Common dates after alignment: {len(common_dates)}")

            if len(common_dates) < 20:  # Minimum days requirement
                logger.warning(f"      [Eval] Insufficient valid dates: {len(common_dates)}")
                return {'rank_ic': 0.0, 'ic_ir': 0.0, 'error': 'Insufficient valid dates'}

            f = factor_matrix.loc[common_dates]
            r1d = self.forward_returns_1d.loc[common_dates]
            r5d = self.forward_returns_5d.loc[common_dates]
            r10d = self.forward_returns_10d.loc[common_dates]

            # 5. Calculate cross-sectional RankIC per date (主要使用5日收益)
            logger.info(f"      [Eval] Calculating RankIC correlations (5-day forward return as primary)...")
            rank_ic_5d = f.corrwith(r5d, axis=1, method='spearman').dropna()  # 主要指标
            rank_ic_1d = f.corrwith(r1d, axis=1, method='spearman').dropna()
            rank_ic_10d = f.corrwith(r10d, axis=1, method='spearman').dropna()
            logger.info(f"      [Eval] RankIC calculation complete. Valid ICs: 5d={len(rank_ic_5d)}, 1d={len(rank_ic_1d)}, 10d={len(rank_ic_10d)}")

            if rank_ic_5d.empty:
                return {'rank_ic': 0.0, 'ic_ir': 0.0, 'error': 'No valid IC values'}

            # 6. IC Decay Analysis (check overfitting) - 5日到10日的衰减
            ic_decay = abs(rank_ic_5d.mean()) - abs(rank_ic_10d.mean()) if not rank_ic_10d.empty else 0

            # 7. Turnover Estimation (factor autocorrelation)
            factor_autocorr = f.corrwith(f.shift(1), axis=1, method='spearman').mean()
            turnover_proxy = 1 - abs(factor_autocorr) if not np.isnan(factor_autocorr) else 1.0

            # 8. Calculate comprehensive metrics (使用5日收益作为主要指标)
            logger.info(f"      [Eval] Computing final metrics (5-day forward return)...")
            rank_ic_mean = rank_ic_5d.mean()  # 主要使用5日IC
            rank_ic_std = rank_ic_5d.std()
            ic_ir = rank_ic_mean / (rank_ic_std + 1e-9)

            # Fixed fitness calculation: rank_ic^2 / std (keep sign information in rank_ic)
            fitness = (rank_ic_mean ** 2) / (rank_ic_std + 1e-9)

            metrics = {
                'rank_ic': rank_ic_mean,  # 这是5日IC
                'ic_ir': ic_ir,
                'ic_std': rank_ic_std,
                'rank_ic_1d': rank_ic_1d.mean() if not rank_ic_1d.empty else 0.0,
                'rank_ic_5d': rank_ic_5d.mean(),  # 冗余但保留兼容性
                'rank_ic_10d': rank_ic_10d.mean() if not rank_ic_10d.empty else 0.0,
                'ic_decay': ic_decay,
                'turnover_proxy': turnover_proxy,
                'fitness': fitness,
                'n_valid_dates': len(common_dates)
            }

            # Clean metrics for JSON serialization
            metrics = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                      for k, v in metrics.items()}

            self._cache[cache_key] = metrics
            logger.info(f"      [Eval] Evaluation complete. RankIC: {rank_ic_mean:.4f}, IR: {ic_ir:.4f}")
            return metrics

        except Exception as e:
            logger.error(f"Error evaluating {formula}: {e}")
            return {'rank_ic': 0.0, 'ic_ir': 0.0, 'error': str(e)}