import sys
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.market_data.db import DB


def _today_yyyymmdd() -> str:
    return datetime.now().strftime("%Y%m%d")


def _days_ago_yyyymmdd(days: int) -> str:
    return (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")


def _chunk_list(items: List[str], chunk_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _select_volume_series(df: pd.DataFrame) -> pd.Series:
    # Prefer "volume". Fallback to "vol". If neither, use monetary "amount".
    for candidate in ("volume", "vol"):
        if candidate in df.columns:
            return df[candidate]
    if "amount" in df.columns:
        return df["amount"]
    # As a last resort, create a near-constant tiny volume to avoid division by zero downstream
    return pd.Series(np.ones(len(df)) * 1e-6, index=df.index)


def _compute_mas(close: pd.Series, periods: List[int]) -> pd.DataFrame:
    ma_df = pd.DataFrame(index=close.index)
    for p in periods:
        ma_df[f"ma{p}"] = close.rolling(window=p, min_periods=max(3, p // 2)).mean()
    return ma_df


def _check_bull_alignment(ma_df: pd.DataFrame, as_of_date: pd.Timestamp) -> bool:
    # Require strict ordering on the last available day for 30/60/120 MAs
    if as_of_date not in ma_df.index:
        return False
    latest = ma_df.loc[as_of_date]
    required_cols = [c for c in ("ma30", "ma60", "ma120") if c in ma_df.columns]
    if len(required_cols) < 3:
        return False
    ordered = latest["ma30"] > latest["ma60"] > latest["ma120"]
    if not ordered:
        return False
    # Slope checks: each MA should be higher than N days ago
    lookbacks = {"ma30": 5, "ma60": 10, "ma120": 20}
    for col, lb in lookbacks.items():
        hist_idx = ma_df.index.get_loc(as_of_date)
        if hist_idx - lb < 0:
            return False
        past_val = ma_df.iloc[hist_idx - lb][col]
        if np.isnan(past_val) or np.isnan(latest[col]):
            return False
        if latest[col] <= past_val:
            return False
    return True


def _check_bottom_stabilization(df: pd.DataFrame, as_of_date: pd.Timestamp) -> bool:
    # Heuristic approximation of bottom stabilization:
    # - No significant new low in recent 60 trading days vs prior 180 days (<= -5%)
    # - 60d price range is relatively tight (<= 35% of prior 180d range)
    # - Volume contraction in recent 60d relative to prior 180d (<= 85%)
    if len(df) < 260:
        return False
    if as_of_date not in df.index:
        return False
    close = df["close"].copy()
    volume = _select_volume_series(df)

    # Indices windows
    end_loc = df.index.get_loc(as_of_date)
    if end_loc < 240:
        return False
    recent_start = max(0, end_loc - 60)
    prior_start = max(0, end_loc - 240)
    prior_end = max(prior_start, end_loc - 61)

    recent_slice = slice(recent_start, end_loc + 1)
    prior_slice = slice(prior_start, prior_end + 1)

    recent_close = close.iloc[recent_slice]
    prior_close = close.iloc[prior_slice]
    if len(prior_close) == 0 or len(recent_close) == 0:
        return False

    recent_min = float(np.nanmin(recent_close))
    prior_min = float(np.nanmin(prior_close))
    if np.isnan(recent_min) or np.isnan(prior_min) or prior_min <= 0:
        return False
    # No material lower low recently
    if recent_min < prior_min * 0.95:
        return False

    # Consolidation: recent range is relatively tighter
    recent_range = float(np.nanmax(recent_close) - np.nanmin(recent_close))
    prior_range = float(np.nanmax(prior_close) - np.nanmin(prior_close))
    if np.isnan(recent_range) or np.isnan(prior_range) or prior_range == 0:
        return False
    if recent_range > prior_range * 0.35:
        return False

    # Volume contraction
    recent_vol_mean = float(np.nanmean(volume.iloc[recent_slice]))
    prior_vol_mean = float(np.nanmean(volume.iloc[prior_slice]))
    if np.isnan(recent_vol_mean) or np.isnan(prior_vol_mean) or prior_vol_mean <= 0:
        return False
    if recent_vol_mean > prior_vol_mean * 0.85:
        return False

    return True


def _check_volume_expansion(df: pd.DataFrame, as_of_date: pd.Timestamp) -> bool:
    # Volume expansion after bottom: 5d > 1.4 * 20d and 20d >= 0.9 * 60d
    if as_of_date not in df.index:
        return False
    volume = _select_volume_series(df)
    idx = df.index.get_loc(as_of_date)
    if idx < 60:
        return False
    v5 = float(np.nanmean(volume.iloc[idx - 4 : idx + 1]))
    v20 = float(np.nanmean(volume.iloc[idx - 19 : idx + 1]))
    v60 = float(np.nanmean(volume.iloc[idx - 59 : idx + 1]))
    if min(v5, v20, v60) <= 0 or any(map(np.isnan, (v5, v20, v60))):
        return False
    if not (v5 > 1.4 * v20 and v20 >= 0.9 * v60):
        return False

    # Price confirmation: close near 60d high (within 5%)
    close = df["close"]
    recent_high_60 = float(np.nanmax(close.iloc[idx - 59 : idx + 1]))
    last_close = float(close.iloc[idx])
    if np.isnan(recent_high_60) or np.isnan(last_close) or recent_high_60 <= 0:
        return False
    if last_close < 0.95 * recent_high_60:
        return False
    return True


def _screen_single(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    as_of_date = df.index[-1]
    # Bottom stabilization
    if not _check_bottom_stabilization(df, as_of_date):
        return False
    # MA bull alignment + slopes
    ma_periods = [30, 60, 120]
    ma_df = _compute_mas(df["close"], ma_periods)
    if not _check_bull_alignment(ma_df, as_of_date):
        return False
    # Volume expansion confirmation
    if not _check_volume_expansion(df, as_of_date):
        return False
    return True


def _compute_float_mcap_yi(
    last_close_by_code: Dict[str, float],
    fin_df: pd.DataFrame,
) -> pd.Series:
    # fin_df indexed by "code" with at least column "circulating_a"
    if fin_df is None or fin_df.empty or "circulating_a" not in fin_df.columns:
        return pd.Series(dtype=float)
    codes = []
    values = []
    for code, row in fin_df.iterrows():
        if code not in last_close_by_code:
            continue
        last_close = last_close_by_code[code]
        circ_a = row.get("circulating_a")
        try:
            circ_a = float(circ_a)
            last_close = float(last_close)
        except Exception:
            continue
        if np.isnan(circ_a) or np.isnan(last_close) or circ_a <= 0 or last_close <= 0:
            continue
        # Convert to Yi (1e8)
        mcap_yi = last_close * circ_a / 1e8
        codes.append(code)
        values.append(mcap_yi)
    return pd.Series(values, index=codes, name="float_mcap_yi")


def screen_stocks(
    end_date: str | None = None,
    lookback_trading_days: int = 800,
    chunk_size: int = 200,
    market_cap_threshold_yi: float = 300.0,
) -> pd.DataFrame:
    db = DB()
    if end_date is None:
        end_date = _today_yyyymmdd()
    # Approx 3 calendar years ~ 3 * 365 = 1095 days; 800 trading days is usually sufficient
    start_date = _days_ago_yyyymmdd(days=3 * 365 + 30)

    all_codes = db.get_all_stock_code()
    passed_codes: List[str] = []
    last_close_map: Dict[str, float] = {}

    for batch in _chunk_list(all_codes, chunk_size):
        df = db.get_price(
            stocks=batch,
            end_date=end_date,
            fields=["close", "open", "high", "low", "volume", "amount"],
            count=lookback_trading_days,
            start_date=start_date,
        )
        if df is None or len(df) == 0:
            continue
        # Ensure MultiIndex and per-code order
        df = df.sort_index()

        for code, code_df in df.groupby(level=0):
            code_df = code_df.droplevel(0)
            # Keep last close for later market cap computation
            try:
                last_close_map[code] = float(code_df["close"].iloc[-1])
            except Exception:
                continue
            try:
                if _screen_single(code_df):
                    passed_codes.append(code)
            except Exception:
                # Be robust to any per-code issues
                continue

    if not passed_codes:
        return pd.DataFrame(columns=["code", "last_close", "float_mcap_yi"]).set_index("code")

    # Market cap filter using circulating A shares
    fin_df = DB().get_stock_fincial(passed_codes, fields=["circulating_a"], date=end_date)
    mcap_series = _compute_float_mcap_yi(last_close_map, fin_df)
    if mcap_series.empty:
        # If we cannot compute, return raw list
        result = pd.DataFrame(index=passed_codes)
        result.index.name = "code"
        result["last_close"] = [last_close_map.get(c, np.nan) for c in result.index]
        result["float_mcap_yi"] = np.nan
        return result

    # Combine and filter threshold
    result = mcap_series.to_frame()
    result["last_close"] = [last_close_map.get(c, np.nan) for c in result.index]
    result = result[result["float_mcap_yi"] < market_cap_threshold_yi]
    return result.sort_values(by=["float_mcap_yi"])  # smaller first


def _compute_forward_returns(close: pd.Series, idx: int, horizons: List[int]) -> Dict[str, float]:
    results: Dict[str, float] = {}
    if idx < 0 or idx >= len(close):
        return results
    base = float(close.iloc[idx])
    if np.isnan(base) or base <= 0:
        return results
    for h in horizons:
        if idx + h < len(close):
            future = float(close.iloc[idx + h])
            if not np.isnan(future) and future > 0:
                results[f"ret_{h}d"] = future / base - 1.0
    return results


def scan_historical_signals(
    end_date: str | None = None,
    lookback_trading_days: int = 800,
    chunk_size: int = 160,
    market_cap_threshold_yi: float = 300.0,
    horizons: List[int] = [5, 10, 20],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    db = DB()
    if end_date is None:
        end_date = _today_yyyymmdd()
    start_date = _days_ago_yyyymmdd(days=3 * 365 + 30)

    all_codes = db.get_all_stock_code()
    signal_rows: List[Dict] = []

    for batch in _chunk_list(all_codes, chunk_size):
        df = db.get_price(
            stocks=batch,
            end_date=end_date,
            fields=["close", "open", "high", "low", "volume", "amount"],
            count=lookback_trading_days,
            start_date=start_date,
        )
        if df is None or len(df) == 0:
            continue
        df = df.sort_index()

        for code, code_df in df.groupby(level=0):
            code_df = code_df.droplevel(0)
            if len(code_df) < 260:
                continue
            ma_df = _compute_mas(code_df["close"], [30, 60, 120])
            # scan all dates, skipping earliest warmup
            for i, as_of_date in enumerate(code_df.index):
                # Require enough history for checks and forward horizons
                if i < 240 or i + min(horizons) >= len(code_df):
                    continue
                try:
                    if not _check_bottom_stabilization(code_df.iloc[: i + 1], as_of_date):
                        continue
                    if not _check_bull_alignment(ma_df.iloc[: i + 1], as_of_date):
                        continue
                    if not _check_volume_expansion(code_df.iloc[: i + 1], as_of_date):
                        continue
                except Exception:
                    continue

                row: Dict[str, object] = {
                    "code": code,
                    "date": as_of_date,
                    "close": float(code_df["close"].iloc[i]) if not np.isnan(code_df["close"].iloc[i]) else np.nan,
                }
                # forward returns
                row.update(_compute_forward_returns(code_df["close"], i, horizons))
                signal_rows.append(row)

    if not signal_rows:
        empty_detail = pd.DataFrame(columns=["code", "date", "close"] + [f"ret_{h}d" for h in horizons])
        empty_detail["float_mcap_yi"] = []
        empty_detail.set_index(["code", "date"], inplace=True)
        return empty_detail, pd.DataFrame()

    detail = pd.DataFrame(signal_rows)
    # Normalize dates to string YYYY-MM-DD for joins
    detail["date_str"] = pd.to_datetime(detail["date"]).dt.strftime("%Y-%m-%d")

    # Compute market cap per (code, date) and filter
    filtered_rows: List[pd.Series] = []
    for date_str, sub in detail.groupby("date_str"):
        codes = sub["code"].tolist()
        try:
            fin = db.get_stock_fincial(codes, fields=["circulating_a"], date=date_str)
        except Exception:
            fin = pd.DataFrame()
        fin = fin if isinstance(fin, pd.DataFrame) else pd.DataFrame()
        circ_map = fin["circulating_a"].to_dict() if "circulating_a" in fin.columns else {}
        for _, r in sub.iterrows():
            circ_a = circ_map.get(r["code"], np.nan)
            close = r["close"]
            if not (isinstance(close, (int, float)) and isinstance(circ_a, (int, float))):
                continue
            if np.isnan(close) or np.isnan(circ_a) or close <= 0 or circ_a <= 0:
                continue
            mcap_yi = close * float(circ_a) / 1e8
            if mcap_yi < market_cap_threshold_yi:
                rr = r.copy()
                rr["float_mcap_yi"] = mcap_yi
                filtered_rows.append(rr)

    if not filtered_rows:
        empty_detail = pd.DataFrame(columns=["code", "date", "close"] + [f"ret_{h}d" for h in horizons] + ["float_mcap_yi"])
        empty_detail.set_index(["code", "date"], inplace=True)
        return empty_detail, pd.DataFrame()

    detail_filtered = pd.DataFrame(filtered_rows)
    detail_filtered.sort_values(["date", "code"], inplace=True)
    detail_filtered.set_index(["code", "date"], inplace=True)

    # Aggregate statistics across all signals
    stat_rows: List[Dict[str, float]] = []
    agg: Dict[str, float] = {}
    for h in horizons:
        col = f"ret_{h}d"
        if col in detail_filtered.columns:
            series = detail_filtered[col].dropna()
            if len(series) > 0:
                agg[f"count_{h}d"] = int(series.count())
                agg[f"mean_{h}d"] = float(series.mean())
                agg[f"median_{h}d"] = float(series.median())
                agg[f"win_rate_{h}d"] = float((series > 0).mean())
    if agg:
        stat_rows.append(agg)
    stats_df = pd.DataFrame(stat_rows)

    return detail_filtered, stats_df


def main():
    end_date = None
    if len(sys.argv) >= 2 and sys.argv[1]:
        end_date = sys.argv[1]

    mode = "current"
    if len(sys.argv) >= 3 and sys.argv[2]:
        mode = sys.argv[2]

    if mode == "hist":
        detail, stats = scan_historical_signals(end_date=end_date)
        detail_path = "historical_signals.csv"
        stats_path = "historical_stats.csv"
        detail.to_csv(detail_path)
        stats.to_csv(stats_path, index=False)
        print(f"Historical signals: {len(detail)} rows -> {detail_path}")
        print(f"Aggregated stats -> {stats_path}")
    else:
        df = screen_stocks(end_date=end_date)
        output_path = "filtered_stocks.csv"
        df.to_csv(output_path)
        print(f"Filtered {len(df)} stocks written to {output_path}")


if __name__ == "__main__":
    main()

