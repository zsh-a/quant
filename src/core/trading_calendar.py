"""统一交易日历 — 调仓日判断的唯一入口。

用法:
    cal = TradingCalendar(db_client)
    cal.is_rebalance_day("2025-11-28", freq="weekly")   # True (周五)
    cal.is_rebalance_day("2025-12-31", freq="monthly")  # True (月末)
    cal.is_rebalance_day("2025-11-28", freq="biweekly", last_rebalance="2025-11-14")  # True

支持频率: daily, weekly, biweekly, monthly
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.market_data.db import DB

# 全局单例缓存，避免多个策略重复查询
_calendar_cache: pd.DataFrame | None = None


class TradingCalendar:
    """交易日历，从 ClickHouse 加载并缓存。"""

    def __init__(self, db_client: DB):
        self._df = self._load(db_client)

    @staticmethod
    def _load(db_client: DB) -> pd.DataFrame:
        global _calendar_cache
        if _calendar_cache is not None:
            return _calendar_cache

        data = db_client.client.query(
            "SELECT calendar_date, is_trading_day "
            "FROM stock_data.trade_dates "
            "ORDER BY calendar_date"
        )
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        df["calendar_date"] = pd.to_datetime(df["calendar_date"])
        df["is_trading_day"] = df["is_trading_day"].astype(int)
        df.set_index("calendar_date", inplace=True)

        trading = df[df["is_trading_day"] == 1]

        # 每周最后一个交易日
        weekly = trading.groupby(trading.index.to_period("W")).apply(lambda g: g.index.max())
        df["week_last"] = 0
        df.loc[df.index.isin(weekly.values), "week_last"] = 1

        # 每月最后一个交易日
        monthly = trading.groupby(trading.index.to_period("M")).apply(lambda g: g.index.max())
        df["month_last"] = 0
        df.loc[df.index.isin(monthly.values), "month_last"] = 1

        _calendar_cache = df
        return df

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def is_trading_day(self, date_str: str) -> bool:
        """判断是否为交易日。"""
        try:
            return int(self._df.loc[date_str, "is_trading_day"]) == 1
        except KeyError:
            return False

    def is_rebalance_day(
        self,
        date_str: str,
        freq: str = "weekly",
        last_rebalance: str | None = None,
    ) -> bool:
        """判断是否为调仓日。

        Args:
            date_str: 当前日期 "YYYY-MM-DD"
            freq: "daily" | "weekly" | "biweekly" | "monthly"
            last_rebalance: 上次调仓日期 (biweekly 模式需要)
        """
        if not self.is_trading_day(date_str):
            return False

        if freq == "daily":
            return True

        if freq == "weekly":
            return int(self._df.loc[date_str, "week_last"]) == 1

        if freq == "monthly":
            return int(self._df.loc[date_str, "month_last"]) == 1

        if freq == "biweekly":
            # 月末一定调仓
            if int(self._df.loc[date_str, "month_last"]) == 1:
                return True
            # 否则看是否为周末且距上次 >= 10 天
            if int(self._df.loc[date_str, "week_last"]) == 1:
                if last_rebalance is None:
                    return True
                days_since = (pd.Timestamp(date_str) - pd.Timestamp(last_rebalance)).days
                return days_since >= 10
            return False

        raise ValueError(f"Unknown rebalance freq: {freq!r}")

    def get_trading_days(self) -> pd.DatetimeIndex:
        """返回所有交易日。"""
        return self._df[self._df["is_trading_day"] == 1].index

    def nearest_trading_day(self, date_str: str, direction: str = "backward") -> str | None:
        """找最近的交易日。direction: 'backward' 或 'forward'。"""
        trading = self.get_trading_days()
        ts = pd.Timestamp(date_str)
        if direction == "backward":
            candidates = trading[trading <= ts]
            return candidates[-1].strftime("%Y-%m-%d") if len(candidates) > 0 else None
        else:
            candidates = trading[trading >= ts]
            return candidates[0].strftime("%Y-%m-%d") if len(candidates) > 0 else None

    @property
    def df(self) -> pd.DataFrame:
        """兼容旧代码: 返回底层 DataFrame (indexed by calendar_date)。

        含列: is_trading_day, week_last, month_last
        为了向后兼容, 也提供 is_last_trading_day 别名。
        """
        return self._df
