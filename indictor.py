def indictor_macd(
    df, short_window=12, long_window=26, signal_window=9, colums=["close"]
):
    for col in colums:
        # 计算短期EMA和长期EMA
        ema_short = df[col].ewm(span=short_window, adjust=False).mean()
        ema_long = df[col].ewm(span=long_window, adjust=False).mean()

        # 计算MACD值
        macd = ema_short - ema_long

        # 计算信号线
        signal_line = macd.ewm(span=signal_window, adjust=False).mean()

        # MACD Histogram，可选，代表MACD与信号线之间的差值
        df[f"macd_{col}"] = macd - signal_line


def indictor_force_index(df, ema_period=2, colums=[("close", "volume")]):
    for col, vol in colums:
        change = df[col].diff()
        force_index = change * df[vol]

        ema_period = 2

        # 计算13日EMA的Force Index
        force_index_ema = force_index.ewm(span=ema_period, adjust=False).mean()
        df[f"force_index_{col}"] = force_index_ema
