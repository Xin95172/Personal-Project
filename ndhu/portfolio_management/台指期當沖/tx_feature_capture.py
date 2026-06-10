from __future__ import annotations

from pathlib import Path

import pandas as pd


def _normalize_date(value: str | pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _build_foreign_opt_signal_a(raw_night: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "foreign_long_call_amount",
        "foreign_short_call_amount",
        "foreign_long_put_amount",
        "foreign_short_put_amount",
    ]
    if raw_night.empty:
        return pd.DataFrame(columns=["Foreign_Opt_Signal_a"])

    missing = [col for col in columns if col not in raw_night.columns]
    if missing:
        raise ValueError(f"Missing night option columns: {missing}")

    df = raw_night.copy()
    df.index = pd.to_datetime(df.index)

    net_call = df["foreign_long_call_amount"] - df["foreign_short_call_amount"]
    net_put = df["foreign_long_put_amount"] - df["foreign_short_put_amount"]
    turnover = (
        df["foreign_long_call_amount"]
        + df["foreign_short_call_amount"]
        + df["foreign_long_put_amount"]
        + df["foreign_short_put_amount"]
    ).replace(0, 1)

    signal = pd.DataFrame(index=df.index)
    signal["Foreign_Opt_Signal_a"] = (net_call - net_put) / turnover
    return signal


def _tradingview_ohlc(symbol: str, exchange: str, n_bars: int) -> pd.DataFrame:
    from tvDatafeed import Interval, TvDatafeed

    tv = TvDatafeed()
    hist = tv.get_hist(
        symbol=symbol,
        exchange=exchange,
        interval=Interval.in_daily,
        n_bars=n_bars,
    )
    if hist is None or hist.empty:
        raise RuntimeError(f"TradingView returned no data for {exchange}:{symbol}")

    hist = hist.copy()
    hist.index = pd.to_datetime(hist.index).normalize()
    return hist[["open", "high", "low", "close"]].sort_index()


def capture_tx_day_features(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    note_root: str | Path,
    output_path: str | Path | None = None,
    tv_n_bars: int = 3000,
) -> pd.DataFrame:
    """Capture day-session TX features used by the day-trade notebook."""

    import sys

    note_root = Path(note_root)
    if str(note_root) not in sys.path:
        sys.path.append(str(note_root))

    from module.get_info_FinMind import FinMindClient
    from module.get_info_TWSE import GetInfoTWSE

    start_str = _normalize_date(start)
    end_str = _normalize_date(end)
    start_ts = pd.Timestamp(start_str)
    end_ts = pd.Timestamp(end_str)
    existing = None
    if output_path is not None and Path(output_path).exists():
        existing = pd.read_parquet(output_path)
        existing.index = pd.to_datetime(existing.index)

    fm = FinMindClient()
    fm.initialize_frame(stock_id="TX", start_time=start_str, end_time=end_str)
    futures = fm.get_future_price()
    if futures.empty:
        raise RuntimeError("FinMind returned no TX futures data.")

    futures.index = pd.to_datetime(futures.index)
    day = futures[futures["trading_session"] == "position"].copy()
    night = futures[futures["trading_session"] == "after_market"].copy().add_suffix("_a")
    features = pd.concat([day, night], axis=1)
    features = features.loc[(features.index >= start_ts) & (features.index <= end_ts)].sort_index()
    if features.empty:
        raise RuntimeError("No day-session TX futures rows after filtering.")

    twse = GetInfoTWSE()
    raw_night = twse.get_institution_option_position(
        trading_session="night",
        start_time=start_str,
        end_time=end_str,
    )
    foreign_signal = _build_foreign_opt_signal_a(raw_night)
    features = features.join(foreign_signal, how="left")

    try:
        move = _tradingview_ohlc("MOVE", "TVC", tv_n_bars)
        move.columns = [f"MOVE_{col}" for col in move.columns]
        move = move.shift(1)
    except ModuleNotFoundError:
        if existing is None:
            raise
        move_cols = [col for col in existing.columns if col.startswith("MOVE_")]
        move = existing[move_cols].copy()

    try:
        sox = _tradingview_ohlc("SOX", "TVC", tv_n_bars)
        sox.columns = [f"SOX_{col}" for col in sox.columns]
        sox = sox.shift(1)
    except ModuleNotFoundError:
        if existing is None:
            raise
        sox_cols = [col for col in existing.columns if col.startswith("SOX_")]
        sox = existing[sox_cols].copy()

    features = features.join(move, how="left").join(sox, how="left")
    features = features.loc[(features.index >= start_ts) & (features.index <= end_ts)]
    features.index.name = "Timestamp"

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(output_path)

    return features
