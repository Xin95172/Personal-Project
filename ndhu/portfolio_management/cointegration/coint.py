"""Cointegration analysis helpers for pairs-trading research."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, coint
except ImportError:  # pragma: no cover - optional runtime dependency
    sm = None
    adfuller = None
    coint = None


@dataclass(frozen=True)
class CointegrationResult:
    """Engle-Granger test result for one pair."""

    y_symbol: str
    x_symbol: str
    pvalue: float
    test_stat: float
    spread_adf_pvalue: float
    spread_adf_stat: float
    hedge_ratio: float
    intercept: float
    n_obs: int


@dataclass(frozen=True)
class RollingWindow:
    """One formation/trading window used for walk-forward analysis."""

    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    trading_start: pd.Timestamp
    trading_end: pd.Timestamp


def load_price_series(
    parquet_path: str | Path,
    price_column: str = "close",
    time_column: str = "open_time",
) -> pd.Series:
    """Load one symbol's price series from a Parquet file."""

    parquet_path = Path(parquet_path)
    df = pd.read_parquet(parquet_path)

    if price_column not in df.columns:
        raise KeyError(f"{price_column!r} not found in {parquet_path.name}")

    if isinstance(df.index, pd.DatetimeIndex):
        index = df.index
    elif time_column in df.columns:
        index = pd.to_datetime(df[time_column])
    else:
        raise KeyError(
            f"Neither DatetimeIndex nor {time_column!r} exists in {parquet_path.name}",
        )

    series = pd.Series(
        pd.to_numeric(df[price_column], errors="coerce").to_numpy(),
        index=index,
        name=parquet_path.stem,
    )
    return series.sort_index().dropna()


def load_price_matrix(
    directory: str | Path,
    symbols: Iterable[str] | None = None,
    price_column: str = "close",
    time_column: str = "open_time",
    min_obs: int = 200,
) -> pd.DataFrame:
    """Load many symbol Parquet files into one aligned price matrix."""

    directory = Path(directory)
    wanted = set(symbols) if symbols is not None else None
    parquet_files = sorted(directory.glob("*.parquet"))
    series_list: list[pd.Series] = []

    for parquet_file in parquet_files:
        if wanted is not None and parquet_file.stem not in wanted:
            continue

        try:
            series = load_price_series(
                parquet_file,
                price_column=price_column,
                time_column=time_column,
            )
        except (KeyError, ValueError):
            continue

        if len(series) >= min_obs:
            series_list.append(series)

    if not series_list:
        return pd.DataFrame()

    prices = pd.concat(series_list, axis=1).sort_index()
    return prices.dropna(axis=1, thresh=min_obs)


def calculate_basis(
    futures: pd.Series,
    spot: pd.Series,
    method: str = "log",
) -> pd.Series:
    """Calculate one asset's futures-spot basis series."""

    aligned = pd.concat([futures, spot], axis=1).dropna()
    futures_price = aligned.iloc[:, 0]
    spot_price = aligned.iloc[:, 1]

    if method == "absolute":
        basis = futures_price - spot_price
    elif method == "pct":
        basis = (futures_price / spot_price) - 1
    elif method == "log":
        basis = np.log(futures_price / spot_price)
    else:
        raise ValueError("method must be one of: 'absolute', 'pct', 'log'")

    basis.name = futures.name
    return basis.replace([np.inf, -np.inf], np.nan).dropna()


def load_basis_matrices(
    spot_dir: str | Path,
    futures_dir: str | Path,
    symbols: Iterable[str] | None = None,
    price_column: str = "close",
    time_column: str = "open_time",
    basis_method: str = "log",
    min_obs: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load aligned basis, spot, and futures matrices for four-leg strategies."""

    spot_prices = load_price_matrix(
        spot_dir,
        symbols=symbols,
        price_column=price_column,
        time_column=time_column,
        min_obs=min_obs,
    )
    futures_prices = load_price_matrix(
        futures_dir,
        symbols=symbols,
        price_column=price_column,
        time_column=time_column,
        min_obs=min_obs,
    )
    common_symbols = sorted(set(spot_prices.columns) & set(futures_prices.columns))
    basis_series: list[pd.Series] = []

    for symbol in common_symbols:
        basis = calculate_basis(
            futures=futures_prices[symbol],
            spot=spot_prices[symbol],
            method=basis_method,
        )
        if len(basis) >= min_obs:
            basis.name = symbol
            basis_series.append(basis)

    if not basis_series:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    basis_matrix = pd.concat(basis_series, axis=1).sort_index()
    basis_matrix = basis_matrix.dropna(axis=1, thresh=min_obs)
    common_symbols = list(basis_matrix.columns)

    return (
        basis_matrix,
        spot_prices.reindex(index=basis_matrix.index, columns=common_symbols),
        futures_prices.reindex(index=basis_matrix.index, columns=common_symbols),
    )


def load_sector_map(
    metadata_path: str | Path,
    skip_sectors: Iterable[str] = ("USD Stablecoin", "Fiat-backed Stablecoin"),
    available_symbols: Iterable[str] | None = None,
) -> dict[str, list[str]]:
    """Load CoinGecko sector metadata and keep symbols available in prices."""

    metadata_path = Path(metadata_path)
    with metadata_path.open("r", encoding="utf-8") as file:
        raw_sector_map = json.load(file)

    skipped = set(skip_sectors)
    available = set(available_symbols) if available_symbols is not None else None
    sector_map: dict[str, list[str]] = {}

    for sector_name, info in raw_sector_map.items():
        if sector_name in skipped:
            continue

        symbols = sorted(set(info.get("symbols", [])))
        if available is not None:
            symbols = [symbol for symbol in symbols if symbol in available]

        if len(symbols) >= 2:
            sector_map[sector_name] = symbols

    return sector_map


def iter_rolling_windows(
    index: pd.DatetimeIndex,
    formation_window: str | pd.Timedelta = "90D",
    trading_window: str | pd.Timedelta = "30D",
    step: str | pd.Timedelta | None = None,
) -> list[RollingWindow]:
    """Create rolling formation/trading windows from a price index."""

    if len(index) == 0:
        return []

    timestamps = pd.DatetimeIndex(index).sort_values().unique()
    formation_delta = pd.Timedelta(formation_window)
    trading_delta = pd.Timedelta(trading_window)
    step_delta = pd.Timedelta(step) if step is not None else trading_delta

    windows: list[RollingWindow] = []
    formation_start = timestamps.min()
    final_timestamp = timestamps.max()

    while True:
        formation_end = formation_start + formation_delta
        trading_start = formation_end
        trading_end = trading_start + trading_delta

        if trading_start > final_timestamp:
            break

        windows.append(
            RollingWindow(
                formation_start=formation_start,
                formation_end=formation_end,
                trading_start=trading_start,
                trading_end=min(trading_end, final_timestamp),
            ),
        )

        if trading_end >= final_timestamp:
            break
        formation_start = formation_start + step_delta

    return windows


def estimate_hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """Estimate y = intercept + hedge_ratio * x using OLS."""

    _require_statsmodels()
    aligned = pd.concat([y, x], axis=1).dropna()
    y_values = aligned.iloc[:, 0]
    x_values = sm.add_constant(aligned.iloc[:, 1])
    model = sm.OLS(y_values, x_values).fit()
    return float(model.params.iloc[1]), float(model.params.iloc[0])


def engle_granger_test(
    y: pd.Series,
    x: pd.Series,
    y_symbol: str | None = None,
    x_symbol: str | None = None,
) -> CointegrationResult:
    """Run Engle-Granger cointegration test for one pair."""

    _require_statsmodels()
    aligned = pd.concat([y, x], axis=1).dropna()
    if aligned.empty:
        raise ValueError("Pair has no overlapping observations.")

    y_values = aligned.iloc[:, 0]
    x_values = aligned.iloc[:, 1]
    test_stat, pvalue, _ = coint(y_values, x_values)
    hedge_ratio, intercept = estimate_hedge_ratio(y_values, x_values)
    spread = calculate_spread(
        y_values,
        x_values,
        hedge_ratio=hedge_ratio,
        intercept=intercept,
    ).dropna()
    spread_adf_stat, spread_adf_pvalue, *_ = adfuller(spread)

    return CointegrationResult(
        y_symbol=y_symbol or str(y.name),
        x_symbol=x_symbol or str(x.name),
        pvalue=float(pvalue),
        test_stat=float(test_stat),
        spread_adf_pvalue=float(spread_adf_pvalue),
        spread_adf_stat=float(spread_adf_stat),
        hedge_ratio=hedge_ratio,
        intercept=intercept,
        n_obs=len(aligned),
    )


def scan_cointegrated_pairs(
    prices: pd.DataFrame,
    max_pvalue: float = 0.05,
    max_spread_adf_pvalue: float = 0.05,
    min_obs: int = 500,
    top_n: int | None = 30,
) -> pd.DataFrame:
    """Test all symbol pairs and return likely cointegrated candidates."""

    results: list[CointegrationResult] = []
    clean_prices = prices.dropna(axis=1, thresh=min_obs)

    for y_symbol, x_symbol in combinations(clean_prices.columns, 2):
        pair = clean_prices[[y_symbol, x_symbol]].dropna()
        if len(pair) < min_obs:
            continue

        try:
            result = engle_granger_test(
                pair[y_symbol],
                pair[x_symbol],
                y_symbol=y_symbol,
                x_symbol=x_symbol,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue

        if result.pvalue <= max_pvalue and result.spread_adf_pvalue <= max_spread_adf_pvalue:
            results.append(result)

    columns = [
        "y_symbol",
        "x_symbol",
        "pvalue",
        "test_stat",
        "spread_adf_pvalue",
        "spread_adf_stat",
        "hedge_ratio",
        "intercept",
        "n_obs",
    ]
    output = pd.DataFrame([result.__dict__ for result in results], columns=columns)
    if output.empty:
        return output

    output = output.sort_values(["pvalue", "spread_adf_pvalue", "test_stat"]).reset_index(drop=True)
    return output.head(top_n) if top_n is not None else output


def scan_sector_cointegrated_pairs(
    prices: pd.DataFrame,
    sector_map: dict[str, list[str]],
    max_pvalue: float = 0.05,
    max_spread_adf_pvalue: float = 0.05,
    min_obs: int = 500,
    top_n_per_sector: int | None = 5,
) -> pd.DataFrame:
    """Run cointegration scan separately within each sector."""

    frames: list[pd.DataFrame] = []

    for sector, symbols in sector_map.items():
        sector_symbols = [symbol for symbol in symbols if symbol in prices.columns]
        if len(sector_symbols) < 2:
            continue

        candidates = scan_cointegrated_pairs(
            prices[sector_symbols],
            max_pvalue=max_pvalue,
            max_spread_adf_pvalue=max_spread_adf_pvalue,
            min_obs=min_obs,
            top_n=top_n_per_sector,
        )
        if candidates.empty:
            continue

        candidates.insert(0, "sector", sector)
        frames.append(candidates)

    if not frames:
        return _empty_pair_frame(include_sector=True)

    return pd.concat(frames, ignore_index=True).sort_values(
        ["sector", "pvalue", "spread_adf_pvalue", "test_stat"],
    ).reset_index(drop=True)


def rolling_sector_cointegration_scan(
    prices: pd.DataFrame,
    sector_map: dict[str, list[str]],
    formation_window: str | pd.Timedelta = "90D",
    trading_window: str | pd.Timedelta = "30D",
    step: str | pd.Timedelta | None = None,
    max_pvalue: float = 0.05,
    max_spread_adf_pvalue: float = 0.05,
    min_obs: int = 500,
    top_n_per_sector: int | None = 3,
) -> pd.DataFrame:
    """Scan sector-level cointegrated pairs in each rolling formation window."""

    frames: list[pd.DataFrame] = []
    windows = iter_rolling_windows(
        prices.index,
        formation_window=formation_window,
        trading_window=trading_window,
        step=step,
    )

    for window_id, window in enumerate(windows, 1):
        formation_prices = prices.loc[
            (prices.index >= window.formation_start)
            & (prices.index < window.formation_end)
        ]
        if formation_prices.empty:
            continue

        candidates = scan_sector_cointegrated_pairs(
            formation_prices,
            sector_map=sector_map,
            max_pvalue=max_pvalue,
            max_spread_adf_pvalue=max_spread_adf_pvalue,
            min_obs=min_obs,
            top_n_per_sector=top_n_per_sector,
        )
        if candidates.empty:
            continue

        candidates.insert(0, "window_id", window_id)
        candidates.insert(1, "formation_start", window.formation_start)
        candidates.insert(2, "formation_end", window.formation_end)
        candidates.insert(3, "trading_start", window.trading_start)
        candidates.insert(4, "trading_end", window.trading_end)
        frames.append(candidates)

    if not frames:
        return _empty_rolling_pair_frame()

    return pd.concat(frames, ignore_index=True).sort_values(
        ["window_id", "sector", "pvalue", "spread_adf_pvalue", "test_stat"],
    ).reset_index(drop=True)


def calculate_spread(
    y: pd.Series,
    x: pd.Series,
    hedge_ratio: float,
    intercept: float = 0.0,
) -> pd.Series:
    """Calculate spread = y - intercept - hedge_ratio * x."""

    aligned = pd.concat([y, x], axis=1).dropna()
    spread = aligned.iloc[:, 0] - intercept - hedge_ratio * aligned.iloc[:, 1]
    spread.name = "spread"
    return spread


def rolling_zscore(series: pd.Series, window: int = 168) -> pd.Series:
    """Calculate rolling z-score for a spread series."""

    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    zscore = (series - mean) / std.replace(0, np.nan)
    zscore.name = "zscore"
    return zscore


def fixed_zscore(
    series: pd.Series,
    mean: float,
    std: float,
) -> pd.Series:
    """Calculate z-score using fixed formation-period parameters."""

    if std == 0 or np.isnan(std):
        return pd.Series(np.nan, index=series.index, name="zscore")

    zscore = (series - mean) / std
    zscore.name = "zscore"
    return zscore


def generate_pair_signals(
    zscore: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.Series:
    """Generate spread position signals from z-score.

    Position convention:
    -1 means short spread, +1 means long spread, 0 means flat.
    """

    positions: list[int] = []
    current = 0

    for value in zscore:
        if np.isnan(value):
            positions.append(current)
            continue

        if current == 0:
            if value > entry_z:
                current = -1
            elif value < -entry_z:
                current = 1
        elif current == 1 and value > -exit_z:
            current = 0
        elif current == -1 and value < exit_z:
            current = 0

        positions.append(current)

    return pd.Series(positions, index=zscore.index, name="position")


def backtest_pair(
    y: pd.Series,
    x: pd.Series,
    hedge_ratio: float,
    intercept: float = 0.0,
    z_window: int = 168,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    fee_rate: float = 0.0004,
) -> pd.DataFrame:
    """Backtest a simple mean-reversion spread strategy."""

    aligned = pd.concat([y, x], axis=1).dropna()
    y_price = aligned.iloc[:, 0]
    x_price = aligned.iloc[:, 1]

    spread = calculate_spread(y_price, x_price, hedge_ratio, intercept)
    zscore = rolling_zscore(spread, window=z_window)
    position = generate_pair_signals(zscore, entry_z=entry_z, exit_z=exit_z)
    lagged_position = position.shift(1).fillna(0)

    y_return = y_price.pct_change().fillna(0)
    x_return = x_price.pct_change().fillna(0)
    spread_return = y_return - hedge_ratio * x_return
    turnover = position.diff().abs().fillna(position.abs())
    strategy_return = lagged_position * spread_return - turnover * fee_rate

    result = pd.DataFrame(
        {
            "y_price": y_price,
            "x_price": x_price,
            "spread": spread,
            "zscore": zscore,
            "position": position,
            "spread_return": spread_return,
            "strategy_return": strategy_return,
            "equity_curve": (1 + strategy_return).cumprod(),
        },
    )
    return result


def backtest_pair_with_formation_stats(
    formation_y: pd.Series,
    formation_x: pd.Series,
    trading_y: pd.Series,
    trading_x: pd.Series,
    hedge_ratio: float,
    intercept: float = 0.0,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    fee_rate: float = 0.0004,
) -> pd.DataFrame:
    """Trade one pair using spread mean/std estimated only in formation data."""

    formation_spread = calculate_spread(
        formation_y,
        formation_x,
        hedge_ratio=hedge_ratio,
        intercept=intercept,
    )
    spread_mean = float(formation_spread.mean())
    spread_std = float(formation_spread.std())

    aligned = pd.concat([trading_y, trading_x], axis=1).dropna()
    if aligned.empty:
        return pd.DataFrame()

    y_price = aligned.iloc[:, 0]
    x_price = aligned.iloc[:, 1]
    trading_spread = calculate_spread(
        y_price,
        x_price,
        hedge_ratio=hedge_ratio,
        intercept=intercept,
    )
    zscore = fixed_zscore(trading_spread, mean=spread_mean, std=spread_std)
    position = generate_pair_signals(zscore, entry_z=entry_z, exit_z=exit_z)
    lagged_position = position.shift(1).fillna(0)

    y_return = y_price.pct_change().fillna(0)
    x_return = x_price.pct_change().fillna(0)
    spread_return = y_return - hedge_ratio * x_return
    turnover = position.diff().abs().fillna(position.abs())
    strategy_return = lagged_position * spread_return - turnover * fee_rate

    return pd.DataFrame(
        {
            "y_price": y_price,
            "x_price": x_price,
            "spread": trading_spread,
            "zscore": zscore,
            "position": position,
            "spread_return": spread_return,
            "strategy_return": strategy_return,
            "equity_curve": (1 + strategy_return).cumprod(),
            "formation_spread_mean": spread_mean,
            "formation_spread_std": spread_std,
        },
    )


def walk_forward_sector_backtest(
    prices: pd.DataFrame,
    rolling_pairs: pd.DataFrame,
    max_pairs_per_window: int | None = 10,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    fee_rate: float = 0.0004,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest rolling-selected sector pairs in their next trading windows."""

    backtest_frames: list[pd.DataFrame] = []
    trade_summaries: list[dict[str, object]] = []

    if rolling_pairs.empty:
        return pd.DataFrame(), pd.DataFrame()

    for window_id, window_pairs in rolling_pairs.groupby("window_id"):
        selected_pairs = window_pairs.sort_values(["pvalue", "spread_adf_pvalue", "test_stat"])
        if max_pairs_per_window is not None:
            selected_pairs = selected_pairs.head(max_pairs_per_window)

        for _, pair in selected_pairs.iterrows():
            y_symbol = pair["y_symbol"]
            x_symbol = pair["x_symbol"]
            formation_prices = prices.loc[
                (prices.index >= pair["formation_start"])
                & (prices.index < pair["formation_end"]),
                [y_symbol, x_symbol],
            ].dropna()
            trading_prices = prices.loc[
                (prices.index >= pair["trading_start"])
                & (prices.index < pair["trading_end"]),
                [y_symbol, x_symbol],
            ].dropna()

            if formation_prices.empty or trading_prices.empty:
                continue

            pair_backtest = backtest_pair_with_formation_stats(
                formation_y=formation_prices[y_symbol],
                formation_x=formation_prices[x_symbol],
                trading_y=trading_prices[y_symbol],
                trading_x=trading_prices[x_symbol],
                hedge_ratio=pair["hedge_ratio"],
                intercept=pair["intercept"],
                entry_z=entry_z,
                exit_z=exit_z,
                fee_rate=fee_rate,
            )
            if pair_backtest.empty:
                continue

            pair_backtest = pair_backtest.copy()
            pair_backtest.insert(0, "window_id", window_id)
            pair_backtest.insert(1, "sector", pair["sector"])
            pair_backtest.insert(2, "y_symbol", y_symbol)
            pair_backtest.insert(3, "x_symbol", x_symbol)
            pair_backtest.insert(4, "pvalue", pair["pvalue"])
            backtest_frames.append(pair_backtest)

            summary = summarize_backtest(pair_backtest)
            summary.update(
                {
                    "window_id": window_id,
                    "sector": pair["sector"],
                    "y_symbol": y_symbol,
                    "x_symbol": x_symbol,
                    "formation_start": pair["formation_start"],
                    "formation_end": pair["formation_end"],
                    "trading_start": pair["trading_start"],
                    "trading_end": pair["trading_end"],
                    "pvalue": pair["pvalue"],
                    "spread_adf_pvalue": pair["spread_adf_pvalue"],
                    "hedge_ratio": pair["hedge_ratio"],
                    "intercept": pair["intercept"],
                },
            )
            trade_summaries.append(summary)

    backtests = pd.concat(backtest_frames) if backtest_frames else pd.DataFrame()
    summaries = pd.DataFrame(trade_summaries)
    return backtests, summaries


def backtest_basis_pair_with_formation_stats(
    formation_y_basis: pd.Series,
    formation_x_basis: pd.Series,
    trading_y_basis: pd.Series,
    trading_x_basis: pd.Series,
    trading_y_spot: pd.Series,
    trading_y_futures: pd.Series,
    trading_x_spot: pd.Series,
    trading_x_futures: pd.Series,
    hedge_ratio: float,
    intercept: float = 0.0,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    fee_rate: float = 0.0004,
) -> pd.DataFrame:
    """Backtest a four-leg relative basis spread.

    The traded target spread follows the user's convention:
    y_basis - hedge_ratio * x_basis. The intercept is kept for diagnostics and
    compatibility with OLS, but the trading spread does not subtract it.
    """

    formation_target_spread = (
        pd.concat([formation_y_basis, formation_x_basis], axis=1).dropna()
    )
    if formation_target_spread.empty:
        return pd.DataFrame()

    formation_spread = (
        formation_target_spread.iloc[:, 0]
        - hedge_ratio * formation_target_spread.iloc[:, 1]
    )
    spread_mean = float(formation_spread.mean())
    spread_std = float(formation_spread.std())

    aligned = pd.concat(
        [
            trading_y_basis,
            trading_x_basis,
            trading_y_spot,
            trading_y_futures,
            trading_x_spot,
            trading_x_futures,
        ],
        axis=1,
    ).dropna()
    if aligned.empty:
        return pd.DataFrame()

    y_basis = aligned.iloc[:, 0]
    x_basis = aligned.iloc[:, 1]
    y_spot = aligned.iloc[:, 2]
    y_futures = aligned.iloc[:, 3]
    x_spot = aligned.iloc[:, 4]
    x_futures = aligned.iloc[:, 5]

    target_spread = y_basis - hedge_ratio * x_basis
    target_spread.name = "target_spread"
    zscore = fixed_zscore(target_spread, mean=spread_mean, std=spread_std)
    position = generate_pair_signals(zscore, entry_z=entry_z, exit_z=exit_z).astype(float)

    lagged_position = position.shift(1).fillna(0)

    y_basis_ret = y_futures.pct_change().fillna(0) - y_spot.pct_change().fillna(0)
    x_basis_ret = x_futures.pct_change().fillna(0) - x_spot.pct_change().fillna(0)
    spread_return = y_basis_ret - hedge_ratio * x_basis_ret
    raw_strategy_return = lagged_position * spread_return

    turnover = position.diff().abs().fillna(position.abs())
    trading_cost = turnover * fee_rate * (2 + 2 * abs(hedge_ratio))
    strategy_return = raw_strategy_return - trading_cost

    result = pd.DataFrame(
        {
            "y_basis": y_basis,
            "x_basis": x_basis,
            "y_spot": y_spot,
            "y_futures": y_futures,
            "x_spot": x_spot,
            "x_futures": x_futures,
            "target_spread": target_spread,
            "zscore": zscore,
            "position": position,
            "lagged_position": lagged_position,
            "y_basis_ret": y_basis_ret,
            "x_basis_ret": x_basis_ret,
            "spread_return": spread_return,
            "raw_strategy_return": raw_strategy_return,
            "turnover": turnover,
            "trading_cost": trading_cost,
            "strategy_return": strategy_return,
            "equity_curve": (1 + strategy_return).cumprod(),
            "formation_spread_mean": spread_mean,
            "formation_spread_std": spread_std,
            "intercept": intercept,
        },
    )
    return result


def walk_forward_basis_backtest(
    basis: pd.DataFrame,
    spot_prices: pd.DataFrame,
    futures_prices: pd.DataFrame,
    rolling_pairs: pd.DataFrame,
    max_pairs_per_window: int | None = 10,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    fee_rate: float = 0.0004,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest rolling-selected sector pairs as four-leg basis trades."""

    backtest_frames: list[pd.DataFrame] = []
    trade_summaries: list[dict[str, object]] = []

    if rolling_pairs.empty:
        return pd.DataFrame(), pd.DataFrame()

    for window_id, window_pairs in rolling_pairs.groupby("window_id"):
        selected_pairs = window_pairs.sort_values(["pvalue", "spread_adf_pvalue", "test_stat"])
        if max_pairs_per_window is not None:
            selected_pairs = selected_pairs.head(max_pairs_per_window)

        for _, pair in selected_pairs.iterrows():
            y_symbol = pair["y_symbol"]
            x_symbol = pair["x_symbol"]

            formation_basis = basis.loc[
                (basis.index >= pair["formation_start"])
                & (basis.index < pair["formation_end"]),
                [y_symbol, x_symbol],
            ].dropna()
            trading_basis = basis.loc[
                (basis.index >= pair["trading_start"])
                & (basis.index < pair["trading_end"]),
                [y_symbol, x_symbol],
            ].dropna()

            if formation_basis.empty or trading_basis.empty:
                continue

            pair_backtest = backtest_basis_pair_with_formation_stats(
                formation_y_basis=formation_basis[y_symbol],
                formation_x_basis=formation_basis[x_symbol],
                trading_y_basis=trading_basis[y_symbol],
                trading_x_basis=trading_basis[x_symbol],
                trading_y_spot=spot_prices[y_symbol],
                trading_y_futures=futures_prices[y_symbol],
                trading_x_spot=spot_prices[x_symbol],
                trading_x_futures=futures_prices[x_symbol],
                hedge_ratio=pair["hedge_ratio"],
                intercept=pair["intercept"],
                entry_z=entry_z,
                exit_z=exit_z,
                fee_rate=fee_rate,
            )
            if pair_backtest.empty:
                continue

            pair_backtest = pair_backtest.copy()
            pair_backtest.insert(0, "window_id", window_id)
            pair_backtest.insert(1, "sector", pair["sector"])
            pair_backtest.insert(2, "y_symbol", y_symbol)
            pair_backtest.insert(3, "x_symbol", x_symbol)
            pair_backtest.insert(4, "pvalue", pair["pvalue"])
            pair_backtest.insert(5, "spread_adf_pvalue", pair["spread_adf_pvalue"])
            pair_backtest.insert(6, "hedge_ratio", pair["hedge_ratio"])
            backtest_frames.append(pair_backtest)

            summary = summarize_backtest(pair_backtest)
            summary.update(
                {
                    "window_id": window_id,
                    "sector": pair["sector"],
                    "y_symbol": y_symbol,
                    "x_symbol": x_symbol,
                    "formation_start": pair["formation_start"],
                    "formation_end": pair["formation_end"],
                    "trading_start": pair["trading_start"],
                    "trading_end": pair["trading_end"],
                    "pvalue": pair["pvalue"],
                    "spread_adf_pvalue": pair["spread_adf_pvalue"],
                    "hedge_ratio": pair["hedge_ratio"],
                    "intercept": pair["intercept"],
                },
            )
            trade_summaries.append(summary)

    backtests = pd.concat(backtest_frames) if backtest_frames else pd.DataFrame()
    summaries = pd.DataFrame(trade_summaries)
    return backtests, summaries


def build_trading_log(backtests: pd.DataFrame) -> pd.DataFrame:
    """Convert pair backtest time series into entry/exit trade records."""

    if backtests.empty:
        return pd.DataFrame(
            columns=[
                "window_id",
                "sector",
                "y_symbol",
                "x_symbol",
                "side",
                "entry_time",
                "exit_time",
                "entry_zscore",
                "exit_zscore",
                "y_futures_position",
                "y_spot_position",
                "x_futures_position",
                "x_spot_position",
                "holding_periods",
                "trade_return",
            ],
        )

    trades: list[dict[str, object]] = []
    group_columns = ["window_id", "sector", "y_symbol", "x_symbol"]

    for keys, group in backtests.groupby(group_columns, sort=False):
        group = group.sort_index()
        open_trade: dict[str, object] | None = None

        for timestamp, row in group.iterrows():
            position = int(row["position"])

            if open_trade is None and position != 0:
                open_trade = {
                    "window_id": keys[0],
                    "sector": keys[1],
                    "y_symbol": keys[2],
                    "x_symbol": keys[3],
                    "side": "long_spread" if position > 0 else "short_spread",
                    "entry_time": timestamp,
                    "entry_zscore": row["zscore"],
                    "entry_equity": row["equity_curve"],
                    "entry_position": position,
                    "y_futures_position": row.get("y_futures_position", np.nan),
                    "y_spot_position": row.get("y_spot_position", np.nan),
                    "x_futures_position": row.get("x_futures_position", np.nan),
                    "x_spot_position": row.get("x_spot_position", np.nan),
                }
                continue

            if open_trade is None:
                continue

            entry_position = int(open_trade["entry_position"])
            should_close = position == 0 or position != entry_position
            if not should_close:
                continue

            entry_equity = float(open_trade["entry_equity"])
            exit_equity = float(row["equity_curve"])
            trade_return = exit_equity / entry_equity - 1 if entry_equity else np.nan
            entry_time = open_trade["entry_time"]

            trades.append(
                {
                    "window_id": open_trade["window_id"],
                    "sector": open_trade["sector"],
                    "y_symbol": open_trade["y_symbol"],
                    "x_symbol": open_trade["x_symbol"],
                    "side": open_trade["side"],
                    "entry_time": entry_time,
                    "exit_time": timestamp,
                    "entry_zscore": open_trade["entry_zscore"],
                    "exit_zscore": row["zscore"],
                    "y_futures_position": open_trade["y_futures_position"],
                    "y_spot_position": open_trade["y_spot_position"],
                    "x_futures_position": open_trade["x_futures_position"],
                    "x_spot_position": open_trade["x_spot_position"],
                    "holding_periods": group.index.get_loc(timestamp) - group.index.get_loc(entry_time),
                    "trade_return": trade_return,
                },
            )

            if position != 0:
                open_trade = {
                    "window_id": keys[0],
                    "sector": keys[1],
                    "y_symbol": keys[2],
                    "x_symbol": keys[3],
                    "side": "long_spread" if position > 0 else "short_spread",
                    "entry_time": timestamp,
                    "entry_zscore": row["zscore"],
                    "entry_equity": row["equity_curve"],
                    "entry_position": position,
                    "y_futures_position": row.get("y_futures_position", np.nan),
                    "y_spot_position": row.get("y_spot_position", np.nan),
                    "x_futures_position": row.get("x_futures_position", np.nan),
                    "x_spot_position": row.get("x_spot_position", np.nan),
                }
            else:
                open_trade = None

        if open_trade is not None:
            last_timestamp = group.index[-1]
            last_row = group.iloc[-1]
            entry_equity = float(open_trade["entry_equity"])
            exit_equity = float(last_row["equity_curve"])
            entry_time = open_trade["entry_time"]

            trades.append(
                {
                    "window_id": open_trade["window_id"],
                    "sector": open_trade["sector"],
                    "y_symbol": open_trade["y_symbol"],
                    "x_symbol": open_trade["x_symbol"],
                    "side": open_trade["side"],
                    "entry_time": entry_time,
                    "exit_time": last_timestamp,
                    "entry_zscore": open_trade["entry_zscore"],
                    "exit_zscore": last_row["zscore"],
                    "y_futures_position": open_trade["y_futures_position"],
                    "y_spot_position": open_trade["y_spot_position"],
                    "x_futures_position": open_trade["x_futures_position"],
                    "x_spot_position": open_trade["x_spot_position"],
                    "holding_periods": group.index.get_loc(last_timestamp) - group.index.get_loc(entry_time),
                    "trade_return": exit_equity / entry_equity - 1 if entry_equity else np.nan,
                },
            )

    return pd.DataFrame(trades)


def summarize_backtest(backtest: pd.DataFrame, periods_per_year: int = 24 * 365) -> dict[str, float]:
    """Summarize a backtest result table."""

    returns = backtest["strategy_return"].dropna()
    if returns.empty:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    equity = (1 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1)
    annual_return = float((1 + total_return) ** (periods_per_year / len(returns)) - 1)
    annual_volatility = float(returns.std() * np.sqrt(periods_per_year))
    sharpe = float(annual_return / annual_volatility) if annual_volatility else 0.0
    drawdown = equity / equity.cummax() - 1

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
    }


def _require_statsmodels() -> None:
    if sm is None or adfuller is None or coint is None:
        raise ImportError("statsmodels is required. Install it with: pip install statsmodels")


def _empty_pair_frame(include_sector: bool = False) -> pd.DataFrame:
    columns = [
        "y_symbol",
        "x_symbol",
        "pvalue",
        "test_stat",
        "spread_adf_pvalue",
        "spread_adf_stat",
        "hedge_ratio",
        "intercept",
        "n_obs",
    ]
    if include_sector:
        columns.insert(0, "sector")
    return pd.DataFrame(columns=columns)


def _empty_rolling_pair_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "window_id",
            "formation_start",
            "formation_end",
            "trading_start",
            "trading_end",
            "sector",
            "y_symbol",
            "x_symbol",
            "pvalue",
            "test_stat",
            "spread_adf_pvalue",
            "spread_adf_stat",
            "hedge_ratio",
            "intercept",
            "n_obs",
        ],
    )
