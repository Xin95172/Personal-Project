"""Cointegration analysis helpers for pairs-trading research."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations
import json
import os
import time
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

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional runtime dependency
    tqdm = None


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
    index = _to_naive_datetime_index(index)

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


def load_funding_matrix(
    directory: str | Path,
    symbols: Iterable[str] | None = None,
    funding_column: str = "fundingRate",
    time_column: str = "open_time",
    min_obs: int = 1,
    timestamp_floor: str | None = "h",
) -> pd.DataFrame:
    """Load funding-rate Parquet files into an aligned matrix.

    Positive funding means longs pay shorts. Missing timestamps are treated as
    zero funding in the backtest because funding is charged only periodically.
    """

    funding = load_price_matrix(
        directory=directory,
        symbols=symbols,
        price_column=funding_column,
        time_column=time_column,
        min_obs=min_obs,
    )
    if timestamp_floor and not funding.empty:
        funding = funding.copy()
        funding.index = funding.index.floor(timestamp_floor)
        funding = funding.groupby(level=0).last().sort_index()
    return funding


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
    prefilter_top_pairs: int | None = None,
    n_jobs: int = 1,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> pd.DataFrame:
    """Test all symbol pairs and return likely cointegrated candidates."""

    _require_statsmodels()
    clean_prices = prices.dropna(axis=1, thresh=min_obs)
    symbol_pairs = _select_pair_symbols(
        clean_prices,
        max_pairs=prefilter_top_pairs,
    )
    pair_args = [
        (
            y_symbol,
            x_symbol,
            clean_prices,
            min_obs,
            max_pvalue,
            max_spread_adf_pvalue,
        )
        for y_symbol, x_symbol in symbol_pairs
    ]

    workers = _resolve_n_jobs(n_jobs)
    if workers == 1 or len(pair_args) <= 1:
        iterator = _progress(
            pair_args,
            total=len(pair_args),
            desc=progress_desc or "cointegration pairs",
            enabled=show_progress,
            leave=False,
        )
        results = [_scan_pair_task(args) for args in iterator]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            iterator = executor.map(_scan_pair_task, pair_args)
            iterator = _progress(
                iterator,
                total=len(pair_args),
                desc=progress_desc or "cointegration pairs",
                enabled=show_progress,
                leave=False,
            )
            results = list(iterator)

    results = [result for result in results if result is not None]

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
    prefilter_top_pairs_per_sector: int | None = None,
    n_jobs: int = 1,
    show_progress: bool = False,
    progress_prefix: str = "",
) -> pd.DataFrame:
    """Run cointegration scan separately within each sector."""

    frames: list[pd.DataFrame] = []

    sector_items = list(sector_map.items())
    sector_iterator = _progress(
        sector_items,
        total=len(sector_items),
        desc=f"{progress_prefix}sectors".strip(),
        enabled=show_progress,
        leave=False,
    )

    for sector, symbols in sector_iterator:
        sector_symbols = [symbol for symbol in symbols if symbol in prices.columns]
        if len(sector_symbols) < 2:
            continue

        pair_count = len(sector_symbols) * (len(sector_symbols) - 1) // 2
        candidates = scan_cointegrated_pairs(
            prices[sector_symbols],
            max_pvalue=max_pvalue,
            max_spread_adf_pvalue=max_spread_adf_pvalue,
            min_obs=min_obs,
            top_n=top_n_per_sector,
            prefilter_top_pairs=prefilter_top_pairs_per_sector,
            n_jobs=n_jobs,
            show_progress=show_progress,
            progress_desc=f"{progress_prefix}{sector} pairs ({pair_count})".strip(),
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
    prefilter_top_pairs_per_sector: int | None = None,
    n_jobs: int = 1,
    cache_path: str | Path | None = None,
    progress_path: str | Path | None = None,
    use_cache: bool = True,
    incremental_cache: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Scan sector-level cointegrated pairs in each rolling formation window.

    When ``incremental_cache`` is enabled, each completed window is written to
    ``cache_path`` immediately and progress diagnostics are written to
    ``progress_path``. An interrupted scan can then resume from the next
    unfinished window.
    """

    cache_path = Path(cache_path) if cache_path is not None else None
    cache_metadata_path = (
        cache_path.with_name(f"{cache_path.stem}_metadata.json")
        if cache_path is not None
        else None
    )
    if progress_path is None and cache_path is not None:
        progress_path = cache_path.with_name(f"{cache_path.stem}_progress.csv")
    progress_path = Path(progress_path) if progress_path is not None else None

    windows = iter_rolling_windows(
        prices.index,
        formation_window=formation_window,
        trading_window=trading_window,
        step=step,
    )
    total_windows = len(windows)

    frames: list[pd.DataFrame] = []
    completed_window_ids: set[int] = set()
    existing_progress = _read_rolling_progress(progress_path)
    cache_metadata = _build_rolling_cache_metadata(
        prices=prices,
        sector_map=sector_map,
        formation_window=formation_window,
        trading_window=trading_window,
        step=step,
        max_pvalue=max_pvalue,
        max_spread_adf_pvalue=max_spread_adf_pvalue,
        min_obs=min_obs,
        top_n_per_sector=top_n_per_sector,
        prefilter_top_pairs_per_sector=prefilter_top_pairs_per_sector,
    )
    cache_is_compatible = _is_rolling_cache_compatible(
        cache_metadata_path,
        cache_metadata,
    )

    if cache_path is not None and cache_path.exists() and not cache_is_compatible:
        _reset_incompatible_rolling_cache(
            cache_path=cache_path,
            progress_path=progress_path,
            cache_metadata_path=cache_metadata_path,
        )
        existing_progress = pd.DataFrame()

    if use_cache and cache_path is not None and cache_path.exists() and cache_is_compatible:
        cached_pairs = pd.read_parquet(cache_path)
        if not incremental_cache or existing_progress.empty:
            return cached_pairs
        if not {"window_id", "status"}.issubset(existing_progress.columns):
            return cached_pairs

        completed_window_ids = set(
            existing_progress.loc[
                existing_progress["status"].eq("completed"),
                "window_id",
            ].astype(int),
        )
        if len(completed_window_ids) >= total_windows:
            return cached_pairs
        if not cached_pairs.empty:
            frames.append(cached_pairs)

    window_iterator = _progress(
        list(enumerate(windows, 1)),
        total=len(windows),
        desc="rolling windows",
        enabled=show_progress,
        leave=True,
    )

    for window_id, window in window_iterator:
        if window_id in completed_window_ids:
            continue

        started_at = time.perf_counter()
        formation_prices = prices.loc[
            (prices.index >= window.formation_start)
            & (prices.index < window.formation_end)
        ]
        effective_symbols, effective_pairs, largest_sector_pairs = _count_effective_sector_pairs(
            formation_prices,
            sector_map=sector_map,
            min_obs=min_obs,
        )

        if formation_prices.empty:
            _write_rolling_progress(
                progress_path,
                {
                    "window_id": window_id,
                    "total_windows": total_windows,
                    "formation_start": window.formation_start,
                    "formation_end": window.formation_end,
                    "trading_start": window.trading_start,
                    "trading_end": window.trading_end,
                    "effective_symbols": 0,
                    "effective_pairs": 0,
                    "largest_sector_pairs": 0,
                    "candidates": 0,
                    "elapsed_seconds": time.perf_counter() - started_at,
                    "status": "completed",
                },
            )
            continue

        candidates = scan_sector_cointegrated_pairs(
            formation_prices,
            sector_map=sector_map,
            max_pvalue=max_pvalue,
            max_spread_adf_pvalue=max_spread_adf_pvalue,
            min_obs=min_obs,
            top_n_per_sector=top_n_per_sector,
            prefilter_top_pairs_per_sector=prefilter_top_pairs_per_sector,
            n_jobs=n_jobs,
            show_progress=show_progress,
            progress_prefix=f"window {window_id}: ",
        )

        if not candidates.empty:
            candidates.insert(0, "window_id", window_id)
            candidates.insert(1, "formation_start", window.formation_start)
            candidates.insert(2, "formation_end", window.formation_end)
            candidates.insert(3, "trading_start", window.trading_start)
            candidates.insert(4, "trading_end", window.trading_end)
            frames.append(candidates)

        elapsed_seconds = time.perf_counter() - started_at
        if incremental_cache and cache_path is not None:
            _write_rolling_cache(cache_path, frames)
            _write_rolling_cache_metadata(cache_metadata_path, cache_metadata)
        _write_rolling_progress(
            progress_path,
            {
                "window_id": window_id,
                "total_windows": total_windows,
                "formation_start": window.formation_start,
                "formation_end": window.formation_end,
                "trading_start": window.trading_start,
                "trading_end": window.trading_end,
                "effective_symbols": effective_symbols,
                "effective_pairs": effective_pairs,
                "largest_sector_pairs": largest_sector_pairs,
                "candidates": len(candidates),
                "elapsed_seconds": elapsed_seconds,
                "status": "completed",
            },
        )

    if not frames:
        output = _empty_rolling_pair_frame()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            output.to_parquet(cache_path, index=False)
            _write_rolling_cache_metadata(cache_metadata_path, cache_metadata)
        return output

    output = pd.concat(frames, ignore_index=True).sort_values(
        ["window_id", "sector", "pvalue", "spread_adf_pvalue", "test_stat"],
    ).reset_index(drop=True)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(cache_path, index=False)
        _write_rolling_cache_metadata(cache_metadata_path, cache_metadata)
    return output


def _count_effective_sector_pairs(
    prices: pd.DataFrame,
    sector_map: dict[str, list[str]],
    min_obs: int,
) -> tuple[int, int, int]:
    """Count symbols and sector-constrained pairs that pass the observation filter."""

    if prices.empty:
        return 0, 0, 0

    clean_columns = set(prices.dropna(axis=1, thresh=min_obs).columns)
    effective_symbols: set[str] = set()
    effective_pairs = 0
    largest_sector_pairs = 0

    for symbols in sector_map.values():
        sector_symbols = [symbol for symbol in symbols if symbol in clean_columns]
        sector_pair_count = len(sector_symbols) * (len(sector_symbols) - 1) // 2
        effective_symbols.update(sector_symbols)
        effective_pairs += sector_pair_count
        largest_sector_pairs = max(largest_sector_pairs, sector_pair_count)

    return len(effective_symbols), effective_pairs, largest_sector_pairs


def _read_rolling_progress(progress_path: Path | None) -> pd.DataFrame:
    if progress_path is None or not progress_path.exists():
        return pd.DataFrame()
    return pd.read_csv(progress_path)


def _write_rolling_progress(progress_path: Path | None, row: dict[str, object]) -> None:
    if progress_path is None:
        return

    progress_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        key: value.isoformat() if isinstance(value, pd.Timestamp) else value
        for key, value in row.items()
    }
    row_frame = pd.DataFrame([row])

    if progress_path.exists():
        progress = pd.read_csv(progress_path)
        if "window_id" in progress.columns:
            progress = progress[progress["window_id"].astype(int) != int(row["window_id"])]
        progress = pd.concat([progress, row_frame], ignore_index=True)
    else:
        progress = row_frame

    progress = progress.sort_values("window_id").reset_index(drop=True)
    progress.to_csv(progress_path, index=False)


def _write_rolling_cache(cache_path: Path, frames: list[pd.DataFrame]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        _empty_rolling_pair_frame().to_parquet(cache_path, index=False)
        return

    output = pd.concat(frames, ignore_index=True)
    if not output.empty:
        output = output.sort_values(
            ["window_id", "sector", "pvalue", "spread_adf_pvalue", "test_stat"],
        ).reset_index(drop=True)
    output.to_parquet(cache_path, index=False)


def _build_rolling_cache_metadata(
    prices: pd.DataFrame,
    sector_map: dict[str, list[str]],
    formation_window: str | pd.Timedelta,
    trading_window: str | pd.Timedelta,
    step: str | pd.Timedelta | None,
    max_pvalue: float,
    max_spread_adf_pvalue: float,
    min_obs: int,
    top_n_per_sector: int | None,
    prefilter_top_pairs_per_sector: int | None,
) -> dict[str, object]:
    return {
        "version": 2,
        "formation_window": str(formation_window),
        "trading_window": str(trading_window),
        "step": str(step),
        "max_pvalue": float(max_pvalue),
        "max_spread_adf_pvalue": float(max_spread_adf_pvalue),
        "min_obs": int(min_obs),
        "top_n_per_sector": top_n_per_sector,
        "prefilter_top_pairs_per_sector": prefilter_top_pairs_per_sector,
        "symbols": sorted(map(str, prices.columns)),
        "sector_names": sorted(map(str, sector_map.keys())),
    }


def _is_rolling_cache_compatible(
    cache_metadata_path: Path | None,
    expected_metadata: dict[str, object],
) -> bool:
    if cache_metadata_path is None:
        return True
    if not cache_metadata_path.exists():
        return False

    try:
        with cache_metadata_path.open("r", encoding="utf-8") as file:
            existing_metadata = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False

    return existing_metadata == expected_metadata


def _write_rolling_cache_metadata(
    cache_metadata_path: Path | None,
    metadata: dict[str, object],
) -> None:
    if cache_metadata_path is None:
        return
    cache_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def _reset_incompatible_rolling_cache(
    cache_path: Path,
    progress_path: Path | None,
    cache_metadata_path: Path | None,
) -> None:
    for path in (cache_path, progress_path, cache_metadata_path):
        if path is not None and path.exists():
            path.unlink()


def _select_pair_symbols(
    clean_prices: pd.DataFrame,
    max_pairs: int | None = None,
) -> list[tuple[str, str]]:
    symbols = list(clean_prices.columns)
    all_pairs = list(combinations(symbols, 2))
    if max_pairs is None or max_pairs <= 0 or len(all_pairs) <= max_pairs:
        return all_pairs

    corr = clean_prices.corr().abs()
    scored_pairs: list[tuple[float, str, str]] = []
    for y_symbol, x_symbol in all_pairs:
        score = corr.at[y_symbol, x_symbol]
        if np.isfinite(score):
            scored_pairs.append((float(score), y_symbol, x_symbol))

    scored_pairs.sort(reverse=True)
    return [(y_symbol, x_symbol) for _, y_symbol, x_symbol in scored_pairs[:max_pairs]]


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
    stop_loss_return: float | None = None,
    stop_loss_z: float | None = None,
    trading_y_funding_rate: pd.Series | None = None,
    trading_x_funding_rate: pd.Series | None = None,
) -> pd.DataFrame:
    """Backtest a return-based relative basis spread with optional funding.

    The traded target spread follows the user's convention:
    y_basis - hedge_ratio * x_basis. The intercept is kept for diagnostics and
    compatibility with OLS, but the trading spread does not subtract it.
    Positive funding means long futures pay short futures.
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

    y_basis_ret = y_futures.pct_change().fillna(0) - y_spot.pct_change().fillna(0)
    x_basis_ret = x_futures.pct_change().fillna(0) - x_spot.pct_change().fillna(0)
    spread_return = y_basis_ret - hedge_ratio * x_basis_ret

    y_funding_rate = _align_optional_rate(trading_y_funding_rate, aligned.index)
    x_funding_rate = _align_optional_rate(trading_x_funding_rate, aligned.index)

    base_position = generate_pair_signals(zscore, entry_z=entry_z, exit_z=exit_z).astype(float)
    position, stop_reason = _apply_stop_loss_to_positions(
        target_position=base_position,
        zscore=zscore,
        spread_return=spread_return,
        y_funding_rate=y_funding_rate,
        x_funding_rate=x_funding_rate,
        hedge_ratio=hedge_ratio,
        stop_loss_return=stop_loss_return,
        stop_loss_z=stop_loss_z,
    )

    lagged_position = position.shift(1).fillna(0)
    y_futures_funding_position = lagged_position
    x_futures_funding_position = -hedge_ratio * lagged_position
    raw_strategy_return = lagged_position * spread_return
    funding_return = -(
        y_futures_funding_position * y_funding_rate
        + x_futures_funding_position * x_funding_rate
    )

    turnover = position.diff().abs().fillna(position.abs())
    trading_cost = turnover * fee_rate * (2 + 2 * abs(hedge_ratio))
    strategy_return = raw_strategy_return + funding_return - trading_cost

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
            "base_position": base_position,
            "position": position,
            "stop_reason": stop_reason,
            "lagged_position": lagged_position,
            "y_futures_funding_position": y_futures_funding_position,
            "x_futures_funding_position": x_futures_funding_position,
            "y_basis_ret": y_basis_ret,
            "x_basis_ret": x_basis_ret,
            "spread_return": spread_return,
            "raw_strategy_return": raw_strategy_return,
            "y_funding_rate": y_funding_rate,
            "x_funding_rate": x_funding_rate,
            "funding_return": funding_return,
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
    funding_rates: pd.DataFrame | None = None,
    max_pairs_per_window: int | None = 10,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    fee_rate: float = 0.0004,
    stop_loss_return: float | None = None,
    stop_loss_z: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest rolling-selected sector pairs as return-based basis trades."""

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
                stop_loss_return=stop_loss_return,
                stop_loss_z=stop_loss_z,
                trading_y_funding_rate=_get_optional_column(funding_rates, y_symbol),
                trading_x_funding_rate=_get_optional_column(funding_rates, x_symbol),
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
                "exit_reason",
                "entry_zscore",
                "exit_zscore",
                "entry_position",
                "hedge_ratio",
                "y_futures_funding_position",
                "x_futures_funding_position",
                "holding_periods",
                "trade_funding_return",
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
                    "hedge_ratio": row.get("hedge_ratio", np.nan),
                    "y_futures_funding_position": row.get("y_futures_funding_position", np.nan),
                    "x_futures_funding_position": row.get("x_futures_funding_position", np.nan),
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
            trade_slice = group.loc[entry_time:timestamp]
            trade_funding_return = float(trade_slice.get("funding_return", pd.Series(dtype=float)).sum())

            trades.append(
                {
                    "window_id": open_trade["window_id"],
                    "sector": open_trade["sector"],
                    "y_symbol": open_trade["y_symbol"],
                    "x_symbol": open_trade["x_symbol"],
                    "side": open_trade["side"],
                    "entry_time": entry_time,
                    "exit_time": timestamp,
                    "exit_reason": row.get("stop_reason", "") or "signal",
                    "entry_zscore": open_trade["entry_zscore"],
                    "exit_zscore": row["zscore"],
                    "entry_position": open_trade["entry_position"],
                    "hedge_ratio": open_trade["hedge_ratio"],
                    "y_futures_funding_position": open_trade["y_futures_funding_position"],
                    "x_futures_funding_position": open_trade["x_futures_funding_position"],
                    "holding_periods": group.index.get_loc(timestamp) - group.index.get_loc(entry_time),
                    "trade_funding_return": trade_funding_return,
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
                    "hedge_ratio": row.get("hedge_ratio", np.nan),
                    "y_futures_funding_position": row.get("y_futures_funding_position", np.nan),
                    "x_futures_funding_position": row.get("x_futures_funding_position", np.nan),
                }
            else:
                open_trade = None

        if open_trade is not None:
            last_timestamp = group.index[-1]
            last_row = group.iloc[-1]
            entry_equity = float(open_trade["entry_equity"])
            exit_equity = float(last_row["equity_curve"])
            entry_time = open_trade["entry_time"]
            trade_slice = group.loc[entry_time:last_timestamp]
            trade_funding_return = float(trade_slice.get("funding_return", pd.Series(dtype=float)).sum())

            trades.append(
                {
                    "window_id": open_trade["window_id"],
                    "sector": open_trade["sector"],
                    "y_symbol": open_trade["y_symbol"],
                    "x_symbol": open_trade["x_symbol"],
                    "side": open_trade["side"],
                    "entry_time": entry_time,
                    "exit_time": last_timestamp,
                    "exit_reason": last_row.get("stop_reason", "") or "window_end",
                    "entry_zscore": open_trade["entry_zscore"],
                    "exit_zscore": last_row["zscore"],
                    "entry_position": open_trade["entry_position"],
                    "hedge_ratio": open_trade["hedge_ratio"],
                    "y_futures_funding_position": open_trade["y_futures_funding_position"],
                    "x_futures_funding_position": open_trade["x_futures_funding_position"],
                    "holding_periods": group.index.get_loc(last_timestamp) - group.index.get_loc(entry_time),
                    "trade_funding_return": trade_funding_return,
                    "trade_return": exit_equity / entry_equity - 1 if entry_equity else np.nan,
                },
            )

    return pd.DataFrame(trades)


def _apply_stop_loss_to_positions(
    target_position: pd.Series,
    zscore: pd.Series,
    spread_return: pd.Series,
    y_funding_rate: pd.Series,
    x_funding_rate: pd.Series,
    hedge_ratio: float,
    stop_loss_return: float | None = None,
    stop_loss_z: float | None = None,
) -> tuple[pd.Series, pd.Series]:
    if stop_loss_return is None and stop_loss_z is None:
        return (
            target_position.astype(float),
            pd.Series("", index=target_position.index, name="stop_reason"),
        )

    positions: list[float] = []
    stop_reasons: list[str] = []
    current_position = 0.0
    trade_equity = 1.0
    stopped_until_flat = False

    for timestamp in target_position.index:
        desired_position = float(target_position.loc[timestamp])
        current_zscore = float(zscore.loc[timestamp]) if pd.notna(zscore.loc[timestamp]) else np.nan

        if current_position != 0:
            bar_return = (
                current_position * float(spread_return.loc[timestamp])
                - (
                    current_position * float(y_funding_rate.loc[timestamp])
                    + (-hedge_ratio * current_position) * float(x_funding_rate.loc[timestamp])
                )
            )
            trade_equity *= 1 + bar_return

        reason = ""
        if current_position != 0:
            trade_return = trade_equity - 1
            hit_return_stop = (
                stop_loss_return is not None
                and np.isfinite(trade_return)
                and trade_return <= stop_loss_return
            )
            hit_z_stop = (
                stop_loss_z is not None
                and np.isfinite(current_zscore)
                and (
                    (current_position > 0 and current_zscore <= -abs(stop_loss_z))
                    or (current_position < 0 and current_zscore >= abs(stop_loss_z))
                )
            )

            if hit_return_stop or hit_z_stop:
                reason = "return_stop" if hit_return_stop else "z_stop"
                current_position = 0.0
                trade_equity = 1.0
                stopped_until_flat = True

        if reason == "":
            if stopped_until_flat and desired_position == 0:
                stopped_until_flat = False

            if not stopped_until_flat:
                if current_position == 0 and desired_position != 0:
                    current_position = desired_position
                    trade_equity = 1.0
                elif current_position != 0 and desired_position == 0:
                    current_position = 0.0
                    trade_equity = 1.0
                elif current_position != 0 and desired_position != current_position:
                    current_position = desired_position
                    trade_equity = 1.0

        positions.append(current_position)
        stop_reasons.append(reason)

    return (
        pd.Series(positions, index=target_position.index, name="position"),
        pd.Series(stop_reasons, index=target_position.index, name="stop_reason"),
    )


def build_equal_weight_portfolio(backtests: pd.DataFrame) -> pd.DataFrame:
    """Combine pair-window returns into an equal-weight portfolio equity curve.

    Each timestamp allocates equally to the pair strategies that have a return
    observation at that timestamp. This keeps capital usage comparable when the
    number of selected pairs changes across rolling windows.
    """

    if backtests.empty:
        return pd.DataFrame(
            columns=[
                "strategy_return",
                "raw_strategy_return",
                "funding_return",
                "trading_cost",
                "active_pairs",
                "equity_curve",
                "drawdown",
            ],
        )

    required_columns = ["strategy_return"]
    missing_columns = [column for column in required_columns if column not in backtests.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    aggregation = {"strategy_return": "mean"}
    optional_return_columns = ["raw_strategy_return", "funding_return", "trading_cost"]
    for column in optional_return_columns:
        if column in backtests.columns:
            aggregation[column] = "mean"

    grouped = backtests.sort_index().groupby(level=0)
    portfolio = grouped.agg(aggregation).sort_index()
    portfolio["active_pairs"] = grouped.size().reindex(portfolio.index)
    portfolio["equity_curve"] = (1 + portfolio["strategy_return"].fillna(0.0)).cumprod()
    portfolio["drawdown"] = portfolio["equity_curve"] / portfolio["equity_curve"].cummax() - 1

    ordered_columns = [
        "strategy_return",
        "raw_strategy_return",
        "funding_return",
        "trading_cost",
        "active_pairs",
        "equity_curve",
        "drawdown",
    ]
    return portfolio[[column for column in ordered_columns if column in portfolio.columns]]


def run_basis_cointegration_strategy(
    database_path: str | Path,
    symbols: Iterable[str] | None = None,
    cache_path: str | Path | None = None,
    progress_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    skip_sectors: Iterable[str] | None = ("USD Stablecoin", "Fiat-backed Stablecoin"),
    basis_method: str = "log",
    load_min_obs: int = 500,
    formation_window: str | pd.Timedelta = "60D",
    trading_window: str | pd.Timedelta = "7D",
    step: str | pd.Timedelta | None = "7D",
    max_pvalue: float = 0.05,
    max_spread_adf_pvalue: float = 0.05,
    scan_min_obs: int = 1000,
    top_n_per_sector: int | None = 3,
    prefilter_top_pairs_per_sector: int | None = None,
    max_pairs_per_window: int | None = 10,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    fee_rate: float = 0.0004,
    stop_loss_return: float | None = None,
    stop_loss_z: float | None = None,
    n_jobs: int = 1,
    use_cache: bool = True,
    incremental_cache: bool = True,
    show_progress: bool = True,
) -> dict[str, object]:
    """Run the full futures-spot basis cointegration research pipeline."""

    database_path = Path(database_path)
    spot_dir = database_path / "spot"
    futures_dir = database_path / "futures"
    funding_dir = database_path / "funding_rate"
    metadata_path = Path(metadata_path) if metadata_path is not None else (
        database_path / "metadata" / "top10_market_sector_map.json"
    )

    basis, spot_prices, futures_prices = load_basis_matrices(
        spot_dir=spot_dir,
        futures_dir=futures_dir,
        symbols=symbols,
        price_column="close",
        basis_method=basis_method,
        min_obs=load_min_obs,
    )
    funding_rates = load_funding_matrix(
        funding_dir,
        symbols=basis.columns,
        min_obs=1,
    )
    sector_map = load_sector_map(
        metadata_path,
        skip_sectors=skip_sectors,
        available_symbols=basis.columns,
    )

    rolling_pairs = rolling_sector_cointegration_scan(
        prices=basis,
        sector_map=sector_map,
        formation_window=formation_window,
        trading_window=trading_window,
        step=step,
        max_pvalue=max_pvalue,
        max_spread_adf_pvalue=max_spread_adf_pvalue,
        min_obs=scan_min_obs,
        top_n_per_sector=top_n_per_sector,
        prefilter_top_pairs_per_sector=prefilter_top_pairs_per_sector,
        n_jobs=n_jobs,
        cache_path=cache_path,
        progress_path=progress_path,
        use_cache=use_cache,
        incremental_cache=incremental_cache,
        show_progress=show_progress,
    )

    backtests, summaries = walk_forward_basis_backtest(
        basis=basis,
        spot_prices=spot_prices,
        futures_prices=futures_prices,
        rolling_pairs=rolling_pairs,
        funding_rates=funding_rates,
        max_pairs_per_window=max_pairs_per_window,
        entry_z=entry_z,
        exit_z=exit_z,
        fee_rate=fee_rate,
        stop_loss_return=stop_loss_return,
        stop_loss_z=stop_loss_z,
    )
    trading_log = build_trading_log(backtests)
    portfolio = build_equal_weight_portfolio(backtests)

    data_summary = pd.DataFrame(
        [
            {
                "basis_rows": basis.shape[0],
                "basis_symbols": basis.shape[1],
                "funding_rows": funding_rates.shape[0],
                "funding_symbols": funding_rates.shape[1],
                "sectors": len(sector_map),
                "rolling_pairs": len(rolling_pairs),
                "backtested_pair_windows": len(summaries),
                "trades": len(trading_log),
            },
        ],
    )
    summary_table = _build_strategy_summary_table(portfolio, trading_log)

    return {
        "basis": basis,
        "spot_prices": spot_prices,
        "futures_prices": futures_prices,
        "funding_rates": funding_rates,
        "sector_map": sector_map,
        "data_summary": data_summary,
        "rolling_pairs": rolling_pairs,
        "backtests": backtests,
        "summaries": summaries,
        "trading_log": trading_log,
        "portfolio": portfolio,
        "summary_table": summary_table,
        "top_rolling_pairs": rolling_pairs.head(20),
        "top_summaries": _top_pair_summaries(summaries),
        "top_trades": _top_trading_log(trading_log),
    }


def optimize_basis_backtest_parameters(
    database_path: str | Path,
    rolling_pairs_path: str | Path,
    param_grid: Iterable[dict[str, object]],
    symbols: Iterable[str] | None = None,
    basis_method: str = "log",
    load_min_obs: int = 500,
    max_pairs_per_window: int | None = 10,
    fee_rate: float = 0.0004,
    n_jobs: int = 1,
    existing_results_path: str | Path | None = None,
    save_results_path: str | Path | None = None,
    save_results_csv_path: str | Path | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Optimize trading-layer parameters using cached rolling pairs only."""

    database_path = Path(database_path)
    grid = list(param_grid)
    existing_results = _load_existing_optimization_results(existing_results_path)
    missing_grid = _filter_missing_parameter_sets(
        grid,
        existing_results=existing_results,
        fee_rate=fee_rate,
    )

    if not missing_grid:
        output = _sort_optimization_results(existing_results)
        _save_optimization_results(output, save_results_path, save_results_csv_path)
        return output

    rolling_pairs = pd.read_parquet(rolling_pairs_path)
    basis, spot_prices, futures_prices = load_basis_matrices(
        spot_dir=database_path / "spot",
        futures_dir=database_path / "futures",
        symbols=symbols,
        price_column="close",
        basis_method=basis_method,
        min_obs=load_min_obs,
    )
    funding_rates = load_funding_matrix(
        database_path / "funding_rate",
        symbols=basis.columns,
        min_obs=1,
    )

    task_args = [
        (
            parameter_set,
            basis,
            spot_prices,
            futures_prices,
            rolling_pairs,
            funding_rates,
            max_pairs_per_window,
            fee_rate,
        )
        for parameter_set in missing_grid
    ]

    workers = _resolve_n_jobs(n_jobs)
    if workers == 1 or len(task_args) <= 1:
        iterator = _progress(
            task_args,
            total=len(task_args),
            desc="parameter grid",
            enabled=show_progress,
            leave=True,
        )
        rows = [_evaluate_basis_parameter_set(args) for args in iterator]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            iterator = executor.map(_evaluate_basis_parameter_set, task_args)
            iterator = _progress(
                iterator,
                total=len(task_args),
                desc="parameter grid",
                enabled=show_progress,
                leave=True,
            )
            rows = list(iterator)

    rows = [row for row in rows if row]
    new_results = pd.DataFrame(rows)
    output = _merge_optimization_results(
        existing_results=existing_results,
        new_results=new_results,
    )
    _save_optimization_results(output, save_results_path, save_results_csv_path)
    return output


def _load_existing_optimization_results(results_path: str | Path | None) -> pd.DataFrame:
    if results_path is None:
        return pd.DataFrame()
    results_path = Path(results_path)
    if not results_path.exists():
        return pd.DataFrame()
    if results_path.suffix.lower() == ".csv":
        return pd.read_csv(results_path)
    return pd.read_parquet(results_path)


def _filter_missing_parameter_sets(
    param_grid: list[dict[str, object]],
    existing_results: pd.DataFrame,
    fee_rate: float,
) -> list[dict[str, object]]:
    if existing_results.empty:
        return param_grid

    existing_keys = {
        _optimization_parameter_key(row.to_dict(), fee_rate=fee_rate)
        for _, row in existing_results.iterrows()
    }
    return [
        parameter_set
        for parameter_set in param_grid
        if _optimization_parameter_key(parameter_set, fee_rate=fee_rate) not in existing_keys
    ]


def _merge_optimization_results(
    existing_results: pd.DataFrame,
    new_results: pd.DataFrame,
) -> pd.DataFrame:
    if existing_results.empty:
        return _sort_optimization_results(new_results)
    if new_results.empty:
        return _sort_optimization_results(existing_results)

    output = pd.concat([existing_results, new_results], ignore_index=True)
    output["_parameter_key"] = output.apply(
        lambda row: _optimization_parameter_key(row.to_dict(), fee_rate=0.0004),
        axis=1,
    )
    output = output.drop_duplicates("_parameter_key", keep="last").drop(columns="_parameter_key")
    return _sort_optimization_results(output)


def _sort_optimization_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    return results.sort_values(
        ["sharpe", "total_return"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _save_optimization_results(
    results: pd.DataFrame,
    parquet_path: str | Path | None,
    csv_path: str | Path | None,
) -> None:
    if parquet_path is not None:
        parquet_path = Path(parquet_path)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_parquet(parquet_path, index=False)
    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(csv_path, index=False)


def _optimization_parameter_key(
    parameter_set: dict[str, object],
    fee_rate: float,
) -> tuple[object, ...]:
    return (
        _normalise_parameter_value(parameter_set.get("entry_z", 2.0)),
        _normalise_parameter_value(parameter_set.get("exit_z", 0.5)),
        _normalise_parameter_value(parameter_set.get("stop_loss_return")),
        _normalise_parameter_value(parameter_set.get("stop_loss_z")),
        _normalise_parameter_value(parameter_set.get("fee_rate", fee_rate)),
    )


def _normalise_parameter_value(value: object) -> object:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        return round(float(value), 12)
    return value


def _evaluate_basis_parameter_set(args: tuple[object, ...]) -> dict[str, object]:
    (
        parameter_set,
        basis,
        spot_prices,
        futures_prices,
        rolling_pairs,
        funding_rates,
        max_pairs_per_window,
        fee_rate,
    ) = args

    entry_z = float(parameter_set.get("entry_z", 2.0))
    exit_z = float(parameter_set.get("exit_z", 0.5))
    stop_loss_return = parameter_set.get("stop_loss_return")
    stop_loss_z = parameter_set.get("stop_loss_z")
    run_fee_rate = float(parameter_set.get("fee_rate", fee_rate))

    backtests, summaries = walk_forward_basis_backtest(
        basis=basis,
        spot_prices=spot_prices,
        futures_prices=futures_prices,
        rolling_pairs=rolling_pairs,
        funding_rates=funding_rates,
        max_pairs_per_window=max_pairs_per_window,
        entry_z=entry_z,
        exit_z=exit_z,
        fee_rate=run_fee_rate,
        stop_loss_return=(
            None if stop_loss_return is None else float(stop_loss_return)
        ),
        stop_loss_z=None if stop_loss_z is None else float(stop_loss_z),
    )
    trading_log = build_trading_log(backtests)
    portfolio = build_equal_weight_portfolio(backtests)
    summary = summarize_backtest(portfolio) if not portfolio.empty else {
        "total_return": 0.0,
        "annual_return": 0.0,
        "annual_volatility": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
    }

    return {
        **parameter_set,
        **summary,
        "fee_rate": run_fee_rate,
        "backtested_pair_windows": len(summaries),
        "trades": len(trading_log),
        "win_rate": (
            float((trading_log["trade_return"] > 0).mean())
            if not trading_log.empty
            else np.nan
        ),
        "avg_active_pairs": (
            float(portfolio["active_pairs"].mean())
            if not portfolio.empty
            else 0.0
        ),
    }


def plot_portfolio_equity(portfolio: pd.DataFrame, figsize: tuple[int, int] = (12, 7)):
    """Plot portfolio equity and drawdown."""

    if portfolio.empty:
        raise ValueError("Portfolio is empty.")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, height_ratios=[3, 1])
    portfolio["equity_curve"].plot(
        ax=axes[0],
        title="Equal-weight Rolling Basis Cointegration Portfolio",
    )
    axes[0].set_ylabel("Equity")
    portfolio["drawdown"].plot(ax=axes[1], color="tab:red", title="Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Time")
    plt.tight_layout()
    return fig, axes


def build_parameter_plateau(
    optimization: pd.DataFrame,
    metric: str = "sharpe",
    x: str = "exit_z",
    y: str = "entry_z",
    target: dict[str, object] | None = None,
    use_filtered: bool = False,
    filter_query: str = "trades >= 30 and max_drawdown > -0.3 and avg_active_pairs >= 1",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a parameter plateau slice, pivot table, and stability ranking."""

    source = optimization.query(filter_query) if use_filtered else optimization
    if source.empty:
        source = optimization

    plateau_slice = source.copy()
    for column, target_value in (target or {}).items():
        if target_value is None:
            plateau_slice = plateau_slice[plateau_slice[column].isna()]
        else:
            plateau_slice = plateau_slice[plateau_slice[column].eq(target_value)]

    if plateau_slice.empty:
        raise ValueError("No rows match target. Check target values or filters.")

    plateau_table = plateau_slice.pivot_table(
        index=y,
        columns=x,
        values=metric,
        aggfunc="mean",
    )

    plateau_rank = (
        plateau_slice.groupby([y, x], dropna=False)
        .agg(
            median_sharpe=("sharpe", "median"),
            mean_return=("total_return", "mean"),
            worst_drawdown=("max_drawdown", "min"),
            sharpe_std=("sharpe", "std"),
            runs=("sharpe", "size"),
        )
        .reset_index()
    )
    plateau_rank["plateau_score"] = (
        plateau_rank["median_sharpe"] - plateau_rank["sharpe_std"].fillna(0)
    )
    plateau_rank = plateau_rank.sort_values(
        ["plateau_score", "median_sharpe"],
        ascending=False,
    ).reset_index(drop=True)

    return plateau_slice, plateau_table, plateau_rank


def plot_parameter_plateau(
    plateau_table: pd.DataFrame,
    metric: str = "sharpe",
    x: str = "exit_z",
    y: str = "entry_z",
    target: dict[str, object] | None = None,
    figsize: tuple[int, int] = (8, 5),
):
    """Plot a parameter plateau heatmap from a pivot table."""

    if plateau_table.empty:
        raise ValueError("Plateau table is empty.")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(plateau_table.to_numpy(), aspect="auto", origin="lower")
    ax.set_xticks(range(len(plateau_table.columns)))
    ax.set_xticklabels(plateau_table.columns)
    ax.set_yticks(range(len(plateau_table.index)))
    ax.set_yticklabels(plateau_table.index)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{metric} plateau | target={target or {}}")
    fig.colorbar(image, ax=ax, label=metric)
    plt.tight_layout()
    return fig, ax


def save_strategy_result(
    result: dict[str, object],
    output_dir: str | Path,
    strategy_name: str,
    config: dict[str, object] | None = None,
) -> dict[str, Path]:
    """Save a strategy result in a format suitable for strategy stacking."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    portfolio_path = output_dir / "portfolio.parquet"
    trades_path = output_dir / "trading_log.parquet"
    summary_path = output_dir / "summary.csv"
    rolling_pairs_path = output_dir / "rolling_pairs.parquet"
    config_path = output_dir / "config.json"
    manifest_path = output_dir / "manifest.json"

    portfolio = result.get("portfolio", pd.DataFrame())
    trading_log = result.get("trading_log", pd.DataFrame())
    summary_table = result.get("summary_table", pd.DataFrame())
    rolling_pairs = result.get("rolling_pairs", pd.DataFrame())

    if isinstance(portfolio, pd.DataFrame) and not portfolio.empty:
        portfolio.to_parquet(portfolio_path)
    else:
        pd.DataFrame(columns=["strategy_return", "equity_curve", "drawdown"]).to_parquet(
            portfolio_path,
        )

    if isinstance(trading_log, pd.DataFrame):
        trading_log.to_parquet(trades_path, index=False)
    if isinstance(summary_table, pd.DataFrame):
        summary_table.to_csv(summary_path)
    if isinstance(rolling_pairs, pd.DataFrame):
        rolling_pairs.to_parquet(rolling_pairs_path, index=False)

    manifest = {
        "strategy_name": strategy_name,
        "files": {
            "portfolio": portfolio_path.name,
            "trading_log": trades_path.name,
            "summary": summary_path.name,
            "rolling_pairs": rolling_pairs_path.name,
            "config": config_path.name,
        },
        "config": config or {},
    }
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config or {}, file, ensure_ascii=False, indent=2)
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    return {
        "portfolio": portfolio_path,
        "trading_log": trades_path,
        "summary": summary_path,
        "rolling_pairs": rolling_pairs_path,
        "config": config_path,
        "manifest": manifest_path,
    }


def load_strategy_portfolio(path: str | Path) -> pd.DataFrame:
    """Load a saved strategy portfolio from a result directory or parquet file."""

    path = Path(path)
    portfolio_path = path / "portfolio.parquet" if path.is_dir() else path
    return pd.read_parquet(portfolio_path)


def _build_strategy_summary_table(
    portfolio: pd.DataFrame,
    trading_log: pd.DataFrame,
) -> pd.DataFrame:
    if portfolio.empty:
        return pd.DataFrame()

    summary_table = pd.DataFrame([summarize_backtest(portfolio)]).T.rename(columns={0: "value"})
    summary_table.loc["trades", "value"] = len(trading_log)
    summary_table.loc["win_rate", "value"] = (
        (trading_log["trade_return"] > 0).mean() if not trading_log.empty else np.nan
    )
    summary_table.loc["avg_active_pairs", "value"] = portfolio["active_pairs"].mean()
    return summary_table


def _top_pair_summaries(summaries: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    if summaries.empty:
        return summaries
    return summaries.sort_values(["window_id", "sharpe"], ascending=[True, False]).head(n)


def _top_trading_log(trading_log: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    if trading_log.empty or "entry_time" not in trading_log.columns:
        return trading_log
    return trading_log.sort_values("entry_time").head(n)


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


def _resolve_n_jobs(n_jobs: int) -> int:
    if n_jobs == -1:
        return max(os.cpu_count() or 1, 1)
    if n_jobs < -1:
        return max((os.cpu_count() or 1) + 1 + n_jobs, 1)
    return max(n_jobs, 1)


def _progress(
    iterable: Iterable,
    total: int | None = None,
    desc: str | None = None,
    enabled: bool = True,
    leave: bool = False,
):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=leave)


def _scan_pair_task(args: tuple[object, ...]) -> CointegrationResult | None:
    (
        y_symbol,
        x_symbol,
        clean_prices,
        min_obs,
        max_pvalue,
        max_spread_adf_pvalue,
    ) = args

    pair = clean_prices[[y_symbol, x_symbol]].dropna()
    if len(pair) < min_obs:
        return None

    try:
        y_values = pair[y_symbol]
        x_values = pair[x_symbol]
        test_stat, pvalue, _ = coint(y_values, x_values)
        if not np.isfinite(pvalue) or pvalue > max_pvalue:
            return None

        hedge_ratio, intercept = estimate_hedge_ratio(y_values, x_values)
        spread = calculate_spread(
            y_values,
            x_values,
            hedge_ratio=hedge_ratio,
            intercept=intercept,
        ).dropna()
        spread_adf_stat, spread_adf_pvalue, *_ = adfuller(spread)
    except (ValueError, np.linalg.LinAlgError):
        return None

    if not np.isfinite(spread_adf_pvalue) or spread_adf_pvalue > max_spread_adf_pvalue:
        return None

    return CointegrationResult(
        y_symbol=str(y_symbol),
        x_symbol=str(x_symbol),
        pvalue=float(pvalue),
        test_stat=float(test_stat),
        spread_adf_pvalue=float(spread_adf_pvalue),
        spread_adf_stat=float(spread_adf_stat),
        hedge_ratio=hedge_ratio,
        intercept=intercept,
        n_obs=len(pair),
    )


def _to_naive_datetime_index(index: pd.DatetimeIndex | pd.Series) -> pd.DatetimeIndex:
    datetime_index = pd.DatetimeIndex(pd.to_datetime(index))
    if datetime_index.tz is not None:
        datetime_index = datetime_index.tz_convert("UTC").tz_localize(None)
    return datetime_index


def _get_optional_column(frame: pd.DataFrame | None, column: str) -> pd.Series | None:
    if frame is None or frame.empty or column not in frame.columns:
        return None
    return frame[column]


def _align_optional_rate(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(0.0, index=index)

    funding_rate = pd.to_numeric(series, errors="coerce")
    funding_rate.index = _to_naive_datetime_index(funding_rate.index)
    aligned = funding_rate.groupby(level=0).last().reindex(index).fillna(0.0)
    return aligned.astype(float)


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
