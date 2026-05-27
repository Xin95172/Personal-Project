"""Cointegration analysis helpers for pairs-trading research."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import coint
except ImportError:  # pragma: no cover - optional runtime dependency
    sm = None
    coint = None


@dataclass(frozen=True)
class CointegrationResult:
    """Engle-Granger test result for one pair."""

    y_symbol: str
    x_symbol: str
    pvalue: float
    test_stat: float
    hedge_ratio: float
    intercept: float
    n_obs: int


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

    return CointegrationResult(
        y_symbol=y_symbol or str(y.name),
        x_symbol=x_symbol or str(x.name),
        pvalue=float(pvalue),
        test_stat=float(test_stat),
        hedge_ratio=hedge_ratio,
        intercept=intercept,
        n_obs=len(aligned),
    )


def scan_cointegrated_pairs(
    prices: pd.DataFrame,
    max_pvalue: float = 0.05,
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

        if result.pvalue <= max_pvalue:
            results.append(result)

    columns = [
        "y_symbol",
        "x_symbol",
        "pvalue",
        "test_stat",
        "hedge_ratio",
        "intercept",
        "n_obs",
    ]
    output = pd.DataFrame([result.__dict__ for result in results], columns=columns)
    if output.empty:
        return output

    output = output.sort_values(["pvalue", "test_stat"]).reset_index(drop=True)
    return output.head(top_n) if top_n is not None else output


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
    if sm is None or coint is None:
        raise ImportError("statsmodels is required. Install it with: pip install statsmodels")
