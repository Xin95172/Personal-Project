from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests


PERFORMANCE_FORMATTERS = {
    "Total Return": "{:.2%}",
    "CAGR": "{:.2%}",
    "Volatility": "{:.2%}",
    "Sharpe": "{:.2f}",
    "Max Drawdown": "{:.2%}",
    "Profit Factor": "{:.2f}",
    "Win Rate": "{:.2%}",
    "Odds": "{:.2f}",
    "Avg Win": "{:.2%}",
    "Avg Loss": "{:.2%}",
    "Avg Return (Exp)": "{:.2%}",
    "Kelly": "{:.2f}",
}


def load_strategy_portfolios(
    results_dir: Path,
    exclude_strategies: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load strategy portfolio parquet files from aggregate/results."""

    exclude_strategies = exclude_strategies or set()
    portfolios = {}
    for strategy_dir in sorted(results_dir.iterdir()):
        if not strategy_dir.is_dir() or strategy_dir.name in exclude_strategies:
            continue

        portfolio_path = strategy_dir / "portfolio.parquet"
        if not portfolio_path.exists():
            continue

        portfolio = pd.read_parquet(portfolio_path).sort_index()
        if "strategy_return" not in portfolio.columns:
            raise KeyError(f"strategy_return not found in {portfolio_path}")
        portfolios[strategy_dir.name] = portfolio
    return portfolios


def calculate_risk_parity_weights(
    returns: pd.DataFrame,
    *,
    lookback: int | None = None,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> pd.Series:
    """Calculate long-only risk parity weights from return covariance."""

    if returns.empty:
        raise ValueError("No returns provided for risk parity weights.")

    clean_returns = returns.replace([np.inf, -np.inf], np.nan)
    if lookback is not None:
        clean_returns = clean_returns.tail(lookback)
    clean_returns = clean_returns.dropna(how="all").fillna(0.0)
    if clean_returns.empty:
        raise ValueError("No usable returns provided for risk parity weights.")

    covariance = clean_returns.cov().to_numpy(dtype=float)
    n_assets = len(clean_returns.columns)
    diagonal = np.diag(covariance)
    if n_assets == 1:
        return pd.Series([1.0], index=clean_returns.columns)

    if (
        not np.isfinite(covariance).all()
        or np.allclose(covariance, 0.0)
        or np.any(diagonal <= 0)
    ):
        volatility = clean_returns.std().replace(0.0, np.nan)
        inv_vol = 1 / volatility
        weights = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if weights.sum() == 0:
            weights = pd.Series(1.0, index=clean_returns.columns)
        weights = weights.clip(lower=min_weight, upper=max_weight)
        return weights / weights.sum()

    from scipy.optimize import minimize

    initial = np.full(n_assets, 1 / n_assets)
    bounds = [(min_weight, max_weight)] * n_assets
    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}

    def objective(weights: np.ndarray) -> float:
        portfolio_variance = float(weights @ covariance @ weights)
        if portfolio_variance <= 0 or not np.isfinite(portfolio_variance):
            return 1e6
        marginal_risk = covariance @ weights
        risk_contribution = weights * marginal_risk / portfolio_variance
        target = np.full(n_assets, 1 / n_assets)
        return float(np.sum((risk_contribution - target) ** 2))

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        volatility = pd.Series(np.sqrt(diagonal), index=clean_returns.columns)
        inv_vol = 1 / volatility.replace(0.0, np.nan)
        weights = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        weights = pd.Series(result.x, index=clean_returns.columns)

    weights = weights.clip(lower=min_weight, upper=max_weight)
    if weights.sum() == 0:
        weights = pd.Series(1.0, index=clean_returns.columns)
    return weights / weights.sum()


def combine_strategy_returns(
    portfolios: dict[str, pd.DataFrame],
    weights: pd.Series | dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Combine strategy return streams into a weighted portfolio."""

    if not portfolios:
        raise ValueError("No strategy portfolios provided.")

    strategy_returns = pd.concat(
        {name: portfolio["strategy_return"] for name, portfolio in portfolios.items()},
        axis=1,
    ).fillna(0.0)

    if weights is None:
        weights = pd.Series(
            {name: 1 / len(strategy_returns.columns) for name in strategy_returns.columns}
        )
    else:
        weights = pd.Series(weights, dtype=float)

    weights = weights.reindex(strategy_returns.columns).fillna(0.0)
    if weights.sum() == 0:
        raise ValueError("At least one strategy weight must be non-zero.")
    weights = weights / weights.sum()

    combined_return = strategy_returns.mul(weights, axis=1).sum(axis=1)
    combined_portfolio = pd.DataFrame(index=strategy_returns.index)
    combined_portfolio["strategy_return"] = combined_return
    combined_portfolio["equity_curve"] = (1 + combined_return).cumprod()
    combined_portfolio["drawdown"] = (
        combined_portfolio["equity_curve"] / combined_portfolio["equity_curve"].cummax() - 1
    )
    return strategy_returns, weights, combined_portfolio


def combine_strategy_returns_drift(
    portfolios: dict[str, pd.DataFrame],
    weights: pd.Series | dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Combine returns with buy-and-hold drift from the initial target weights."""

    if not portfolios:
        raise ValueError("No strategy portfolios provided.")

    strategy_returns = pd.concat(
        {name: portfolio["strategy_return"] for name, portfolio in portfolios.items()},
        axis=1,
    ).fillna(0.0)

    if weights is None:
        weights = pd.Series(
            {name: 1 / len(strategy_returns.columns) for name in strategy_returns.columns}
        )
    else:
        weights = pd.Series(weights, dtype=float)

    target_weights = weights.reindex(strategy_returns.columns).fillna(0.0)
    if target_weights.sum() == 0:
        raise ValueError("At least one strategy weight must be non-zero.")
    target_weights = target_weights / target_weights.sum()

    asset_equity = (1 + strategy_returns).cumprod().mul(target_weights, axis=1)
    portfolio_equity = asset_equity.sum(axis=1)
    portfolio_return = portfolio_equity.pct_change().fillna(portfolio_equity.iloc[0] - 1)
    drift_weights = asset_equity.div(portfolio_equity.replace(0.0, np.nan), axis=0)
    drift_weights = drift_weights.fillna(0.0)
    target_weight_frame = pd.DataFrame(
        np.tile(target_weights.to_numpy(), (len(strategy_returns), 1)),
        index=strategy_returns.index,
        columns=strategy_returns.columns,
    )
    weight_deviation = drift_weights - target_weight_frame

    combined_portfolio = pd.DataFrame(index=strategy_returns.index)
    combined_portfolio["strategy_return"] = portfolio_return
    combined_portfolio["equity_curve"] = portfolio_equity
    combined_portfolio["drawdown"] = portfolio_equity / portfolio_equity.cummax() - 1
    return (
        strategy_returns,
        target_weights,
        combined_portfolio,
        drift_weights,
        target_weight_frame,
        weight_deviation,
    )


def select_portfolios(
    candidates: dict[str, pd.DataFrame],
    names: list[str] | tuple[str, ...] | None,
) -> dict[str, pd.DataFrame]:
    """Select portfolios by name, or return all candidates when names is None."""

    if names is None:
        return candidates

    missing = [name for name in names if name not in candidates]
    if missing:
        raise KeyError(
            f"Unknown portfolio names: {missing}. Available names: {list(candidates)}"
        )
    return {name: candidates[name] for name in names}


def load_benchmarks_for_portfolios(
    portfolios: dict[str, pd.DataFrame],
    *,
    cache_dir: Path,
    note_root: Path,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load benchmark portfolios using the date span covered by existing portfolios."""

    if not portfolios:
        raise ValueError("No portfolios provided to infer benchmark date range.")

    benchmark_start = min(portfolio.index.min() for portfolio in portfolios.values()).date()
    benchmark_end = max(portfolio.index.max() for portfolio in portfolios.values()).date()
    return load_benchmark_portfolios(
        benchmark_start,
        benchmark_end,
        cache_dir=cache_dir,
        note_root=note_root,
        refresh=refresh,
    )


def resolve_combined_weights(
    returns: pd.DataFrame,
    *,
    method: str = "equal",
    custom_weights: pd.Series | dict[str, float] | None = None,
    risk_parity_lookback: int | None = None,
    risk_parity_min_weight: float = 0.0,
    risk_parity_max_weight: float = 1.0,
) -> pd.Series | dict[str, float] | None:
    """Resolve combined portfolio target weights from a named weighting method."""

    if method == "equal":
        return None
    if method == "custom":
        return custom_weights
    if method == "risk_parity":
        return calculate_risk_parity_weights(
            returns,
            lookback=risk_parity_lookback,
            min_weight=risk_parity_min_weight,
            max_weight=risk_parity_max_weight,
        )
    raise ValueError("method must be one of: equal, custom, risk_parity")


def build_combined_backtest(
    strategy_portfolios: dict[str, pd.DataFrame],
    benchmark_portfolios: dict[str, pd.DataFrame],
    *,
    combined_series: list[str] | tuple[str, ...] | None,
    weight_method: str = "equal",
    combine_mode: str = "constant",
    custom_weights: pd.Series | dict[str, float] | None = None,
    risk_parity_lookback: int | None = None,
    risk_parity_min_weight: float = 0.0,
    risk_parity_max_weight: float = 1.0,
) -> dict[str, Any]:
    """Build a combined backtest from selected strategies and benchmark assets."""

    combined_candidates = {
        **strategy_portfolios,
        **benchmark_portfolios,
    }
    combined_inputs = select_portfolios(combined_candidates, combined_series)
    combined_return_inputs = pd.concat(
        {
            name: portfolio["strategy_return"]
            for name, portfolio in combined_inputs.items()
        },
        axis=1,
    ).fillna(0.0)

    selected_weights = resolve_combined_weights(
        combined_return_inputs,
        method=weight_method,
        custom_weights=custom_weights,
        risk_parity_lookback=risk_parity_lookback,
        risk_parity_min_weight=risk_parity_min_weight,
        risk_parity_max_weight=risk_parity_max_weight,
    )

    if combine_mode == "constant":
        strategy_returns, weights, combined_portfolio = combine_strategy_returns(
            combined_inputs,
            weights=selected_weights,
        )
        drift_weights = None
        target_weight_frame = None
        weight_deviation = None
    elif combine_mode == "drift":
        (
            strategy_returns,
            weights,
            combined_portfolio,
            drift_weights,
            target_weight_frame,
            weight_deviation,
        ) = combine_strategy_returns_drift(
            combined_inputs,
            weights=selected_weights,
        )
    else:
        raise ValueError("combine_mode must be one of: constant, drift")

    return {
        "combined_candidates": combined_candidates,
        "combined_inputs": combined_inputs,
        "strategy_returns": strategy_returns,
        "weights": weights,
        "combined_portfolio": combined_portfolio,
        "drift_weights": drift_weights,
        "target_weight_frame": target_weight_frame,
        "weight_deviation": weight_deviation,
    }


def _normalize_cost_rates(
    cost_rates: float | pd.Series | dict[str, float],
    columns: pd.Index,
) -> pd.Series:
    if isinstance(cost_rates, int | float):
        return pd.Series(float(cost_rates), index=columns)
    rates = pd.Series(cost_rates, dtype=float).reindex(columns).fillna(0.0)
    return rates


def _risk_contribution_deviation(
    weights: pd.Series,
    covariance: pd.DataFrame,
) -> float:
    aligned = weights.reindex(covariance.index).fillna(0.0)
    covariance_matrix = covariance.to_numpy(dtype=float)
    weight_array = aligned.to_numpy(dtype=float)
    portfolio_variance = float(weight_array @ covariance_matrix @ weight_array)
    if portfolio_variance <= 0 or not np.isfinite(portfolio_variance):
        return 0.0

    marginal_risk = covariance_matrix @ weight_array
    risk_contribution = weight_array * marginal_risk / portfolio_variance
    target = np.full(len(weight_array), 1 / len(weight_array))
    return float(np.mean(np.abs(risk_contribution - target)))


def backtest_threshold_rebalance(
    returns: pd.DataFrame,
    target_weights: pd.Series | dict[str, float],
    thresholds: list[float] | np.ndarray,
    *,
    cost_rates: float | pd.Series | dict[str, float] = 0.0,
    covariance_lookback: int | None = None,
    keep_details: bool = False,
) -> tuple[pd.DataFrame, dict[float, pd.DataFrame]]:
    """Backtest daily threshold rebalancing with transaction costs."""

    if returns.empty:
        raise ValueError("No returns provided for threshold rebalancing.")

    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0).sort_index()
    target_weights = pd.Series(target_weights, dtype=float).reindex(returns.columns).fillna(0.0)
    if target_weights.sum() == 0:
        raise ValueError("At least one target weight must be non-zero.")
    target_weights = target_weights / target_weights.sum()
    cost_rates = _normalize_cost_rates(cost_rates, returns.columns)

    target_array = target_weights.to_numpy(dtype=float)
    cost_array = cost_rates.to_numpy(dtype=float)
    return_array = returns.to_numpy(dtype=float)

    covariance_returns = returns.tail(covariance_lookback) if covariance_lookback else returns
    covariance = covariance_returns.cov().reindex(
        index=returns.columns,
        columns=returns.columns,
    ).fillna(0.0)
    covariance_array = covariance.to_numpy(dtype=float)
    check_timestamps = returns.groupby(
        pd.to_datetime(returns.index).normalize()
    ).tail(1).index
    check_mask = returns.index.isin(check_timestamps)

    def rc_deviation(weight_array: np.ndarray) -> float:
        portfolio_variance = float(weight_array @ covariance_array @ weight_array)
        if portfolio_variance <= 0 or not np.isfinite(portfolio_variance):
            return 0.0
        marginal_risk = covariance_array @ weight_array
        risk_contribution = weight_array * marginal_risk / portfolio_variance
        risk_target = np.full(len(weight_array), 1 / len(weight_array))
        return float(np.mean(np.abs(risk_contribution - risk_target)))

    column_names = {
        "annual_return": "\u5e74\u5316\u5831\u916c",
        "annual_volatility": "\u5e74\u5316\u6ce2\u52d5",
        "rebalance_count": "\u518d\u5e73\u8861\u6b21\u6578",
        "avg_rc_deviation": "\u5e73\u5747 RC \u504f\u96e2",
    }
    index = returns.index
    thresholds = [float(threshold) for threshold in thresholds]
    periods_per_year = infer_periods_per_year(index)

    rows = []
    details: dict[float, pd.DataFrame] = {}
    for threshold in thresholds:
        asset_value = target_array.copy()
        equity_values = np.empty(len(returns), dtype=float)
        weight_values = np.empty((len(returns), len(returns.columns)), dtype=float)
        turnover_total = 0.0
        rebalance_count = 0
        rc_deviation_sum = 0.0
        rc_deviation_count = 0

        for row_index, period_return in enumerate(return_array):
            asset_value *= 1 + period_return
            portfolio_value = float(asset_value.sum())
            if portfolio_value <= 0:
                current_weights = target_array.copy()
            else:
                current_weights = asset_value / portfolio_value

            if check_mask[row_index]:
                rc_deviation_sum += rc_deviation(current_weights)
                rc_deviation_count += 1

                weight_diff = target_array - current_weights
                if float(np.max(np.abs(weight_diff))) >= threshold:
                    turnover = float(np.sum(np.abs(weight_diff)))
                    cost = float(np.sum(np.abs(weight_diff) * cost_array))
                    portfolio_value *= 1 - cost
                    asset_value = target_array * portfolio_value
                    current_weights = target_array.copy()
                    turnover_total += turnover
                    rebalance_count += 1

            equity_values[row_index] = portfolio_value
            if keep_details:
                weight_values[row_index] = current_weights

        equity = pd.Series(equity_values, index=index, name="equity_curve")
        portfolio_return = equity.pct_change().fillna(equity.iloc[0] - 1)
        drawdown = equity / equity.cummax() - 1
        if keep_details:
            portfolio = pd.DataFrame(
                {
                    "strategy_return": portfolio_return,
                    "equity_curve": equity,
                    "drawdown": drawdown,
                },
                index=index,
            )
            weights_path = pd.DataFrame(
                weight_values,
                index=index,
                columns=returns.columns,
            )
            details[threshold] = pd.concat(
                {
                    "portfolio": portfolio,
                    "weights": weights_path,
                },
                axis=1,
            )

        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
        years = (index[-1] - index[0]) / pd.Timedelta(days=365.25)
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
        annual_volatility = float(portfolio_return.std() * np.sqrt(periods_per_year))
        sharpe = annual_return / annual_volatility if annual_volatility > 0 else np.nan

        rows.append(
            {
                "Threshold": threshold,
                column_names["annual_return"]: annual_return,
                column_names["annual_volatility"]: annual_volatility,
                "Sharpe": sharpe,
                "MDD": float(drawdown.min()),
                "Turnover": turnover_total,
                column_names["rebalance_count"]: rebalance_count,
                column_names["avg_rc_deviation"]: rc_deviation_sum
                / max(rc_deviation_count, 1),
            }
        )

    summary = pd.DataFrame(rows).set_index("Threshold")
    return summary, details


def build_threshold_rebalance_backtest(
    returns: pd.DataFrame,
    target_weights: pd.Series | dict[str, float],
    threshold: float,
    *,
    cost_rates: float | pd.Series | dict[str, float] = 0.0,
    covariance_lookback: int | None = None,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Build one threshold-rebalanced portfolio for use as the combined backtest."""

    summary, details = backtest_threshold_rebalance(
        returns,
        target_weights,
        [threshold],
        cost_rates=cost_rates,
        covariance_lookback=covariance_lookback,
        keep_details=True,
    )
    detail = details[float(threshold)]
    portfolio = detail["portfolio"].copy()
    rebalance_weights = detail["weights"].copy()

    normalized_target = (
        pd.Series(target_weights, dtype=float)
        .reindex(returns.columns)
        .fillna(0.0)
    )
    if normalized_target.sum() == 0:
        raise ValueError("At least one target weight must be non-zero.")
    normalized_target = normalized_target / normalized_target.sum()
    target_weight_frame = pd.DataFrame(
        np.tile(normalized_target.to_numpy(), (len(rebalance_weights), 1)),
        index=rebalance_weights.index,
        columns=rebalance_weights.columns,
    )
    weight_deviation = rebalance_weights - target_weight_frame
    return {
        "summary": summary,
        "portfolio": portfolio,
        "target_weights": normalized_target,
        "rebalance_weights": rebalance_weights,
        "target_weight_frame": target_weight_frame,
        "weight_deviation": weight_deviation,
    }


def infer_periods_per_year(index: pd.Index) -> float:
    """Infer annualization periods from a DatetimeIndex."""

    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 252.0

    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        return 252.0

    median_delta = deltas.median()
    if median_delta <= pd.Timedelta(0):
        return 252.0
    if median_delta >= pd.Timedelta(days=1):
        return 252.0
    return float(pd.Timedelta(days=365) / median_delta)


def infer_observations_per_year(index: pd.Index) -> float:
    """Infer annualization periods from actual observations over elapsed years."""

    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 252.0
    years = float((index.max() - index.min()) / pd.Timedelta(days=365.25))
    if years <= 0:
        return 252.0
    return float((len(index) - 1) / years)


def max_drawdown_duration(equity: pd.Series) -> pd.Timedelta:
    """Return the longest time spent below a previous equity high."""

    if equity.empty:
        return pd.Timedelta(0)

    running_max = equity.cummax()
    underwater = equity < running_max
    max_duration = pd.Timedelta(0)
    start = None
    last_seen = None

    for ts, is_underwater in underwater.items():
        if is_underwater and start is None:
            start = ts
        if not is_underwater and start is not None:
            max_duration = max(max_duration, ts - start)
            start = None
        last_seen = ts

    if start is not None and last_seen is not None:
        max_duration = max(max_duration, last_seen - start)
    return max_duration


def format_duration(delta: pd.Timedelta) -> str:
    days = int(round(delta / pd.Timedelta(days=1)))
    return f"{days} days"


def aggregate_returns_to_daily(returns: pd.Series) -> pd.Series:
    """Compound intraday returns into daily returns."""

    if not isinstance(returns.index, pd.DatetimeIndex):
        return returns
    return (1 + returns).groupby(returns.index.normalize()).prod() - 1


def resample_returns(returns: pd.Series, freq: str | None) -> pd.Series:
    """Compound returns to a target frequency."""

    if freq is None or not isinstance(returns.index, pd.DatetimeIndex):
        return returns
    if freq == "D":
        return aggregate_returns_to_daily(returns)
    counts = returns.resample(freq).count()
    converted = (1 + returns).resample(freq).prod() - 1
    return converted.loc[counts > 0]


def resample_additive_returns(returns: pd.Series, freq: str | None) -> pd.Series:
    """Sum additive returns to a target frequency."""

    if freq is None or not isinstance(returns.index, pd.DatetimeIndex):
        return returns
    if freq == "D":
        return returns.groupby(returns.index.normalize()).sum()
    counts = returns.resample(freq).count()
    converted = returns.resample(freq).sum()
    return converted.loc[counts > 0]


def is_additive_equity(
    returns: pd.Series,
    equity_curve: pd.Series | None,
) -> bool:
    """Return whether an equity curve represents cumulative additive returns."""

    if equity_curve is None:
        return False
    equity = equity_curve.reindex(returns.index).replace([np.inf, -np.inf], np.nan).ffill().dropna()
    if equity.empty:
        return False
    aligned_returns = returns.reindex(equity.index).fillna(0.0)
    return bool(
        np.allclose(
            equity.to_numpy(dtype=float),
            aligned_returns.cumsum().to_numpy(dtype=float),
            rtol=1e-7,
            atol=1e-10,
            equal_nan=False,
        )
    )


def portfolio_to_return_frequency(
    portfolio: pd.DataFrame,
    freq: str | None,
) -> pd.DataFrame:
    """Convert a portfolio return stream to a target return frequency."""

    if freq is None:
        return portfolio

    returns = portfolio["strategy_return"].replace([np.inf, -np.inf], np.nan).dropna()
    additive_equity = is_additive_equity(returns, portfolio.get("equity_curve"))
    if additive_equity:
        converted_returns = resample_additive_returns(returns.astype(float), freq)
    else:
        converted_returns = resample_returns(returns.astype(float), freq)
    converted_returns = converted_returns.replace([np.inf, -np.inf], np.nan).dropna()

    converted = pd.DataFrame(index=converted_returns.index)
    converted["strategy_return"] = converted_returns
    if additive_equity:
        converted["equity_curve"] = converted_returns.cumsum()
        converted["drawdown"] = converted["equity_curve"] - converted["equity_curve"].cummax()
    else:
        converted["equity_curve"] = (1 + converted_returns).cumprod()
        converted["drawdown"] = (
            converted["equity_curve"] / converted["equity_curve"].cummax() - 1
        )
    return converted


def summarize_returns(
    returns: pd.Series,
    periods_per_year: float | None = None,
    equity_curve: pd.Series | None = None,
    drawdown: pd.Series | None = None,
) -> dict[str, float | str]:
    """Calculate the aggregate performance table metrics."""

    returns = returns.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if returns.empty:
        return {
            "Total Return": 0.0,
            "CAGR": 0.0,
            "Volatility": 0.0,
            "Sharpe": 0.0,
            "Max Drawdown": 0.0,
            "Max DD Duration": "0 days",
            "Profit Factor": 0.0,
            "Win Rate": 0.0,
            "Odds": 0.0,
            "Avg Win": 0.0,
            "Avg Loss": 0.0,
            "Avg Return (Exp)": 0.0,
            "Kelly": 0.0,
        }

    periods = periods_per_year or infer_periods_per_year(returns.index)
    if equity_curve is None:
        equity = (1 + returns).cumprod()
        total_return = float(equity.iloc[-1] - 1)
        additive_equity = False
    else:
        equity = equity_curve.reindex(returns.index).replace([np.inf, -np.inf], np.nan).ffill().dropna()
        if equity.empty:
            equity = (1 + returns).cumprod()
            total_return = float(equity.iloc[-1] - 1)
            additive_equity = False
        else:
            aligned_returns = returns.reindex(equity.index).fillna(0.0)
            additive_equity = bool(
                np.allclose(
                    equity.to_numpy(dtype=float),
                    aligned_returns.cumsum().to_numpy(dtype=float),
                    rtol=1e-7,
                    atol=1e-10,
                    equal_nan=False,
                )
            )
        if additive_equity:
            # Additive curves, such as cumsum return curves, are already expressed as total return.
            total_return = float(equity.iloc[-1])
        else:
            total_return = float(equity.iloc[-1] - 1)

    if isinstance(returns.index, pd.DatetimeIndex) and len(returns.index) > 1:
        years = float((returns.index.max() - returns.index.min()) / pd.Timedelta(days=365.25))
    else:
        years = len(returns) / periods if periods else 0.0
    cagr = float((1 + total_return) ** (1 / years) - 1) if years > 0 and total_return > -1 else 0.0
    volatility = float(returns.std() * np.sqrt(periods)) if len(returns) > 1 else 0.0
    sharpe = float(cagr / volatility) if volatility else 0.0

    if drawdown is None:
        if additive_equity:
            drawdown_series = equity - equity.cummax()
        else:
            drawdown_series = equity / equity.cummax() - 1
    else:
        drawdown_series = drawdown.reindex(equity.index).replace([np.inf, -np.inf], np.nan).ffill()
    max_drawdown = float(drawdown_series.min())

    active_returns = returns[returns != 0]
    wins = active_returns[active_returns > 0]
    losses = active_returns[active_returns < 0]
    gross_win = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    profit_factor = float(gross_win / gross_loss) if gross_loss else np.inf
    win_rate = float(len(wins) / len(active_returns)) if len(active_returns) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    odds = float(avg_win / abs(avg_loss)) if avg_loss else np.inf
    daily_returns = aggregate_returns_to_daily(returns)
    active_daily_returns = daily_returns[daily_returns != 0]
    avg_return = (
        float(active_daily_returns.mean()) if len(active_daily_returns) else 0.0
    )
    kelly = float(win_rate - (1 - win_rate) / odds) if np.isfinite(odds) and odds else 0.0

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown,
        "Max DD Duration": format_duration(max_drawdown_duration(equity)),
        "Profit Factor": profit_factor,
        "Win Rate": win_rate,
        "Odds": odds,
        "Avg Win": avg_win,
        "Avg Loss": avg_loss,
        "Avg Return (Exp)": avg_return,
        "Kelly": kelly,
    }


def build_performance_summary(
    portfolios: dict[str, pd.DataFrame],
    *,
    freq: str | None = None,
) -> pd.DataFrame:
    """Build a performance summary table for portfolios keyed by name."""

    return pd.DataFrame(
        {
            name: summarize_returns(
                converted["strategy_return"],
                periods_per_year=(
                    infer_observations_per_year(converted.index)
                    if freq is not None
                    else None
                ),
                equity_curve=converted.get("equity_curve"),
                drawdown=converted.get("drawdown"),
            )
            for name, portfolio in portfolios.items()
            for converted in [portfolio_to_return_frequency(portfolio, freq)]
        }
    ).T


def plot_equity_comparison(
    portfolios: dict[str, pd.DataFrame],
    *,
    include: list[str] | tuple[str, ...] | None = None,
    figsize: tuple[int, int] = (12, 6),
    title: str = "Strategy vs Buy-and-Hold Benchmarks",
    freq: str = "D",
):
    """Plot normalized daily equity curves for strategies and comparison benchmarks."""

    if not portfolios:
        raise ValueError("No portfolios provided.")

    if include is not None:
        missing = [name for name in include if name not in portfolios]
        if missing:
            raise KeyError(f"Unknown portfolio names for plot: {missing}")
        portfolios = {name: portfolios[name] for name in include}

    curves = {}
    for name, portfolio in portfolios.items():
        if portfolio.empty:
            continue
        if "strategy_return" not in portfolio.columns:
            continue

        returns = portfolio["strategy_return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        equity = (1 + returns).cumprod()
        equity = equity.dropna().sort_index()
        if equity.empty:
            continue

        daily_equity = equity.copy()
        daily_equity.index = pd.to_datetime(daily_equity.index).normalize()
        daily_equity = daily_equity.groupby(level=0).last().sort_index()
        curves[name] = daily_equity / daily_equity.iloc[0]

    if not curves:
        raise ValueError("No equity curves can be plotted.")

    import matplotlib.pyplot as plt

    configure_matplotlib_chinese_font()
    start = min(curve.index.min() for curve in curves.values())
    end = max(curve.index.max() for curve in curves.values())
    plot_index = pd.date_range(start=start, end=end, freq=freq)
    equity_curves = pd.concat(
        {name: curve.reindex(plot_index).ffill() for name, curve in curves.items()},
        axis=1,
    )
    fig, ax = plt.subplots(figsize=figsize)
    equity_curves.plot(ax=ax, linewidth=1.6)
    ax.set_title(title)
    ax.set_ylabel("Equity (Start = 1)")
    ax.set_xlabel("Time")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    plt.tight_layout()
    return fig, ax


def configure_matplotlib_chinese_font() -> str | None:
    """Configure matplotlib to use an installed CJK-capable font when available."""

    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    preferred_fonts = [
        "Microsoft JhengHei",
        "Noto Sans TC",
        "Noto Sans CJK TC",
        "MingLiU",
        "SimHei",
        "SimSun",
        "DFKai-SB",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            return font_name
    plt.rcParams["axes.unicode_minus"] = False
    return None


def plot_metric_comparison(
    data: pd.DataFrame,
    *,
    x: str | None = None,
    y: str | list[str] | tuple[str, ...],
    kind: str = "line",
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    marker: str = "o",
    annotate: bool = False,
    x_percent: bool = False,
    y_percent: bool = False,
):
    """Plot one or more metrics against an index or another metric."""

    if data.empty:
        raise ValueError("No data provided for plotting.")
    if isinstance(y, str):
        y_columns = [y]
    else:
        y_columns = list(y)
    missing_y = [column for column in y_columns if column not in data.columns]
    if missing_y:
        raise KeyError(f"Unknown y columns: {missing_y}. Available columns: {list(data.columns)}")

    plot_data = data.copy()
    if x is None:
        x_values = plot_data.index.astype(float)
        x_label = plot_data.index.name or "Index"
    else:
        if x not in plot_data.columns:
            raise KeyError(f"Unknown x column: {x}. Available columns: {list(plot_data.columns)}")
        x_values = plot_data[x]
        x_label = x

    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    configure_matplotlib_chinese_font()
    fig, ax = plt.subplots(figsize=figsize)
    if kind == "line":
        for column in y_columns:
            ax.plot(x_values, plot_data[column], marker=marker, label=column)
    elif kind == "scatter":
        for column in y_columns:
            ax.scatter(x_values, plot_data[column], label=column)
            if annotate:
                for label, x_value, y_value in zip(plot_data.index, x_values, plot_data[column]):
                    ax.annotate(
                        f"{float(label):.1%}" if isinstance(label, float) else str(label),
                        (x_value, y_value),
                        fontsize=8,
                        xytext=(4, 4),
                        textcoords="offset points",
                    )
    elif kind == "bar":
        width = 0.8 / max(len(y_columns), 1)
        x_positions = np.arange(len(plot_data))
        for offset, column in enumerate(y_columns):
            ax.bar(
                x_positions + offset * width,
                plot_data[column],
                width=width,
                label=column,
            )
        ax.set_xticks(x_positions + width * (len(y_columns) - 1) / 2)
        ax.set_xticklabels([f"{value:.1%}" if x_percent else str(value) for value in x_values])
    else:
        raise ValueError("kind must be one of: line, scatter, bar")

    if kind != "bar" and x_percent:
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    if y_percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    ax.set_title(title or f"{', '.join(y_columns)} vs {x_label}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(", ".join(y_columns))
    if len(y_columns) > 1 or kind == "scatter":
        ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_weight_drift(
    weights: pd.DataFrame,
    *,
    include: list[str] | tuple[str, ...] | None = None,
    title: str = "Portfolio Weight Drift",
    figsize: tuple[int, int] = (12, 6),
    as_area: bool = True,
):
    """Plot portfolio weights over time."""

    if weights is None or weights.empty:
        raise ValueError(
            "No weight data provided. Set COMBINE_MODE='drift' and rerun, "
            "or load results/combined/drift_weights.parquet."
        )

    plot_data = weights.copy()
    if include is not None:
        missing = [column for column in include if column not in plot_data.columns]
        if missing:
            raise KeyError(f"Unknown weight columns: {missing}. Available columns: {list(plot_data.columns)}")
        plot_data = plot_data.loc[:, list(include)]

    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    configure_matplotlib_chinese_font()
    fig, ax = plt.subplots(figsize=figsize)
    if as_area:
        plot_data.plot.area(ax=ax, stacked=True, alpha=0.8)
    else:
        plot_data.plot(ax=ax, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Weight")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    plt.tight_layout()
    return fig, ax


def plot_weight_deviation_heatmap(
    deviation: pd.DataFrame,
    *,
    include: list[str] | tuple[str, ...] | None = None,
    freq: str = "ME",
    absolute: bool = True,
    title: str = "Weight Deviation Heatmap",
    figsize: tuple[int, int] = (12, 5),
):
    """Plot a heatmap of average weight deviation by period."""

    if deviation is None or deviation.empty:
        raise ValueError(
            "No weight deviation data provided. Set COMBINE_MODE='drift' and rerun, "
            "or load results/combined/weight_deviation.parquet."
        )

    plot_data = deviation.copy()
    if include is not None:
        missing = [column for column in include if column not in plot_data.columns]
        if missing:
            raise KeyError(f"Unknown deviation columns: {missing}. Available columns: {list(plot_data.columns)}")
        plot_data = plot_data.loc[:, list(include)]
    if absolute:
        plot_data = plot_data.abs()

    plot_data = plot_data.copy()
    plot_data.index = pd.to_datetime(plot_data.index)
    heatmap_data = plot_data.resample(freq).mean().T

    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    configure_matplotlib_chinese_font()
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(heatmap_data, aspect="auto", cmap="Reds" if absolute else "coolwarm")
    ax.set_title(title)
    ax.set_ylabel("Asset")
    ax.set_xlabel("Period")
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)

    if len(heatmap_data.columns) <= 18:
        tick_positions = np.arange(len(heatmap_data.columns))
    else:
        tick_positions = np.linspace(0, len(heatmap_data.columns) - 1, 12, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [pd.Timestamp(heatmap_data.columns[i]).strftime("%Y-%m") for i in tick_positions],
        rotation=45,
        ha="right",
    )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    colorbar.set_label("Abs Deviation" if absolute else "Deviation")
    plt.tight_layout()
    return fig, ax


def rebuild_drift_outputs(combined_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rebuild saved drift outputs from strategy returns and target weights."""

    paths = {
        "drift_weights": combined_dir / "drift_weights.parquet",
        "target_weights": combined_dir / "target_weights.parquet",
        "weight_deviation": combined_dir / "weight_deviation.parquet",
        "strategy_returns": combined_dir / "strategy_returns.parquet",
        "weights": combined_dir / "weights.csv",
    }
    if not paths["strategy_returns"].exists() or not paths["weights"].exists():
        raise FileNotFoundError(
            "strategy_returns.parquet and weights.csv are required to rebuild drift outputs."
        )

    strategy_returns = pd.read_parquet(paths["strategy_returns"])
    weights = pd.read_csv(paths["weights"], index_col=0).iloc[:, 0]
    portfolios = {
        name: pd.DataFrame({"strategy_return": strategy_returns[name]})
        for name in strategy_returns.columns
    }
    (
        _,
        _,
        _,
        drift_weights,
        target_weights,
        weight_deviation,
    ) = combine_strategy_returns_drift(portfolios, weights=weights)

    drift_weights.to_parquet(paths["drift_weights"])
    target_weights.to_parquet(paths["target_weights"])
    weight_deviation.to_parquet(paths["weight_deviation"])
    return drift_weights, target_weights, weight_deviation


def load_drift_outputs(
    combined_dir: Path,
    *,
    rebuild_if_invalid: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load saved drift outputs, optionally rebuilding corrupt or missing files."""

    paths = {
        "drift_weights": combined_dir / "drift_weights.parquet",
        "target_weights": combined_dir / "target_weights.parquet",
        "weight_deviation": combined_dir / "weight_deviation.parquet",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        if rebuild_if_invalid:
            return rebuild_drift_outputs(combined_dir)
        raise FileNotFoundError(
            f"Missing drift output files: {missing}. Set COMBINE_MODE='drift' and rerun."
        )

    try:
        return (
            pd.read_parquet(paths["drift_weights"]),
            pd.read_parquet(paths["target_weights"]),
            pd.read_parquet(paths["weight_deviation"]),
        )
    except Exception:
        if rebuild_if_invalid:
            return rebuild_drift_outputs(combined_dir)
        raise


def _portfolio_from_returns(returns: pd.Series) -> pd.DataFrame:
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    portfolio = pd.DataFrame(index=returns.index)
    portfolio["strategy_return"] = returns
    portfolio["equity_curve"] = (1 + returns).cumprod()
    portfolio["drawdown"] = portfolio["equity_curve"] / portfolio["equity_curve"].cummax() - 1
    return portfolio


def _read_cached_benchmark(cache_path: Path) -> pd.DataFrame | None:
    if not cache_path.exists():
        return None
    cached = pd.read_parquet(cache_path).sort_index()
    return cached if not cached.empty else None


def _apply_taiwan_split_adjustment(
    returns: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    stock_id: str,
) -> pd.Series:
    """Adjust raw Taiwan stock/ETF returns for split reference prices."""

    from FinMind.data import DataLoader

    loader = DataLoader()
    split_events = loader.taiwan_stock_split_price(
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
    )
    if split_events.empty:
        return returns

    split_events = split_events.loc[split_events["stock_id"].astype(str) == stock_id].copy()
    if split_events.empty:
        return returns

    adjusted = returns.copy()
    split_events["date"] = pd.to_datetime(split_events["date"]).dt.normalize()
    for _, event in split_events.iterrows():
        event_date = event["date"]
        if event_date not in adjusted.index:
            continue

        before_price = float(event["before_price"])
        after_price = float(event["after_price"])
        if before_price <= 0 or after_price <= 0:
            continue

        adjusted.loc[event_date] = (1 + adjusted.loc[event_date]) * (
            before_price / after_price
        ) - 1
    return adjusted


def fetch_0050_buy_hold(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    note_root: Path,
    cache_path: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch/cache split-adjusted 0050 buy-and-hold daily returns."""

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if not refresh:
        cached = _read_cached_benchmark(cache_path)
        if cached is not None and cached.index.min() <= start_ts and cached.index.max() >= end_ts:
            return cached.loc[start_ts:end_ts]

    import sys

    if str(note_root) not in sys.path:
        sys.path.append(str(note_root))
    from module.get_info_FinMind import FinMindClient

    fm = FinMindClient()
    fm.initialize_frame(
        stock_id="0050",
        start_time=start_ts.strftime("%Y-%m-%d"),
        end_time=end_ts.strftime("%Y-%m-%d"),
    )
    price = fm.get_stock()
    if price.empty or "Close" not in price.columns:
        raise RuntimeError("No 0050 close price data fetched from FinMind.")

    close = price["Close"].astype(float).sort_index()
    returns = close.pct_change()
    returns = _apply_taiwan_split_adjustment(
        returns,
        start_ts,
        end_ts,
        stock_id="0050",
    )
    portfolio = _portfolio_from_returns(returns)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    portfolio.to_parquet(cache_path)
    return portfolio


def fetch_yahoo_buy_hold(
    symbol: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    cache_path: Path,
    refresh: bool = False,
    fallback_symbol: str | None = None,
) -> pd.DataFrame:
    """Fetch/cache a Yahoo Finance buy-and-hold daily return series."""

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if not refresh:
        cached = _read_cached_benchmark(cache_path)
        if cached is not None and cached.index.min() <= start_ts and cached.index.max() >= end_ts:
            return cached.loc[start_ts:end_ts]

    payload = None
    errors = []
    symbols = [symbol]
    if fallback_symbol and fallback_symbol not in symbols:
        symbols.append(fallback_symbol)

    period1 = int(start_ts.timestamp())
    # Yahoo period2 is exclusive; add one day so end date is included.
    period2 = int((end_ts + pd.Timedelta(days=1)).timestamp())
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
    }

    for candidate_symbol in symbols:
        for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
            url = (
                f"https://{host}/v8/finance/chart/{quote(candidate_symbol, safe='')}"
                f"?period1={period1}&period2={period2}"
                "&interval=1d&events=history&includeAdjustedClose=true"
            )
            try:
                response = requests.get(url, headers=headers, timeout=20)
                response.raise_for_status()
                candidate_payload = response.json()
                result = candidate_payload.get("chart", {}).get("result")
                if result:
                    payload = candidate_payload
                    break
                errors.append(f"{candidate_symbol}@{host}: empty chart result")
            except Exception as exc:
                errors.append(f"{candidate_symbol}@{host}: {exc}")
        if payload is not None:
            break

    if payload is None:
        raise RuntimeError(
            f"No Yahoo chart data returned for {symbol}. Tried: {'; '.join(errors)}"
        )

    data = result[0]
    timestamps = data.get("timestamp", [])
    quote_data = data.get("indicators", {}).get("quote", [{}])[0]
    adjclose_data = data.get("indicators", {}).get("adjclose", [{}])[0]
    close_values = adjclose_data.get("adjclose") or quote_data.get("close")
    if not timestamps or not close_values:
        raise RuntimeError(f"No Yahoo close data returned for {symbol}.")

    index = pd.to_datetime(timestamps, unit="s").normalize()
    close = pd.Series(close_values, index=index, dtype=float).dropna().sort_index()
    close = close.loc[(close.index >= start_ts) & (close.index <= end_ts)]
    returns = close.pct_change()
    portfolio = _portfolio_from_returns(returns)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    portfolio.to_parquet(cache_path)
    return portfolio


def fetch_finmind_us_buy_hold(
    symbol: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    cache_path: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch/cache a FinMind US stock or ETF buy-and-hold daily return series."""

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if not refresh:
        cached = _read_cached_benchmark(cache_path)
        if cached is not None and cached.index.min() <= start_ts and cached.index.max() >= end_ts:
            return cached.loc[start_ts:end_ts]

    from FinMind.data import DataLoader

    loader = DataLoader()
    price = loader.us_stock_price(
        stock_id=symbol,
        start_date=start_ts.strftime("%Y-%m-%d"),
        end_date=end_ts.strftime("%Y-%m-%d"),
    )
    if price.empty:
        raise RuntimeError(f"No FinMind US price data fetched for {symbol}.")

    price["date"] = pd.to_datetime(price["date"])
    price = price.set_index("date").sort_index()
    close_column = "Adj_Close" if "Adj_Close" in price.columns else "Close"
    close = price[close_column].astype(float)
    returns = close.pct_change()
    portfolio = _portfolio_from_returns(returns)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    portfolio.to_parquet(cache_path)
    return portfolio


def fetch_anue_fund_buy_hold(
    fund_id: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    cache_path: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch/cache Anue Fund NAV buy-and-hold daily returns."""

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if not refresh:
        cached = _read_cached_benchmark(cache_path)
        if cached is not None and cached.index.min() <= start_ts and cached.index.max() >= end_ts:
            return cached.loc[start_ts:end_ts]

    url = "https://www.anuefund.com/anuefundApi/FundDetail/Price"
    payload = {
        "fundID": fund_id,
        "priceENUM": "NAVHIS",
        "strDate": start_ts.strftime("%Y/%m/%d"),
        "endDate": end_ts.strftime("%Y/%m/%d"),
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Content-Type": "application/json",
        "Referer": f"https://www.anuefund.com/fund/detail/{fund_id}",
    }
    response = requests.post(url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    data_text = response.json().get("data")
    if not data_text:
        raise RuntimeError(f"No Anue Fund NAV data returned for {fund_id}.")

    import json

    data = json.loads(data_text)
    if not data:
        raise RuntimeError(f"Empty Anue Fund NAV data returned for {fund_id}.")

    nav = pd.Series(
        [float(row[1]) for row in data],
        index=pd.to_datetime([int(row[0]) for row in data], unit="ms").normalize(),
        dtype=float,
        name=fund_id,
    ).sort_index()
    nav = nav.loc[(nav.index >= start_ts) & (nav.index <= end_ts)]
    if nav.empty:
        raise RuntimeError(f"No Anue Fund NAV data in requested range for {fund_id}.")

    returns = nav.pct_change()
    portfolio = _portfolio_from_returns(returns)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    portfolio.to_parquet(cache_path)
    return portfolio


def load_benchmark_portfolios(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    cache_dir: Path,
    note_root: Path,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load benchmark buy-and-hold portfolios used for aggregate comparison."""

    benchmarks = {
        "0050_buy_hold": fetch_0050_buy_hold(
            start,
            end,
            note_root=note_root,
            cache_path=cache_dir / "0050_buy_hold.parquet",
            refresh=refresh,
        ),
    }
    try:
        benchmarks["sp500_buy_hold"] = fetch_yahoo_buy_hold(
            "^GSPC",
            start,
            end,
            cache_path=cache_dir / "sp500_buy_hold.parquet",
            refresh=refresh,
            fallback_symbol="SPY",
        )
    except Exception:
        benchmarks["sp500_buy_hold"] = fetch_finmind_us_buy_hold(
            "SPY",
            start,
            end,
            cache_path=cache_dir / "sp500_buy_hold.parquet",
            refresh=refresh,
        )
    benchmarks["gold_buy_hold"] = fetch_yahoo_buy_hold(
        "GC=F",
        start,
        end,
        cache_path=cache_dir / "gold_buy_hold.parquet",
        refresh=refresh,
        fallback_symbol="GLD",
    )
    benchmarks["qqq_buy_hold"] = fetch_yahoo_buy_hold(
        "QQQ",
        start,
        end,
        cache_path=cache_dir / "qqq_buy_hold.parquet",
        refresh=refresh,
    )
    benchmarks["bitcoin_buy_hold"] = fetch_yahoo_buy_hold(
        "BTC-USD",
        start,
        end,
        cache_path=cache_dir / "bitcoin_buy_hold.parquet",
        refresh=refresh,
    )
    benchmarks["fidelity_global_income_buy_hold"] = fetch_anue_fund_buy_hold(
        "B14248",
        start,
        end,
        cache_path=cache_dir / "fidelity_global_income_buy_hold.parquet",
        refresh=refresh,
    )
    return benchmarks


def save_combined_outputs(
    combined_dir: Path,
    combined_portfolio: pd.DataFrame,
    summary: pd.DataFrame,
    weights: pd.Series,
    strategy_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame | None = None,
    drift_weights: pd.DataFrame | None = None,
    target_weights: pd.DataFrame | None = None,
    weight_deviation: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Persist aggregate outputs."""

    combined_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "portfolio": combined_dir / "portfolio.parquet",
        "summary": combined_dir / "summary.csv",
        "weights": combined_dir / "weights.csv",
        "strategy_returns": combined_dir / "strategy_returns.parquet",
    }
    if benchmark_returns is not None:
        paths["benchmark_returns"] = combined_dir / "benchmark_returns.parquet"
    if drift_weights is not None:
        paths["drift_weights"] = combined_dir / "drift_weights.parquet"
    if target_weights is not None:
        paths["target_weights"] = combined_dir / "target_weights.parquet"
    if weight_deviation is not None:
        paths["weight_deviation"] = combined_dir / "weight_deviation.parquet"

    combined_portfolio.to_parquet(paths["portfolio"])
    summary.to_csv(paths["summary"])
    weights.rename("weight").to_csv(paths["weights"])
    strategy_returns.to_parquet(paths["strategy_returns"])
    if benchmark_returns is not None:
        benchmark_returns.to_parquet(paths["benchmark_returns"])
    if drift_weights is not None:
        drift_weights.to_parquet(paths["drift_weights"])
    if target_weights is not None:
        target_weights.to_parquet(paths["target_weights"])
    if weight_deviation is not None:
        weight_deviation.to_parquet(paths["weight_deviation"])
    return paths
