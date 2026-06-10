from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_RAW_COLUMNS = {"trading_session", "Open", "Close"}
REQUIRED_FACTOR_COLUMNS = {
    "SOX_open",
    "SOX_close",
    "MOVE_open",
    "MOVE_high",
    "MOVE_low",
    "MOVE_close",
    "Foreign_Opt_Signal_a",
}


class TXDayTradeStrategy:
    """Taiex futures day-session-only strategy migrated from workStrategy/TX."""

    def __init__(self, futures_price: pd.DataFrame):
        self.raw = futures_price.copy()
        self.frame = prepare_tx_frame(self.raw)

    def backtest(
        self,
        point_version: bool = False,
        price_mode: str = "legacy_next_session",
        factor_mode: str = "legacy",
        additive_equity: bool = True,
    ) -> dict[str, pd.DataFrame]:
        backtest = run_tx_daytrade_backtest(
            self.frame,
            point_version=point_version,
            price_mode=price_mode,
            factor_mode=factor_mode,
        )
        portfolio = build_portfolio(
            backtest,
            point_version=point_version,
            additive_equity=additive_equity,
        )
        summary = summarize_tx_backtest(portfolio, backtest=backtest, point_version=point_version)
        for row in summary:
            row["price_mode"] = price_mode
            row["factor_mode"] = factor_mode
            row["equity_mode"] = "additive" if additive_equity else "compound"
        return {
            "backtest": backtest,
            "portfolio": portfolio,
            "summary_table": pd.DataFrame(summary, index=["Strategy", "BuyHold"]),
        }


def prepare_tx_frame(futures_price: pd.DataFrame) -> pd.DataFrame:
    """Prepare a day-session-only TX table."""

    missing = REQUIRED_RAW_COLUMNS - set(futures_price.columns)
    if missing:
        raise ValueError(f"Missing required raw columns: {sorted(missing)}")

    df = futures_price.copy()
    if "trading_session" in df.columns:
        if {"Open_a", "Close_a"}.issubset(df.columns):
            df = df[df["trading_session"] == "position"].copy()
        else:
            day = df[df["trading_session"] == "position"].copy()
            night = df[df["trading_session"] == "after_market"].copy().add_suffix("_a")
            df = pd.concat([day, night], axis=1)

    df["daily_ret"] = df["Close"] / df["Open"] - 1
    df["daily_pnl"] = df["Close"] - df["Open"]
    if {"Open_a", "Close_a"}.issubset(df.columns):
        df["daily_ret_a"] = df["Close_a"] / df["Open_a"] - 1
        df["daily_pnl_a"] = df["Close_a"] - df["Open_a"]
    return df.sort_index()


def calculate_factors(frame: pd.DataFrame, factor_mode: str = "legacy") -> pd.DataFrame:
    """Calculate the factors used by the migrated TX signal logic."""

    valid_factor_modes = {"legacy", "pure_day"}
    if factor_mode not in valid_factor_modes:
        raise ValueError(f"factor_mode must be one of {sorted(valid_factor_modes)}")

    missing = REQUIRED_FACTOR_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            "Missing factor columns. Merge these before running the backtest: "
            f"{sorted(missing)}"
        )

    df = frame.copy()
    df["SOX_ind"] = (df["SOX_close"] / df["SOX_open"] - 1).shift(1).ffill()
    df["MOVE_ind"] = df["MOVE_close"] / df["MOVE_open"] - 1
    df["MOVE_vol"] = (df["MOVE_high"] / df["MOVE_low"] - 1).shift(1)
    if factor_mode == "legacy":
        price_col = "Close_a" if "Close_a" in df.columns else "Close"
        gap_col = "Close_a" if "Close_a" in df.columns else "Open"
    else:
        price_col = "Close"
        gap_col = "Open"
    df["3_ma"] = df[price_col].rolling(window=3).mean()
    df["divergence"] = df[price_col] / df["3_ma"] - 1
    df["gap"] = df[gap_col] / df["Close"].shift(1) - 1
    return df


def apply_signals(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the active strat 3 signal tree from the source notebook."""

    df = frame.copy()
    if "futures_id" in df.columns:
        df = df.dropna(subset=["futures_id"])

    df["pos_day"] = 0.0

    move_split = df["MOVE_ind"] < 0.0001
    sox_split = df["SOX_ind"] < 0.0075
    gap_s_l = df["gap"] < 0.001
    foreign_split = df["Foreign_Opt_Signal_a"] < -0.0035
    divergence_f_r = df["divergence"] < -0.0015

    df.loc[move_split & sox_split & ~gap_s_l, "pos_day"] = 1.0
    df.loc[move_split & ~sox_split, "pos_day"] = 1.0
    df.loc[~move_split & foreign_split, "pos_day"] = -1.0
    df.loc[~move_split & ~foreign_split & ~divergence_f_r, "pos_day"] = 1.0
    return df


def run_tx_daytrade_backtest(
    frame: pd.DataFrame,
    point_version: bool = False,
    price_mode: str = "legacy_next_session",
    factor_mode: str = "legacy",
) -> pd.DataFrame:
    """Run the migrated TX day-trade backtest."""

    valid_price_modes = {"legacy_next_session", "day_open_close"}
    if price_mode not in valid_price_modes:
        raise ValueError(f"price_mode must be one of {sorted(valid_price_modes)}")

    df = calculate_factors(frame, factor_mode=factor_mode)
    df = apply_signals(df)

    if price_mode == "legacy_next_session":
        missing = {"Open", "Open_a"} - set(df.columns)
        if missing:
            raise ValueError(
                "legacy_next_session requires night-session price columns: "
                f"{sorted(missing)}"
            )
        if point_version:
            df["day_return"] = df["Open_a"].shift(-1) - df["Open"]
        else:
            df["day_return"] = df["Open_a"].shift(-1) / df["Open"] - 1
    elif point_version:
        df["day_return"] = df["Close"] - df["Open"]
    else:
        df["day_return"] = df["Close"] / df["Open"] - 1

    if point_version:
        df["buy_hold_return"] = df["Close"] - df["Close"].shift(1)
    else:
        df["buy_hold_return"] = df["Close"] / df["Close"].shift(1) - 1

    df["strategy_return"] = df["day_return"] * df["pos_day"]
    df["benchmark_return"] = df["buy_hold_return"]
    df["cum_strategy_additive"] = df["strategy_return"].cumsum()
    df["cum_benchmark_additive"] = df["benchmark_return"].cumsum()
    return df


def build_portfolio(
    backtest: pd.DataFrame,
    point_version: bool = False,
    additive_equity: bool = True,
) -> pd.DataFrame:
    """Build an aggregate-compatible portfolio table."""

    returns = backtest["strategy_return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    benchmark_returns = backtest["benchmark_return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    portfolio = pd.DataFrame(index=backtest.index)
    portfolio["strategy_return"] = returns
    portfolio["benchmark_return"] = benchmark_returns

    if additive_equity or point_version:
        portfolio["equity_curve"] = returns.cumsum()
        portfolio["drawdown"] = portfolio["equity_curve"] - portfolio["equity_curve"].cummax()
        portfolio["benchmark_equity_curve"] = benchmark_returns.cumsum()
        portfolio["benchmark_drawdown"] = (
            portfolio["benchmark_equity_curve"] - portfolio["benchmark_equity_curve"].cummax()
        )
        portfolio.attrs["equity_mode"] = "additive"
    else:
        portfolio["equity_curve"] = (1 + returns).cumprod()
        portfolio["drawdown"] = portfolio["equity_curve"] / portfolio["equity_curve"].cummax() - 1
        portfolio["benchmark_equity_curve"] = (1 + benchmark_returns).cumprod()
        portfolio["benchmark_drawdown"] = (
            portfolio["benchmark_equity_curve"] / portfolio["benchmark_equity_curve"].cummax() - 1
        )
        portfolio.attrs["equity_mode"] = "compound"

    portfolio["pos_day"] = backtest.get("pos_day", 0.0)
    return portfolio


def summarize_tx_backtest(
    portfolio: pd.DataFrame,
    backtest: pd.DataFrame | None = None,
    point_version: bool = False,
    periods_per_year: int = 252,
) -> list[dict[str, float]]:
    """Summarize TX backtest performance."""

    def summarize_one(
        return_col: str,
        equity_col: str,
        drawdown_col: str,
    ) -> dict[str, float]:
        returns = portfolio[return_col].replace([np.inf, -np.inf], np.nan).dropna()
        if returns.empty:
            return {
                "total_return": 0.0,
                "annual_return": 0.0,
                "annual_volatility": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "trades": 0.0,
                "win_rate": 0.0,
            }

        if point_version or portfolio.attrs.get("equity_mode") == "additive":
            total_return = float(returns.sum())
            annual_return = float(returns.mean() * periods_per_year)
            max_drawdown = float(portfolio[drawdown_col].min())
        else:
            total_return = float(portfolio[equity_col].iloc[-1] - 1)
            annual_return = float((1 + total_return) ** (periods_per_year / len(returns)) - 1)
            max_drawdown = float(portfolio[drawdown_col].min())

        annual_volatility = float(returns.std() * np.sqrt(periods_per_year))
        sharpe = float(annual_return / annual_volatility) if annual_volatility else 0.0

        active_returns = returns[returns != 0]
        wins = active_returns[active_returns > 0]
        losses = active_returns[active_returns < 0]
        gross_win = float(wins.sum())
        gross_loss = float(abs(losses.sum()))

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "trades": float(len(active_returns)),
            "win_rate": float(len(wins) / len(active_returns)) if len(active_returns) else 0.0,
            "profit_factor": float(gross_win / gross_loss) if gross_loss else np.inf,
            "avg_return": float(active_returns.mean()) if len(active_returns) else 0.0,
        }

    return [
        summarize_one("strategy_return", "equity_curve", "drawdown"),
        summarize_one("benchmark_return", "benchmark_equity_curve", "benchmark_drawdown"),
    ]


def save_tx_strategy_result(
    result: dict[str, pd.DataFrame],
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Save TX strategy outputs for later strategy aggregation."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "portfolio": output_dir / "portfolio.parquet",
        "backtest": output_dir / "backtest.parquet",
        "summary": output_dir / "summary.csv",
        "config": output_dir / "config.json",
        "manifest": output_dir / "manifest.json",
    }

    result["portfolio"].to_parquet(paths["portfolio"])
    result["backtest"].to_parquet(paths["backtest"])
    result["summary_table"].to_csv(paths["summary"], encoding="utf-8-sig")

    manifest = {
        "strategy_name": "tx_daytrade",
        "files": {key: str(path) for key, path in paths.items() if key != "manifest"},
    }
    paths["config"].write_text(
        json.dumps(config or {}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return paths
