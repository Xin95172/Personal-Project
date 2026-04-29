"""
從 yfinance 抓取因子所需的原始資料，並計算各項因子分數。
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time


def fetch_factors(ticker_symbol: str) -> dict | None:
    """
    抓取單一 ticker 的所有因子原始資料並計算指標。

    Parameters
    ----------
    ticker_symbol : str
        股票代碼，例如 "AAPL" 或 "2330.TW"

    Returns
    -------
    dict | None
        包含各因子數值的字典；若資料不足則回傳 None。
    """
    tk = yf.Ticker(ticker_symbol)

    # ── 財報資料 ──
    inc = tk.quarterly_income_stmt   # 行: 會計項目, 列: 季度日期（由近到遠）
    bs = tk.quarterly_balance_sheet
    cf = tk.quarterly_cashflow

    if inc.empty or bs.empty or cf.empty:
        return None

    # ── 成長面 ──
    revenue = _safe_loc(inc, "Total Revenue")
    eps = _safe_loc(inc, "Diluted EPS")

    revenue_yoy = _calc_yoy(revenue)          # 最近一季 vs 去年同季
    eps_yoy = _calc_yoy(eps)
    revenue_std = _calc_recent_std(revenue)    # 近 4 季營收標準差

    # ── 財務面 ──
    net_income = _safe_loc(inc, "Net Income")
    equity = _safe_loc(bs, "Stockholders Equity")
    roe_avg = _calc_roe_avg(net_income, equity)  # 近 4 季 ROE 平均

    # ── 創新面 ──
    capex = _safe_loc(cf, "Capital Expenditure")
    capex_ratio = _calc_capex_ratio(capex, revenue)  # |capex| / 營收

    # ── 技術面 ──
    hist = tk.history(period="1y")
    ma_bullish = _check_ma_cross(hist)  # 30MA > 120MA

    return {
        "ticker": ticker_symbol,
        "revenue_yoy": revenue_yoy,
        "eps_yoy": eps_yoy,
        "revenue_std": revenue_std,
        "roe_avg": roe_avg,
        "capex_ratio": capex_ratio,
        "ma_bullish": ma_bullish,
    }


def fetch_factors_batch(tickers: list[str]) -> pd.DataFrame:
    """
    批次抓取多檔 ticker 的因子資料，回傳 DataFrame。
    """
    rows = []
    for t in tickers:
        result = fetch_factors(t)
        if result is not None:
            rows.append(result)
        time.sleep(0.5)
    return pd.DataFrame(rows).set_index("ticker")


# ────────────────────────────────────────
# 內部工具函數
# ────────────────────────────────────────

def _safe_loc(df: pd.DataFrame, label: str) -> pd.Series | None:
    """安全取出某一行，找不到就回傳 None。"""
    if label in df.index:
        return df.loc[label].dropna().sort_index()
    return None


def _calc_yoy(series: pd.Series | None) -> float | None:
    """
    用最近一季 vs 去年同季計算 YoY 成長率。
    yfinance 季報欄位通常有 4~5 季，取 index[0](最新) 和 index[-1](約一年前)。
    """
    if series is None or len(series) < 5:
        return None
    latest = series.iloc[-1]    # 最新一季（sort_index 後最大日期在最後）
    year_ago = series.iloc[-5]  # 去年同季
    if year_ago == 0:
        return None
    return (latest - year_ago) / abs(year_ago)


def _calc_recent_std(series: pd.Series | None) -> float | None:
    """近 4 季營收標準差。"""
    if series is None or len(series) < 4:
        return None
    return series.iloc[-4:].std()


def _calc_roe_avg(
    net_income: pd.Series | None,
    equity: pd.Series | None,
) -> float | None:
    """近 4 季 ROE 的平均值。"""
    if net_income is None or equity is None:
        return None
    # 對齊共同日期
    common = net_income.index.intersection(equity.index)
    if len(common) < 4:
        return None
    common = common.sort_values()[-4:]
    roe = net_income[common] / equity[common]
    return roe.mean()


def _calc_capex_ratio(
    capex: pd.Series | None,
    revenue: pd.Series | None,
) -> float | None:
    """最近一季 |capex| / 營收。"""
    if capex is None or revenue is None:
        return None
    # capex 在 cashflow 裡通常是負值
    latest_capex = abs(capex.iloc[-1])
    latest_rev = revenue.iloc[-1]
    if latest_rev == 0:
        return None
    return latest_capex / latest_rev


def _check_ma_cross(hist: pd.DataFrame) -> bool | None:
    """檢查最新收盤的 30MA 是否 > 120MA。"""
    if hist.empty or len(hist) < 120:
        return None
    close = hist["Close"].squeeze()  # 確保是 Series
    ma30 = close.rolling(30).mean()
    ma120 = close.rolling(120).mean()
    return bool(ma30.iloc[-1] > ma120.iloc[-1])
