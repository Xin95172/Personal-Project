"""
多因子選股月度回測引擎。

設計原則
--------
- 月底換股，因子權重 / 配置方式 / 持股檔數皆可調
- 技術面因子（MA 交叉、動量）從歷史價格動態計算
- 基本面因子（ROE、營收成長等）以靜態分數帶入（yfinance 無法取得歷史財報）
- 配置方式預設等權，可透過 weight_fn 替換
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from typing import Callable


# ────────────────────────────────────────
# 配置
# ────────────────────────────────────────
def _configure_fonts():
    """設定 matplotlib 中文字體。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft JhengHei", "Microsoft YaHei",
        "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


# ────────────────────────────────────────
# 配置函數（可替換）
# ────────────────────────────────────────
def equal_weight(scores: pd.Series, top_n: int) -> pd.Series:
    """等權：選 top_n 檔，每檔 1/top_n。"""
    selected = scores.nlargest(top_n).index
    return pd.Series(1.0 / top_n, index=selected)


def score_weight(scores: pd.Series, top_n: int) -> pd.Series:
    """依分數加權：分數越高權重越大。"""
    selected = scores.nlargest(top_n)
    total = selected.sum()
    if total == 0:
        return pd.Series(1.0 / top_n, index=selected.index)
    return selected / total


# ────────────────────────────────────────
# 價格下載
# ────────────────────────────────────────
import time
import os

def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    chunk_size: int = 100,
    cache_path: str = "prices_cache.pkl"
) -> pd.DataFrame:
    """下載多檔收盤價，回傳 DataFrame (date × ticker)。
    包含分批下載與延遲機制，防止觸發 yfinance 的 Rate Limit。
    並具備本地快取機制，自動補足缺少的標的或時間範圍。
    """
    tickers = list(dict.fromkeys(tickers)) # 確保不重複
    
    # --- 1. 嘗試讀取快取 ---
    cached_df = pd.DataFrame()
    if os.path.exists(cache_path):
        try:
            cached_df = pd.read_pickle(cache_path)
            print(f"發現本地價格快取 (大小: {cached_df.shape})。")
        except Exception as e:
            print(f"讀取快取失敗: {e}")
            
    # --- 2. 判斷需要下載的標的 ---
    missing_tickers = tickers.copy()
    if not cached_df.empty:
        try:
            req_start = pd.Timestamp(start)
            req_end = pd.Timestamp(end)
            cache_start = cached_df.index.min()
            cache_end = cached_df.index.max()
            
            # 給予 7 天容錯，避免因為假日導致第一筆報價較晚
            if req_start < cache_start - pd.Timedelta(days=7):
                print(f"⚠️ 快取起點({cache_start.date()})晚於要求({req_start.date()})，將重新下載。")
            elif req_end > cache_end + pd.Timedelta(days=7):
                print(f"⚠️ 快取終點({cache_end.date()})早於要求({req_end.date()})，將重新下載。")
            else:
                # 日期在範圍內，只需補足不在快取中的股票
                missing_tickers = [t for t in tickers if t not in cached_df.columns]
        except Exception:
            missing_tickers = [t for t in tickers if t not in cached_df.columns]
            
    # --- 3. 執行分批下載 ---
    all_closes = []
    if missing_tickers:
        print(f"需要下載 {len(missing_tickers)} 檔標的資料...")
        for i in range(0, len(missing_tickers), chunk_size):
            chunk = missing_tickers[i:i + chunk_size]
            print(f"  - 下載進度: {i+1} ~ {min(i+chunk_size, len(missing_tickers))} / {len(missing_tickers)}...")
            try:
                data = yf.download(
                    chunk, start=start, end=end,
                    auto_adjust=True, progress=False,
                )
                if not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        if "Close" in data.columns.levels[0]:
                            close = data["Close"]
                        else:
                            close = data
                    else:
                        close = data.to_frame()
                        close.columns = chunk[:1]
                    all_closes.append(close)
            except Exception as e:
                print(f"    下載區塊 {chunk[:3]}... 發生錯誤: {e}")
                
            # 暫停 1.5 秒避免觸發 API 限制
            time.sleep(1.5)
            
    # --- 4. 合併資料並寫回快取 ---
    if missing_tickers:
        if all_closes:
            new_df = pd.concat(all_closes, axis=1)
            new_df = new_df.loc[:, ~new_df.columns.duplicated()]
        else:
            new_df = pd.DataFrame()
            
        # 針對抓不到資料的股票，強制補上 NaN 欄位，這樣下次就不會再重複抓取
        for t in missing_tickers:
            if t not in new_df.columns:
                new_df[t] = np.nan
                
        if not cached_df.empty:
            # 剔除快取中要被新資料覆蓋的欄位，再與新資料合併
            keep_cols = [c for c in cached_df.columns if c not in new_df.columns]
            final_df = pd.concat([cached_df[keep_cols], new_df], axis=1)
        else:
            final_df = new_df
            
        final_df = final_df.sort_index()
        
        try:
            final_df.to_pickle(cache_path)
            print("已更新本地價格快取。")
        except Exception as e:
            print(f"寫入快取失敗: {e}")
    else:
        final_df = cached_df

    if final_df.empty:
        return pd.DataFrame()
        
    # --- 5. 裁切所需範圍回傳 ---
    avail_tickers = [t for t in tickers if t in final_df.columns]
    try:
        s_date = pd.Timestamp(start)
        e_date = pd.Timestamp(end)
        return final_df.loc[s_date:e_date, avail_tickers]
    except Exception:
        return final_df.loc[start:end, avail_tickers]


# ────────────────────────────────────────
# 動態因子計算
# ────────────────────────────────────────
def score_at_date(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    factor_weights: dict[str, float],
    static_scores: pd.DataFrame | None = None,
) -> pd.Series:
    """
    在指定日期計算複合因子分數。

    動態因子（從價格即時計算）：
      - ma_bullish : 30MA > 120MA（布林）
      - momentum_3m : 近 63 交易日報酬
      - momentum_6m : 近 126 交易日報酬

    靜態因子（由外部帶入）：
      - revenue_yoy, eps_yoy, roe_avg, capex_ratio 等
    """
    hist = prices.loc[:date]
    if len(hist) < 120:
        return pd.Series(dtype=float)

    scores = pd.DataFrame(index=prices.columns)

    # 技術面：均線多頭排列
    ma30 = hist.iloc[-30:].mean()
    ma120 = hist.iloc[-120:].mean()
    scores["ma_bullish"] = (ma30 > ma120).astype(float)

    # 動量
    if len(hist) >= 63:
        scores["momentum_3m"] = hist.iloc[-1] / hist.iloc[-63] - 1
    if len(hist) >= 126:
        scores["momentum_6m"] = hist.iloc[-1] / hist.iloc[-126] - 1

    # 合併靜態分數
    if static_scores is not None:
        for col in static_scores.columns:
            if col in factor_weights:
                scores[col] = static_scores[col].reindex(scores.index)

    # 各因子標準化為百分位排名 [0, 1]
    for col in scores.columns:
        if col in factor_weights:
            scores[col] = scores[col].rank(pct=True)

    # 加權合成
    composite = pd.Series(0.0, index=scores.index)
    total_w = 0.0
    for col, w in factor_weights.items():
        if col in scores.columns:
            composite += scores[col].fillna(0) * w
            total_w += w
    if total_w > 0:
        composite /= total_w

    return composite.dropna()


# ────────────────────────────────────────
# 績效指標（日頻）
# ────────────────────────────────────────
TRADING_DAYS_PER_YEAR = 252


def _calc_metrics(rets: pd.Series, name: str) -> dict:
    """從日報酬率序列算績效指標。"""
    if len(rets) == 0:
        return {"name": name}

    cum = (1 + rets).cumprod()
    total = cum.iloc[-1] - 1
    years = len(rets) / TRADING_DAYS_PER_YEAR
    ann_ret = (1 + total) ** (1 / years) - 1 if years > 0 else 0.0
    ann_vol = rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    
    roll_max = cum.cummax()
    drawdown = cum / roll_max - 1
    max_dd = drawdown.min()
    
    # 最大回撤天數
    # 計算 drawdown < 0 的連續天數
    is_dd = (drawdown < 0).astype(int)
    # is_dd == 0 表示該日創新高（或平盤高點）。以其累加作為群組標籤，
    # 每個群組內的加總即為該次連續回撤的天數。
    max_dd_duration = is_dd.groupby((is_dd == 0).cumsum()).sum().max()
    
    # 日報酬一階動差與標準差 (均值與標準差)
    ret_mean = rets.mean()
    ret_std = rets.std()
    
    # 日虧損一階動差與標準差 (僅取 < 0 的報酬)
    losses = rets[rets < 0]
    loss_mean = losses.mean() if len(losses) > 0 else 0.0
    loss_std = losses.std() if len(losses) > 1 else 0.0
    
    # 凱利公式 (Kelly Criterion) - 勝率與盈虧比版本： f = p - (q / b)
    wins = rets[rets > 0]
    p = len(wins) / len(rets) if len(rets) > 0 else 0.0
    q = 1.0 - p
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = abs(rets[rets < 0].mean()) if len(rets[rets < 0]) > 0 else 0.0
    b = (avg_win / avg_loss) if avg_loss > 0 else 0.0
    
    kelly = p - (q / b) if b > 0 else 0.0

    # 月勝率：先聚合成月報酬再算
    monthly = (1 + rets).resample("ME").prod() - 1
    win_rate = (monthly > 0).mean()
    
    return {
        "name": name,
        "累積報酬": f"{total:.2%}",
        "年化報酬": f"{ann_ret:.2%}",
        "年化波動": f"{ann_vol:.2%}",
        "Sharpe": round(sharpe, 2),
        "最大回撤": f"{max_dd:.2%}",
        "最大回撤天數": int(max_dd_duration),
        "月勝率": f"{win_rate:.1%}",
        "Kelly": round(kelly, 4),
        "報酬一階動差": f"{ret_mean:.4%}",
        "報酬標準差": f"{ret_std:.4%}",
        "虧損一階動差": f"{loss_mean:.4%}",
        "虧損標準差": f"{loss_std:.4%}",
        "交易日數": len(rets),
    }


# ────────────────────────────────────────
# 回測主體
# ────────────────────────────────────────
class MultiFactorBacktester:
    """
    多因子選股月度回測。

    Parameters
    ----------
    tickers : list[str]
        投資宇宙
    start, end : str
        回測起訖日 YYYY-MM-DD
    top_n : int
        每期持股檔數
    benchmark : str
        大盤代碼
    factor_weights : dict
        因子名稱 → 權重（合計不需為 1，內部會正規化）
    weight_fn : Callable
        配置函數，簽名 (scores, top_n) → Series of weights
    static_scores : DataFrame | None
        靜態基本面因子分數（index=ticker）
    """

    def __init__(
        self,
        tickers: list[str],
        start: str = "2021-05-01",
        end: str = "2026-05-01",
        top_n: int = 15,
        benchmark: str = "^TWII",
        factor_weights: dict[str, float] | None = None,
        weight_fn: Callable = equal_weight,
        static_scores: pd.DataFrame | None = None,
        fee_rate: float = 0.0,
    ):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.top_n = top_n
        self.benchmark = benchmark
        self.factor_weights = factor_weights or {
            "momentum_3m": 0.35,
            "roe_avg": 0.30,
            "capex_ratio": 0.20,
            "ma_bullish": 0.15,
        }
        self.weight_fn = weight_fn
        self.static_scores = static_scores
        self.fee_rate = fee_rate

        # 結果
        self.portfolio_returns: pd.Series | None = None
        self.benchmark_returns: pd.Series | None = None
        self.holdings_history: dict[pd.Timestamp, pd.Series] = {}
        self.trade_log: list[dict] = []

    # ── 執行 ──────────────────────────
    def run(self) -> pd.DataFrame:
        """執行回測，回傳績效摘要 DataFrame。"""
        _configure_fonts()

        # 多抓 8 個月的緩衝期給 MA120 計算
        buf_start = (
            pd.Timestamp(self.start) - pd.DateOffset(months=8)
        ).strftime("%Y-%m-%d")

        print(f"下載 {len(self.tickers)} 檔股票 ({buf_start} ~ {self.end})...")
        prices = download_prices(self.tickers, buf_start, self.end)
        if prices.empty:
            raise RuntimeError("無法下載任何價格資料")

        print(f"下載大盤 {self.benchmark}...")
        bench_raw = download_prices([self.benchmark], buf_start, self.end)

        # 月底價格
        monthly = prices.resample("ME").last().dropna(how="all")
        rebal_dates = monthly.loc[self.start : self.end].index

        daily_rets_all = []
        last_w = None  # 紀錄前一期期末的權重，用來算換手率和手續費
        for i in range(len(rebal_dates) - 1):
            dt = rebal_dates[i]
            dt_next = rebal_dates[i + 1]

            # 計算因子分數
            scores = score_at_date(
                prices, dt, self.factor_weights, self.static_scores,
            )
            if scores.empty:
                continue

            # 當期必須有報價
            valid = monthly.loc[dt].dropna().index
            scores = scores.reindex(valid).dropna()
            if len(scores) < self.top_n:
                continue

            # 選股 & 配置
            weights = self.weight_fn(scores, self.top_n)
            self.holdings_history[dt] = weights

            # ── 日報酬追蹤（buy-and-hold 到下次換股）──
            period = prices.loc[dt:dt_next, weights.index]
            if len(period) < 2:
                continue
                
            # 計算手續費 (買賣雙邊合計的總變動權重 * fee_rate)
            cost = 0.0
            if self.fee_rate > 0:
                if last_w is not None:
                    all_idx = last_w.index.union(weights.index)
                    diff = weights.reindex(all_idx).fillna(0) - last_w.reindex(all_idx).fillna(0)
                    cost = diff.abs().sum() * self.fee_rate
                else:
                    # 第一次進場
                    cost = weights.sum() * self.fee_rate

            # 記錄各標的該月表現
            period_ret = period.iloc[-1] / period.iloc[0] - 1
            for ticker, w_val in weights.items():
                self.trade_log.append({
                    "進場日期": dt.strftime("%Y-%m-%d"),
                    "出場日期": dt_next.strftime("%Y-%m-%d"),
                    "股票代號": ticker,
                    "初始權重": w_val,
                    "區間報酬": period_ret[ticker]
                })

            # 各股每日報酬
            stock_daily_ret = period.pct_change(fill_method=None).iloc[1:]  # 去掉第一天 NaN
            # 用起始權重逐日計算組合報酬（權重隨價格漂移）
            port_val = pd.Series(1.0, index=stock_daily_ret.index[:0])  # 空
            w = weights.copy()
            for day_idx in range(len(stock_daily_ret)):
                day_ret = stock_daily_ret.iloc[day_idx].fillna(0)
                port_day_ret = (w * day_ret).sum()
                
                # 在這期進場的第一天扣除手續費
                if day_idx == 0:
                    port_day_ret -= cost
                    
                daily_rets_all.append({
                    "date": stock_daily_ret.index[day_idx],
                    "return": port_day_ret,
                })
                # 權重隨價格漂移
                w = w * (1 + day_ret)
                w = w / w.sum()  # 正規化
                
            # 更新期末權重供下期計算換倉成本
            last_w = w

        if not daily_rets_all:
            raise RuntimeError("回測期間無任何有效換股期")

        self.portfolio_returns = (
            pd.DataFrame(daily_rets_all).set_index("date")["return"]
        )
        # 避免重複日期（跨期邊界）
        self.portfolio_returns = self.portfolio_returns[
            ~self.portfolio_returns.index.duplicated(keep="last")
        ]

        # 大盤日報酬
        if isinstance(bench_raw, pd.DataFrame):
            bench_close = bench_raw.iloc[:, 0] if bench_raw.shape[1] >= 1 else bench_raw
        else:
            bench_close = bench_raw
        bench_daily = bench_close.pct_change(fill_method=None).dropna()
        self.benchmark_returns = bench_daily.reindex(
            self.portfolio_returns.index
        ).fillna(0)

        n = len(self.portfolio_returns)
        months = len(self.holdings_history)
        print(f"回測完成：{n} 個交易日，{months} 次換股")
        return self.summary()

    # ── 績效摘要 ──────────────────────
    def summary(self) -> pd.DataFrame:
        """回傳策略 vs 大盤的績效摘要表。"""
        common = self.portfolio_returns.index.intersection(
            self.benchmark_returns.index
        )
        rows = [
            _calc_metrics(self.portfolio_returns.loc[common], "策略"),
            _calc_metrics(self.benchmark_returns.loc[common], "大盤"),
        ]
        return pd.DataFrame(rows).set_index("name")

    # ── 繪圖 ──────────────────────────
    def plot(self) -> plt.Figure:
        """繪製累積報酬 & 回撤圖。"""
        _configure_fonts()
        common = self.portfolio_returns.index.intersection(
            self.benchmark_returns.index
        )
        cum_p = (1 + self.portfolio_returns.loc[common]).cumprod()
        cum_b = (1 + self.benchmark_returns.loc[common]).cumprod()

        fig, axes = plt.subplots(
            2, 1, figsize=(12, 7),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )

        # 累積報酬
        ax = axes[0]
        ax.plot(cum_p, label="策略", linewidth=2)
        ax.plot(cum_b, label=f"大盤 ({self.benchmark})", linewidth=2, alpha=0.7)
        ax.set_title("多因子選股策略回測", fontsize=14, fontweight="bold")
        ax.set_ylabel("累積報酬（倍數）")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # 回撤
        dd = cum_p / cum_p.cummax() - 1
        axes[1].fill_between(dd.index, dd.values, 0, alpha=0.5, color="red")
        axes[1].set_ylabel("回撤")
        axes[1].set_xlabel("日期")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # ── 持股明細 ──────────────────────
    def get_holdings(self, date: pd.Timestamp | None = None) -> pd.Series:
        """取得某期持股。None 取最近一期。"""
        if not self.holdings_history:
            return pd.Series(dtype=float)
        if date is None:
            date = max(self.holdings_history.keys())
        return self.holdings_history.get(date, pd.Series(dtype=float))

    def turnover(self) -> pd.Series:
        """計算每期換手率（更換的股票佔比）。"""
        dates = sorted(self.holdings_history.keys())
        result = {}
        for i in range(1, len(dates)):
            prev = set(self.holdings_history[dates[i - 1]].index)
            curr = set(self.holdings_history[dates[i]].index)
            changed = len(prev.symmetric_difference(curr))
            total = len(prev | curr)
            # 格式化日期為 YYYY-MM，這樣畫長條圖時 X 軸標籤才不會太長
            fmt_date = dates[i].strftime("%Y-%m")
            result[fmt_date] = changed / total if total > 0 else 0.0
        return pd.Series(result, name="turnover")

    def get_trade_log(self, twse_csv_path: str = "twse.csv") -> pd.DataFrame:
        """取得詳細的換股與各標的績效紀錄表格，並嘗試標註產業類別。"""
        if not hasattr(self, "trade_log") or not self.trade_log:
            return pd.DataFrame()
        df = pd.DataFrame(self.trade_log)
        
        # 嘗試加上產業類別
        import os
        if os.path.exists(twse_csv_path):
            try:
                twse = pd.read_csv(twse_csv_path)
                if "yfinance_symbol" in twse.columns and "industry" in twse.columns:
                    mapping = twse.dropna(subset=["yfinance_symbol"])[["yfinance_symbol", "industry"]]
                    mapping = mapping.drop_duplicates(subset=["yfinance_symbol"])
                    df = df.merge(
                        mapping, 
                        how="left", 
                        left_on="股票代號", 
                        right_on="yfinance_symbol"
                    )
                    df.rename(columns={"industry": "產業類別"}, inplace=True)
                    df.drop(columns=["yfinance_symbol"], inplace=True)
                    
                    # 把產業類別移到前面一點
                    cols = df.columns.tolist()
                    cols.insert(3, cols.pop(cols.index("產業類別")))
                    df = df[cols]
            except Exception as e:
                print(f"無法讀取產業對應檔 {twse_csv_path}: {e}")

        # 將數值格式化為百分比
        df["初始權重"] = df["初始權重"].apply(lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else x)
        df["區間報酬"] = df["區間報酬"].apply(lambda x: f"{x:.2%}" if pd.notna(x) and isinstance(x, (int, float)) else "NaN")
        return df

    def plot_industry_weights(self, twse_csv_path: str = "twse.csv") -> plt.Figure | None:
        """繪製各月份產業比重變化的堆疊面積圖。"""
        df = self.get_trade_log(twse_csv_path)
        if df.empty or "產業類別" not in df.columns:
            print("無交易紀錄或缺少產業資料，無法繪圖")
            return None
            
        _configure_fonts()
        
        # 準備資料
        df_calc = df.copy()
        df_calc["權重數值"] = df_calc["初始權重"].str.rstrip('%').astype(float) / 100
        df_calc["進場日期"] = pd.to_datetime(df_calc["進場日期"])
        
        # 統一清理產業名稱 (把字尾的 "業" 拿掉，避免 "光電" 和 "光電業" 分開算)
        df_calc["產業類別"] = df_calc["產業類別"].astype(str).str.replace("業$", "", regex=True)
        
        # 樞紐分析
        industry_weight = df_calc.groupby(["進場日期", "產業類別"])["權重數值"].sum().reset_index()
        pivot = industry_weight.pivot(index="進場日期", columns="產業類別", values="權重數值").fillna(0)
        
        # 為了美觀，將整個回測期間「平均權重小於 2%」的冷門產業合併為「其他」
        mean_weights = pivot.mean()
        # 把低於 2% 且不是叫做「其他」的產業挑出來合併
        small_inds = [c for c in mean_weights[mean_weights < 0.02].index if c != "其他"]
        
        if len(small_inds) > 0:
            if "其他" in pivot.columns:
                pivot["其他"] += pivot[small_inds].sum(axis=1)
            else:
                pivot["其他"] = pivot[small_inds].sum(axis=1)
            pivot = pivot.drop(columns=small_inds)
            
            # 把「其他」移到最後一欄
            cols = [c for c in pivot.columns if c != "其他"] + ["其他"]
            pivot = pivot[cols]
        
        # 繪圖
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 使用堆疊面積圖 (Area Chart)
        pivot.plot(kind='area', stacked=True, ax=ax, alpha=0.85, colormap='tab20')
        
        ax.set_title("策略每月產業配置比重", fontsize=15, fontweight="bold")
        ax.set_ylabel("配置權重")
        ax.set_xlabel("日期")
        ax.set_ylim(0, 1.0)
        
        # 將 Y 軸標籤轉為百分比
        from matplotlib.ticker import PercentFormatter
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        
        # 圖例改放下方，設定成多欄位 (ncol=5) 就不會讓畫面變成長條形
        ax.legend(
            title="產業類別 (微小佔比已歸類為其他)", 
            bbox_to_anchor=(0.5, -0.15), 
            loc='upper center', 
            ncol=6,
            fontsize=10
        )
        
        # 使用 bbox_inches='tight' 可讓儲存或顯示時自動包覆圖例，不再有大片空白
        plt.tight_layout()
        plt.show()
