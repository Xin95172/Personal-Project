
### 這個是拿來加入模組的
"""

import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import openpyxl
import zipfile
import os
import io
import re
import glob
import platform
from scipy.stats import norm
from scipy.optimize import brentq
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
try:
    import cupy as cp
except ImportError:
    print("無法匯入 cupy，請確認已切換至 GPU 執行階段")

"""GPU 加速的選擇權 IV 計算模組
使用 CuPy + T4 GPU 進行向量化二分法求解 IV
"""

try:
    import cupy as cp
    from cupyx.scipy.special import erfc
except ImportError:
    pass

# ========== 工具函數 ==========

def _date_fmt_no_pad():
    """跨平台日期格式（去除前導零），Windows 用 %#m，Linux 用 %-m"""
    if platform.system() == 'Windows':
        return '%Y/%#m/%#d'
    return '%Y/%-m/%-d'


# ========== 資料準備 ==========

def get_interest_df(file_path='interest.xls'):
    """讀取利率"""
    df = pd.read_excel(file_path, header=2)
    df = df.iloc[:, [0, 13]]
    df = df.loc[2:].copy()
    df.rename(columns={df.columns[0]: '年月', df.columns[1]: 'interest'}, inplace=True)
    df.columns = df.columns.str.strip()
    def convert_roc_to_ad(x):
        s = str(x).zfill(5)
        ad_year = int(s[:3]) + 1911
        month = s[3:]
        return f"{ad_year}/{month}"
    df['年月'] = pd.to_datetime(df['年月'].apply(convert_roc_to_ad), format='%Y/%m')
    df['年月'] = df['年月'].dt.to_period('M')
    df['interest'] = pd.to_numeric(df['interest'], errors='coerce') * 0.01
    return df.reset_index(drop=True)


def get_settle_df(file_path='taifex.csv'):
    """讀期交所結算資料"""
    df = pd.read_csv(file_path)
    df = df[~df['contract'].str.contains('W')]
    df['settledate'] = pd.to_datetime(df['settledate'])
    return df.sort_values('settledate')


def get_twii(start='2022-01-01', end='2024-12-31'):
    """抓台指"""
    df = yf.download('^twii', start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def generate_path_file(start_date, end_date, df_interest, settledate_df, twii_df, output_path='Path.csv'):
    """合成 Path"""
    twii = twii_df.copy()
    if isinstance(twii, pd.DataFrame) and isinstance(twii.columns, pd.MultiIndex):
        twii.columns = twii.columns.get_level_values(0)
    twii.index = pd.to_datetime(twii.index)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    fmt = _date_fmt_no_pad()
    data = []
    for date in dates:
        date_str = date.strftime(fmt)
        file_name = f"OptionsDaily_{date.strftime('%Y_%m_%d')}.csv"
        current_period = date.to_period('M')
        interest_row = df_interest[df_interest['年月'] == current_period]
        rf = interest_row['interest'].values[0] if not interest_row.empty else np.nan
        future_options = settledate_df[settledate_df['settledate'] > date]
        if not future_options.empty:
            future_settle = future_options.iloc[0]
            contract, maturity = future_settle['contract'], (future_settle['settledate'] - date).days
        else:
            contract, maturity = "N/A", np.nan
        try:
            s0 = twii.loc[:date, 'Close'].iloc[-1] if isinstance(twii, pd.DataFrame) else twii.loc[:date].iloc[-1]
        except Exception:
            s0 = np.nan
        data.append([date_str, file_name, s0, maturity, contract, rf])
    df_path = pd.DataFrame(data, columns=['Date', 'File', 'S0', 'Maturity', 'Contract', 'rf'])
    df_path.dropna(subset='S0', inplace=True)
    df_path.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Path 檔案已產生：{output_path}")
    return df_path


# ========== GPU 核心 ==========

def _normal_cdf_gpu(x):
    """GPU 上的標準常態分布 CDF"""
    # 使用 cupyx.scipy.special.erfc 修復錯誤
    return 0.5 * erfc(-x / cp.sqrt(2.0))


def _bs_price_gpu(S, K, T, r, sigma, is_call):
    """GPU 向量化 Black-Scholes 定價"""
    sqrt_T = cp.sqrt(T)
    d1 = (cp.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    call_price = S * _normal_cdf_gpu(d1) - K * cp.exp(-r * T) * _normal_cdf_gpu(d2)
    put_price = K * cp.exp(-r * T) * _normal_cdf_gpu(-d2) - S * _normal_cdf_gpu(-d1)
    return cp.where(is_call, call_price, put_price)


def _find_iv_gpu(prices, S, K, T, r, is_call, n_iter=64):
    """GPU 向量化二分法求解 IV（64 次迭代精度足夠 float64）"""
    n = len(prices)
    low = cp.full(n, 1e-6, dtype=cp.float64)
    high = cp.full(n, 5.0, dtype=cp.float64)

    for _ in range(n_iter):
        mid = (low + high) * 0.5
        bs = _bs_price_gpu(S, K, T, r, mid, is_call)
        too_high = bs > prices
        high = cp.where(too_high, mid, high)
        low = cp.where(too_high, low, mid)

    iv = (low + high) * 0.5
    # 標記無效資料為 NaN
    invalid = (prices <= 0) | (T <= 0)
    return cp.where(invalid, cp.nan, iv)


# ========== ZIP 讀取（支援巢狀） ==========

def _build_file_map(zip_source):
    """建構 file_map，支援巢狀 ZIP"""
    zip_files = glob.glob(os.path.join(zip_source, "*.zip")) if os.path.isdir(zip_source) else [zip_source]
    file_map = {}
    for zp in zip_files:
        with zipfile.ZipFile(zp, 'r') as z:
            for name in z.namelist():
                if name.endswith('.csv'):
                    file_map[name] = (zp, None)
                elif name.endswith('.zip'):
                    try:
                        with z.open(name) as inner_file:
                            inner_bytes = io.BytesIO(inner_file.read())
                            with zipfile.ZipFile(inner_bytes, 'r') as iz:
                                for csv_name in iz.namelist():
                                    if csv_name.endswith('.csv'):
                                        file_map[csv_name] = (zp, name)
                    except Exception:
                        pass
    return file_map


def _read_csv_from_zip(target_zip, target_csv, inner_zip_name=None):
    """從 ZIP（可能巢狀）中讀取 CSV"""
    def try_read(enc):
        with zipfile.ZipFile(target_zip, 'r') as z:
            if inner_zip_name:
                with z.open(inner_zip_name) as inf:
                    buf = io.BytesIO(inf.read())
                    with zipfile.ZipFile(buf, 'r') as iz:
                        with iz.open(target_csv) as f:
                            return pd.read_csv(f, encoding=enc, low_memory=False)
            else:
                with z.open(target_csv) as f:
                    return pd.read_csv(f, encoding=enc, low_memory=False)
    try:
        return try_read('utf-8')
    except Exception:
        return try_read('cp950')


# ========== GPU 批次 IV 計算 ==========

def calculate_iv_batch_gpu(path_csv, zip_source, output_dir='IV_Results', batch_size=50):
    """GPU 加速的批次 IV 計算"""
    os.makedirs(output_dir, exist_ok=True)
    df_path = pd.read_csv(path_csv)
    df_path['Date'] = pd.to_datetime(df_path['Date'])

    # 跳過已處理的日期
    existing = set(os.path.basename(f) for f in glob.glob(os.path.join(output_dir, "IV_*.csv")))

    print("建構檔案索引（含巢狀 ZIP 掃描）...")
    file_map = _build_file_map(zip_source)
    print(f"索引完成，共 {len(file_map)} 個 CSV")

    tasks = []
    for _, row in df_path.iterrows():
        curr_date = row['Date']
        out_name = f"IV_{curr_date.strftime('%Y%m%d')}.csv"
        if out_name in existing:
            continue
        cands = [row['File'],
                 f"o{curr_date.strftime('%Y%m%d')}.csv",
                 f"OptionsDaily_{curr_date.strftime('%Y_%m_%d')}.csv"]
        target_info, target_csv = None, None
        for name, info in file_map.items():
            if any(c in name for c in cands):
                target_info, target_csv = info, name
                break
        if target_csv:
            tasks.append((row, target_info[0], target_csv, target_info[1]))

    print(f"共 {len(tasks)} 天需要處理（已跳過 {len(existing)} 天已完成）")
    if not tasks:
        return

    for batch_start in tqdm(range(0, len(tasks), batch_size), desc="批次處理"):
        batch = tasks[batch_start:batch_start + batch_size]
        all_dfs = []
        day_meta = {}

        for day_idx, (row, target_zip, target_csv, inner_zip_name) in enumerate(batch):
            try:
                df = _read_csv_from_zip(target_zip, target_csv, inner_zip_name)
                df.columns = df.columns.str.strip().str.replace('*', '', regex=False)
                df['商品代號'] = df['商品代號'].astype(str).str.strip()
                df = df[df['商品代號'] == 'TXO']
                df['成交數量(B or S)'] = pd.to_numeric(df['成交數量(B or S)'], errors='coerce')
                df = df[df['成交數量(B or S)'] > 30].copy()
                if df.empty:
                    continue

                df['成交價格'] = pd.to_numeric(df['成交價格'], errors='coerce')
                df['履約價格'] = pd.to_numeric(df['履約價格'], errors='coerce')
                df['買賣權別'] = df['買賣權別'].astype(str).str.strip()

                S, r, T = float(row['S0']), float(row['rf']), float(row['Maturity']) / 365.0
                df['_S'] = S
                df['_r'] = r
                df['_T'] = T
                df['_is_call'] = (df['買賣權別'] == 'C')
                df['_day_idx'] = day_idx
                day_meta[day_idx] = row['Date']
                all_dfs.append(df)
            except Exception as e:
                print(f"讀取失敗: {target_csv} -> {e}")

        if not all_dfs:
            continue

        combined = pd.concat(all_dfs, ignore_index=True)
        prices = cp.asarray(combined['成交價格'].values, dtype=cp.float64)
        S_arr = cp.asarray(combined['_S'].values, dtype=cp.float64)
        K_arr = cp.asarray(combined['履約價格'].values, dtype=cp.float64)
        T_arr = cp.asarray(combined['_T'].values, dtype=cp.float64)
        r_arr = cp.asarray(combined['_r'].values, dtype=cp.float64)
        is_call = cp.asarray(combined['_is_call'].values)

        iv_result = _find_iv_gpu(prices, S_arr, K_arr, T_arr, r_arr, is_call)
        combined['IV'] = cp.asnumpy(iv_result)

        drop_cols = ['_S', '_r', '_T', '_is_call', '_day_idx']
        for day_idx, curr_date in day_meta.items():
            day_data = combined[combined['_day_idx'] == day_idx].drop(columns=drop_cols)
            out_name = f"IV_{curr_date.strftime('%Y%m%d')}.csv"
            day_data.to_csv(os.path.join(output_dir, out_name), index=False, encoding='utf-8-sig')

    print(f"GPU 計算完成，結果已儲存至 {output_dir}/")


# ========== 統計與合併 ==========

def process_iv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.empty or 'IV' not in df.columns:
            return None
        df = df.dropna(subset=['IV'])
        df['買賣權別'] = df['買賣權別'].astype(str).str.strip()
        if df.empty:
            return None

        call_df = df[df['買賣權別'] == 'C']
        put_df = df[df['買賣權別'] == 'P']

        stats = {
            'Call_IV_mean': call_df['IV'].mean(),
            'Call_IV_std': call_df['IV'].std(),
            'Put_IV_mean': put_df['IV'].mean(),
            'Put_IV_std': put_df['IV'].std(),
            'IV_Skew': call_df['IV'].mean() - put_df['IV'].mean(),
            'IV_Spread': df['IV'].max() - df['IV'].min(),
            'IV_Range': df['IV'].std(),
            'PCR': (put_df['成交數量(B or S)'].sum() / call_df['成交數量(B or S)'].sum()
                    if not call_df.empty and call_df['成交數量(B or S)'].sum() > 0 else np.nan)
        }
        return stats
    except Exception:
        return None


def merge_iv_stats_to_path(path_csv='Path.csv', iv_dir='IV_Results'):
    df_path = pd.read_csv(path_csv)
    iv_cols = ['Call_IV_mean', 'Call_IV_std', 'Put_IV_mean', 'Put_IV_std',
               'IV_Skew', 'IV_Spread', 'IV_Range', 'PCR']
    df_path = df_path.drop(columns=[c for c in iv_cols if c in df_path.columns])
    df_path['Date_Key'] = pd.to_datetime(df_path['Date']).dt.strftime('%Y%m%d')
    all_stats = []
    iv_files = glob.glob(os.path.join(iv_dir, "IV_*.csv"))
    print(f"正在整合 {len(iv_files)} 個 IV 結果檔案...")
    for f in tqdm(iv_files):
        match = re.search(r'(\d{8})', os.path.basename(f))
        date_str = match.group(1) if match else None
        if not date_str: continue
        stat = process_iv_file(f)
        if stat:
            stat['Date_Key'] = date_str
            all_stats.append(stat)
    if not all_stats:
        print("警告：沒有找到任何有效的 IV 統計資料。")
        return df_path
    df_stats = pd.DataFrame(all_stats)
    df_final = df_path.merge(df_stats, on='Date_Key', how='left').drop(columns=['Date_Key'])
    df_final.to_csv(path_csv, index=False, encoding='utf-8-sig')
    print(f"整合完成，結果已更新至 {path_csv}")
    return df_final

# 驗證用，跑完確認數字對了再正式跑
import zipfile, io, glob, os

zip_source = '.'
zip_files = glob.glob(os.path.join(zip_source, "*.zip"))
file_map = {}
for zp in zip_files:
    with zipfile.ZipFile(zp, 'r') as z:
        for name in z.namelist():
            if name.endswith('.csv'):
                file_map[name] = (zp, None)
            elif name.endswith('.zip'):
                try:
                    with z.open(name) as inner_file:
                        inner_bytes = io.BytesIO(inner_file.read())
                        with zipfile.ZipFile(inner_bytes, 'r') as iz:
                            for csv_name in iz.namelist():
                                if csv_name.endswith('.csv'):
                                    file_map[csv_name] = (zp, name)
                except:
                    pass

# 統計每個來源 ZIP 的 CSV 數量
from collections import Counter
counts = Counter(os.path.basename(info[0]) for info in file_map.values())
print(f"file_map 總共 {len(file_map)} 個 CSV")
for zp, cnt in counts.items():
    print(f"  {zp}: {cnt} 個")

"""### 設定時間
你自己設定日期，你做 2024 年就設定 '2024-01-01', '2024-12-31'  
我設定 2021 是因為我只有2021的資料
"""

start, end = '2022-01-01', '2024-12-31'
df_interest = get_interest_df()
df_settle = get_settle_df()
twii = get_twii(start, end)
df_path_final = generate_path_file(start, end, df_interest, df_settle, twii)

if os.path.exists('Path.csv'):
    print("確認：Path.csv 已成功建立，可以進行下一步 GPU 計算。")
else:
    print("錯誤：Path.csv 並未產生，請檢查日期範圍或原始資料是否存在。")

"""### 有了 path 之後這個可以幫你算 iv，會新增一個資料夾，叫 IV_Result"""

import os

# 檢查 GPU 是否可用
try:
    import cupy as cp
    print(f"GPU 裝置: {cp.cuda.Device(0)}")
except:
    print("警告：未偵測到 GPU")

# 使用統一的 GPU 版本函式名稱
calculate_iv_batch_gpu('Path.csv', '.', output_dir='IV_Results', batch_size=50)

"""### 有 IV_Result 後，這個會自己算指標，再放回 Path 裡面
我還幫你多加了幾個指標，看不懂就複製上面很多 def 那邊，然後問 ai
"""

df_result = merge_iv_stats_to_path()

from typing import Any
import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy.stats import linregress

pio.renderers.default = 'colab'

def plot(
        df: pd.DataFrame,
        ly: str | list[str] | tuple[str, ...],
        x: str = "index",
        ry: str | list[str] | tuple[str, ...] | None = None,
        ry_dashed: bool = True,
        sub_ly: str | list[str] | tuple[str, ...] | None = None,
        ly_type: str = "line",
):
    if len(df) == 0: return None

    if x == "index":
        x_vals = df.index if isinstance(df.index, pd.DatetimeIndex) else np.arange(len(df))
    else:
        x_vals = df[x]

    # 決定是否有子圖
    rows = 2 if sub_ly else 1
    row_heights = [0.7, 0.3] if sub_ly else [1.0]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        specs=[[{"secondary_y": ry is not None}], [{"secondary_y": False}]] if sub_ly else [[{"secondary_y": ry is not None}]],
        row_heights=row_heights
    )

    # 主圖左軸
    ly_cols = [ly] if isinstance(ly, str) else list(ly)
    colors = ["rgb(31, 119, 180)", "rgb(44, 160, 44)", "rgb(148, 103, 189)", "rgb(255, 127, 14)"]
    for idx, col in enumerate(ly_cols):
        fig.add_trace(go.Scatter(x=x_vals, y=df[col], name=col, line=dict(color=colors[idx % len(colors)])), row=1, col=1, secondary_y=False)

    # 主圖右軸
    if ry:
        ry_cols = [ry] if isinstance(ry, str) else list(ry)
        dash = "dash" if ry_dashed else None
        for col in ry_cols:
            fig.add_trace(go.Scatter(x=x_vals, y=df[col], name=col, line=dict(dash=dash, color="red")), row=1, col=1, secondary_y=True)

    # 子圖 (顯示價格等)
    if sub_ly:
        sub_cols = [sub_ly] if isinstance(sub_ly, str) else list(sub_ly)
        for col in sub_cols:
            fig.add_trace(go.Scatter(x=x_vals, y=df[col], name=col, line=dict(color="gray")), row=2, col=1)

    fig.update_layout(template="plotly_white", height=600 if sub_ly else 500, hovermode="x unified", showlegend=True)
    fig.show()
    return None

import yfinance as yf
twii = yf.download('^twii', start=start, end=end)

if isinstance(twii.columns, pd.MultiIndex):
    twii.columns = twii.columns.get_level_values(0)

df = pd.read_csv('Path.csv')
df.set_index('Date', inplace=True)
df.index = pd.to_datetime(df.index)

full_df = twii.merge(df, left_index=True, right_index=True)

full_df[['Call_IV_mean_demean', 'Put_IV_mean_demean']] = full_df[['Call_IV_mean', 'Put_IV_mean']].sub(full_df[['Call_IV_mean', 'Put_IV_mean']].mean())

plot(full_df, ly='Call_IV_mean', ry='Put_IV_mean', ry_dashed=False, sub_ly='Close')
plot(full_df, ly='Call_IV_mean_demean', ry='Put_IV_mean_demean', ry_dashed=False, sub_ly='Close')

temp_df = full_df.copy()
temp_df['ret'] = (temp_df['Close'] / temp_df['Close'].shift(1)) - 1
temp_df['ret_demean'] = temp_df['ret'].sub(temp_df['ret'].mean())
temp_df['PCR'] = temp_df['PCR'].shift(1)
temp_df = temp_df.sort_values('PCR', ignore_index=True)
temp_df['cum_ret_demean'] = temp_df['ret_demean'].cumsum()
plot(temp_df, ly='cum_ret_demean', ry='PCR')

temp_df = full_df.copy()
temp_df['Call_IV_std_demean'], temp_df['Put_IV_std_demean'] = temp_df['Call_IV_std'] - temp_df['Call_IV_std'].mean(), temp_df['Put_IV_std'] - temp_df['Put_IV_std'].mean()
temp_df['cum_Call_IV_std_demean'], temp_df['cum_Put_IV_std_demean'] = temp_df['Call_IV_std_demean'].cumsum(), temp_df['Put_IV_std_demean'].cumsum()
plot(temp_df, ly='cum_Call_IV_std_demean', ry='cum_Put_IV_std_demean', ry_dashed=False, sub_ly='Close')

# 多空不對稱
plot(full_df, ly='Close', ry='IV_Skew', ry_dashed=False)

temp_df = full_df.copy()
temp_df['ret'] = (temp_df['Close'] / temp_df['Close'].shift(1)) - 1
temp_df['ret_demean'] = temp_df['ret'].sub(temp_df['ret'].mean())
temp_df['IV_Skew'] = temp_df['IV_Skew'].shift(1)
temp_df = temp_df.sort_values('IV_Skew', ignore_index=True)
temp_df['cum_ret_demean'] = temp_df['ret_demean'].cumsum()
plot(temp_df, ly='cum_ret_demean', ry='IV_Skew')

# 市場驚恐指數
temp_df = full_df.copy()
temp_df['IV_Spread_demean'], temp_df['PCR_demean'] = temp_df['IV_Spread'] - temp_df['IV_Spread'].mean(), temp_df['PCR'] - temp_df['PCR'].mean()
temp_df['cum_IV_Spread_demean'], temp_df['cum_PCR_demean'] = temp_df['IV_Spread_demean'].cumsum(), temp_df['PCR_demean'].cumsum()
plot(temp_df, ly='cum_IV_Spread_demean', ry='cum_PCR_demean', ry_dashed=False)

# 看法分歧度
temp_df = full_df.copy()
temp_df['Call_IV_std_demean'], temp_df['Put_IV_std_demean'] = temp_df['Call_IV_std'] - temp_df['Call_IV_std'].mean(), temp_df['Put_IV_std'] - temp_df['Put_IV_std'].mean()
temp_df['cum_Call_IV_std_demean'], temp_df['cum_Put_IV_std_demean'] = temp_df['Call_IV_std_demean'].cumsum(), temp_df['Put_IV_std_demean'].cumsum()
plot(temp_df, ly='cum_Call_IV_std_demean', ry='cum_Put_IV_std_demean', ry_dashed=False)
