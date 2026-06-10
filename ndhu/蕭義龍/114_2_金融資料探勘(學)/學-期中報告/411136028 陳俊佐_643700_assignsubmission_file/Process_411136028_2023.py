import pandas as pd
import numpy as np
from scipy.stats import norm # Keep for potential non-JIT use or other context
import os
from datetime import datetime
from google.colab import drive
from numba import jit # Added numba import
import math # Import math for erf and sqrt

# ============================================
# 1. 掛載 Google Drive (確保在 Colab 環境中可以存取您的雲端硬碟)
# ============================================
drive.mount('/content/drive')

# ============================================
# 2. Black-Scholes (Call) - JIT Optimized
# ============================================
@jit(nopython=True)
def bs_call_price(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0.0)

    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
d2 = d1 - sigma*math.sqrt(T)

    # Numba-compatible standard normal CDF using math.erf
    cdf_d1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
    cdf_d2 = 0.5 * (1 + math.erf(d2 / math.sqrt(2)))

    return S * cdf_d1 - K * np.exp(-r*T) * cdf_d2

# ============================================
# 2.1 Black-Scholes (Put) - JIT Optimized
# ============================================
@jit(nopython=True)
def bs_put_price(S, K, T, r, sigma):
    if T <= 0:
        return max(K - S, 0.0)

    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
d2 = d1 - sigma*math.sqrt(T)

    # Numba-compatible standard normal CDF using math.erf
    cdf_neg_d1 = 0.5 * (1 + math.erf(-d1 / math.sqrt(2))) # N(-d1)
    cdf_neg_d2 = 0.5 * (1 + math.erf(-d2 / math.sqrt(2))) # N(-d2)

    return K * np.exp(-r*T) * cdf_neg_d2 - S * cdf_neg_d1


# ============================================
# 3. 二分法求 IV - JIT Optimized (Generic for Call/Put)
# ============================================
@jit(nopython=True)
def implied_vol_bisection_generic(C, S, K, T, r, option_type):
    low = 1e-6
    high = 5.0
    tol = 1e-5
    max_iter = 100

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        price = 0.0
        if option_type == ord('C'): # ASCII for 'C'
            price = bs_call_price(S, K, T, r, mid)
        elif option_type == ord('P'): # ASCII for 'P'
            price = bs_put_price(S, K, T, r, mid)
        else:
            return np.nan # Should not happen with proper filtering

        if abs(price - C) < tol:
            return mid

        if price > C:
            high = mid
        else:
            low = mid

    return np.nan

# ============================================
# 4. 自動讀取 CSV（處理編碼問題）
# ============================================
def read_csv_auto(path):
    for enc in ['utf-8', 'big5', 'cp950']:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except:
            continue
    raise ValueError(f"Cannot read file: {path}")

# ============================================
# 5. 設定路徑（請改成你的）
# ============================================
# 索引檔案 `Path_教學_0409.csv` 的路徑
index_file_path_prefix = "/content/drive/MyDrive/"
index_file = os.path.join(index_file_path_prefix, "資料來源1_含Rf.csv")

# 每日選擇權檔案 `OptionsDaily_YYYY_MM_DD.csv` 的路徑
daily_options_files_path_prefix = "/content/drive/MyDrive/新增資料夾/"

index_df = read_csv_auto(index_file)

# 確保欄位名稱去除前後空白
index_df.columns = index_df.columns.str.strip()

# ============================================
# 6. 主迴圈 (Modified for Call/Put IV and PCR)
# ============================================
all_daily_results = [] # To store dicts of daily stats and PCR

for i, row in index_df.iterrows():
    try:
        date = pd.to_datetime(row['Date'])
        file_name = row['File']
        S0 = float(row['S0'])
        contract = str(row['Contract']).strip()
        r = float(row['Rf'])
        maturity_days = row['Maturity']
        T = maturity_days / 365

        file_path = os.path.join(daily_options_files_path_prefix, file_name)
        print(f"\n--- Processing file: {file_path} for date {date.strftime('%Y-%m-%d')} ---") # Added debug print

        df_raw = read_csv_auto(file_path) # Read raw df
        df_raw.columns = df_raw.columns.str.strip()
        print(f"--- {file_name} --- Initial rows: {len(df_raw)}")

        # --- 6.1 共同數據清洗與過濾 ---
        for col in df_raw.columns:
            if df_raw[col].dtype == object:
                df_raw[col] = df_raw[col].astype(str).str.strip()

        # Filter by Commodity Code
        df_filtered = df_raw[df_raw['商品代號'] == 'TXO'].copy() # Use .copy() to avoid SettingWithCopyWarning
        print(f"After '商品代號' == 'TXO': {len(df_filtered)} rows")

        # Further filtering common to both Call and Put
        df_filtered = df_filtered[df_filtered['到期月份(週別)'] == contract]
        print(f"After '到期月份(週別)' == contract ({contract}): {len(df_filtered)} rows")

        df_filtered['成交數量'] = pd.to_numeric(df_filtered['成交數量(B or S)'], errors='coerce')
        df_filtered = df_filtered[df_filtered['成交數量'] >= 30] # Volume filter
        print(f"After '成交數量' >= 30: {len(df_filtered)} rows")

        df_filtered['履約價'] = pd.to_numeric(df_filtered['履約價格'], errors='coerce')
        df_filtered['成交價格'] = pd.to_numeric(df_filtered['成交價格'], errors='coerce')
        df_filtered = df_filtered.dropna(subset=['履約價', '成交價格']) # Drop rows with missing key pricing data
        print(f"After dropna on '履約價', '成交價格': {len(df_filtered)} rows")

        if df_filtered.empty:
            print(f"No data remaining for {file_name} after filtering. Skipping IV and PCR calculation.")
            continue

        # --- 6.2 分離 Call 和 Put 選項數據 ---
        call_df = df_filtered[df_filtered['買賣權別'] == 'C'].copy()
        put_df = df_filtered[df_filtered['買賣權別'] == 'P'].copy()

        # --- 6.3 計算 Call 選項的隱含波動率 (IV) ---
        call_iv_mean, call_iv_std, call_iv_count = np.nan, np.nan, 0
        if not call_df.empty:
            call_prices = call_df['成交價格'].to_numpy()
            strike_prices = call_df['履約價'].to_numpy()
            ivs = np.empty_like(call_prices, dtype=np.float64)

            for j in range(len(call_prices)):
                ivs[j] = implied_vol_bisection_generic(
                    C=call_prices[j], S=S0, K=strike_prices[j], T=T, r=r, option_type=ord('C')
                )
            call_df['IV'] = ivs
            call_iv_mean = call_df['IV'].mean()
            call_iv_std = call_df['IV'].std()
            call_iv_count = call_df['IV'].count()

        # --- 6.4 計算 Put 選項的隱含波動率 (IV) ---
        put_iv_mean, put_iv_std, put_iv_count = np.nan, np.nan, 0
        if not put_df.empty:
            put_prices = put_df['成交價格'].to_numpy()
            strike_prices_put = put_df['履約價'].to_numpy()
            ivs_put = np.empty_like(put_prices, dtype=np.float64)

            for j in range(len(put_prices)):
                ivs_put[j] = implied_vol_bisection_generic(
                    C=put_prices[j], S=S0, K=strike_prices_put[j], T=T, r=r, option_type=ord('P')
                )
            put_df['IV'] = ivs_put
            put_iv_mean = put_df['IV'].mean()
            put_iv_std = put_df['IV'].std()
            put_iv_count = put_df['IV'].count()

        # --- 6.5 計算每日總量與 Put/Call Ratio (PCR) ---
        daily_call_volume = call_df['成交數量'].sum() if not call_df.empty else 0
        daily_put_volume = put_df['成交數量'].sum() if not put_df.empty else 0
        pcr = daily_put_volume / daily_call_volume if daily_call_volume > 0 else np.nan

        # --- 6.6 儲存每日結果 ---
        all_daily_results.append({
            'Date': date,
            'Call_IV_Count': call_iv_count,
            'Call_IV_Mean': call_iv_mean,
            'Call_IV_Std': call_iv_std,
            'Put_IV_Count': put_iv_count,
            'Put_IV_Mean': put_iv_mean,
            'Put_IV_Std': put_iv_std,
            'Total_Call_Volume': daily_call_volume,
            'Total_Put_Volume': daily_put_volume,
            'PCR': pcr
        })

        print(f"Processed: {file_name} - Call IV entries: {call_iv_count}, Put IV entries: {put_iv_count}, PCR: {pcr:.2f}")

    except Exception as e:
        print(f"Error processing {file_name} at path {file_path}: {e}")

# ============================================
# 7. 合併所有每日統計
# ============================================
daily_stats_df = pd.DataFrame(all_daily_results)

# Reorder columns for better readability
new_column_order = [
    'Date',
    'Total_Call_Volume', 'Call_IV_Count', 'Call_IV_Mean', 'Call_IV_Std',
    'Total_Put_Volume', 'Put_IV_Count', 'Put_IV_Mean', 'Put_IV_Std',
    'PCR'
]
daily_stats_df = daily_stats_df[new_column_order]

# ============================================
# 8. 顯示結果
# ============================================
print("\n=== 每日隱含波動率統計與 PCR ===")
print(daily_stats_df.head())

# 可存檔
output_path = os.path.join(index_file_path_prefix, "IV_PCR_daily_stats.csv")
daily_stats_df.to_csv(output_path, index=False)

print(f"\nSaved to: {output_path}")