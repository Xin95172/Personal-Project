import os
import glob
import pandas as pd
import numpy as np
import math
import gc
from scipy.stats import norm

# 1. 設定 2023 年路徑 (包含雙層資料夾處理)
folder_name = r"C:\Users\chen2\Downloads\Option_2023\Option_2023"
file_pattern = os.path.join(folder_name, '*.zip')
file_list = glob.glob(file_pattern)

# 2. 設定 2023 年基本參數 (IV 計算用)
DEFAULT_S = 16500.0  # 2023 年大盤平均水位概估
R_FIXED = 0.01      # 無風險利率 1%
T_FIXED = 0.04      # 距離到期時間 (約 14 天)

# === 核心數學函數 (Black-Scholes 與 二分法找 IV) ===
def bs_price(S, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0: return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'Call':
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def find_iv(market_price, S, K, T, r, option_type):
    if market_price <= 0: return np.nan
    low, high = 0.0001, 3.0
    for _ in range(20):
        mid = (low + high) / 2
        price = bs_price(S, K, T, r, mid, option_type)
        if price < market_price:
            low = mid
        else:
            high = mid
    return (low + high) / 2

# === 主程式迴圈 ===
daily_summary = []

print(f"🚀 開始執行 2023 年數據提煉... 總共 {len(file_list)} 個檔案")

if len(file_list) == 0:
    print(f"❌ 錯誤：在 {folder_name} 找不到 ZIP 檔案！請檢查路徑是否有誤。")
else:
    for i, file_path in enumerate(file_list):
        file_name = os.path.basename(file_path)
        if i % 10 == 0:
            print(f"進度: {i+1}/{len(file_list)} 正在處理 {file_name}")
            
        try:
            # 讀取與初步清理
            df = pd.read_csv(file_path, encoding='Big5', low_memory=False, compression='zip')
            df.columns = df.columns.str.strip()
            df = df[~df['成交日期'].astype(str).str.contains('-')]
            df_txo = df[df['商品代號'].astype(str).str.strip() == 'TXO'].copy()
            
            if len(df_txo) == 0: continue
            
            # 欄位定位與轉型
            price_col = [c for c in df_txo.columns if '成交價格' in c][0]
            vol_col = [c for c in df_txo.columns if '成交數量' in c][0]
            strike_col = [c for c in df_txo.columns if '履約價格' in c][0]
            
            df_txo[price_col] = pd.to_numeric(df_txo[price_col], errors='coerce')
            df_txo[vol_col] = pd.to_numeric(df_txo[vol_col], errors='coerce')
            df_txo[strike_col] = pd.to_numeric(df_txo[strike_col], errors='coerce')
            df_txo['買賣權別'] = df_txo['買賣權別'].astype(str).str.strip().map({'買權':'Call','C':'Call','賣權':'Put','P':'Put'})
            
            # 1. PCR 計算[cite: 1]
            vols = df_txo.groupby('買賣權別')[vol_col].sum()
            c_vol, p_vol = vols.get('Call', 0), vols.get('Put', 0)
            pcr = p_vol / c_vol if c_vol > 0 else np.nan
            
            # 2. IV 計算 (加速版：先群組計算平均價格)[cite: 1]
            top_contracts = df_txo.groupby(['買賣權別', strike_col])[price_col].mean().reset_index()
            top_contracts.columns = ['Type', 'K', 'Price']
            
            iv_results = []
            for _, row in top_contracts.iterrows():
                iv = find_iv(row['Price'], DEFAULT_S, row['K'], T_FIXED, R_FIXED, row['Type'])
                iv_results.append({'Type': row['Type'], 'IV': iv})
            
            iv_df = pd.DataFrame(iv_results).dropna()
            
            # 存入每日總結
            daily_summary.append({
                'Date': df_txo['成交日期'].iloc[0],
                'PCR': pcr,
                'Call_IV_Mean': iv_df[iv_df['Type']=='Call']['IV'].mean(),
                'Call_IV_Std': iv_df[iv_df['Type']=='Call']['IV'].std(ddof=1), # 樣本標準差[cite: 1]
                'Put_IV_Mean': iv_df[iv_df['Type']=='Put']['IV'].mean(),
                'Put_IV_Std': iv_df[iv_df['Type']=='Put']['IV'].std(ddof=1)
            })
            
            # 釋放記憶體
            del df, df_txo, iv_df
            gc.collect()
            
        except Exception as e:
            print(f"跳過 {file_name} 錯誤: {e}")

    # --- 產出 2023 年報表 ---
    if daily_summary:
        final_df = pd.DataFrame(daily_summary)
        # 存成 2023 專屬檔案
        final_df.to_csv("2023_Final_Report_Data.csv", index=False)
        print("\n🎉 2023 年數據提煉完成！檔案已存為 '2023_Final_Report_Data.csv'")
        print(final_df.head())