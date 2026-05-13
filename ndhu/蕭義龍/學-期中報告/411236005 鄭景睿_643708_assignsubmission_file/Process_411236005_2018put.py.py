import pandas as pd
import numpy as np
from scipy.stats import norm
import os
import zipfile
import io
import re

# 1. 掛載 Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# --- Black-Scholes 與 IV 運算 (支援 Call & Put) ---
def bs_price(S, K, T, r, sigma, cp='P'):
    if sigma <= 0 or T <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if cp.upper() == 'C':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def find_iv(market_price, S, K, T, r, cp='P'):
    # 賣權內含價值：max(0, K*e^-rT - S)
    intrinsic = max(0, K * np.exp(-r * T) - S) if cp.upper() == 'P' else max(0, S - K * np.exp(-r * T))
    if market_price <= intrinsic or market_price <= 0: return np.nan
    
    low, high = 1e-5, 3.0
    for _ in range(40):
        mid = (low + high) / 2
        price = bs_price(S, K, T, r, mid, cp)
        if abs(price - market_price) < 1e-5: return mid
        if price < market_price: low = mid
        else: high = mid
    return mid

# --- 主程式 ---
def main():
    base_path = '/content/drive/MyDrive/金融資料探勘'
    zip_files = [f for f in os.listdir(base_path) if f.endswith('.zip')]
    
    index_path = os.path.join(base_path, 'Path_教學_0409.csv')
    df_index = pd.read_csv(index_path) if os.path.exists(index_path) else pd.DataFrame()

    all_data_list = []

    for zip_name in zip_files:
        zip_path = os.path.join(base_path, zip_name)
        print(f"📦 正在掃描壓縮檔: {zip_name}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                for csv_name in z.namelist():
                    if not csv_name.lower().endswith('.csv'): continue
                    
                    # 參數提取邏輯
                    idx_row = df_index[df_index['File'] == csv_name]
                    if not idx_row.empty:
                        S0, T_days, Rf = idx_row.iloc[0]['S0'], idx_row.iloc[0]['Maturity'], idx_row.iloc[0]['Rf']
                        contract = str(idx_row.iloc[0]['Contract'])
                        trade_date = str(idx_row.iloc[0]['Date'])
                    else:
                        S0, T_days, Rf = 11000, 20, 0.008 # 預設值
                        contract = re.search(r'\d{6}', csv_name).group() if re.search(r'\d{6}', csv_name) else "Unknown"
                        trade_date = csv_name
                    
                    with z.open(csv_name) as f:
                        df = pd.read_csv(io.BytesIO(f.read()), encoding='big5', dtype=str)
                    
                    df.columns = df.columns.str.replace(r'\s+', '', regex=True)
                    mapping = {
                        '商品代號': 'Symbol', '履約價': 'StrikePrice', '成交價格': 'Price',
                        '成交價': 'Price', '到期月份': 'Contract', '買賣權別': 'CP'
                    }
                    final_map = {c: v for c in df.columns for k, v in mapping.items() if k in c}
                    df = df.rename(columns=final_map)
                    
                    if 'StrikePrice' not in df.columns: continue
                    
                    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
                    df['StrikePrice'] = pd.to_numeric(df['StrikePrice'], errors='coerce')
                    
                    # ⭐ 關鍵修改：篩選 P (Put)
                    mask = (df['Symbol'].str.contains('TXO', na=False)) & \
                           (df['CP'].str.contains('P', na=False, case=False)) & \
                           (df['Contract'].str.contains(contract, na=False)) & \
                           (df['Price'] > 0)
                    
                    df_day = df[mask].copy()
                    if not df_day.empty:
                        print(f"  🔍 正在計算賣權 IV: {csv_name} ({len(df_day)} 筆)")
                        lookup = df_day[['StrikePrice', 'Price']].drop_duplicates().copy()
                        # ⭐ 關鍵修改：帶入 'P' 參數
                        lookup['IV'] = lookup.apply(lambda x: find_iv(x['Price'], S0, x['StrikePrice'], T_days/252, Rf, 'P'), axis=1)
                        
                        df_day = df_day.merge(lookup, on=['StrikePrice', 'Price'], how='left')
                        df_day['TradeDate'] = trade_date
                        all_data_list.append(df_day)

        except Exception as e:
            print(f"  ❌ 出錯: {e}")

    if all_data_list:
        final_df = pd.concat(all_data_list, ignore_index=True).dropna(subset=['IV'])
        summary = final_df.groupby('TradeDate')['IV'].describe()
        
        # 存檔 (檔名標註為 Put)
        final_df.to_csv(os.path.join(base_path, 'TXO_Put_Detailed.csv'), index=False, encoding='utf-8-sig')
        summary.to_csv(os.path.join(base_path, 'TXO_Put_Summary.csv'), encoding='utf-8-sig')
        
        print("\n" + "="*40)
        print("💾 賣權 (Put) 資料處理完成！")
        display(summary)
    else:
        print("\n❌ 掃描完成，找不到符合條件的賣權資料。")

main()