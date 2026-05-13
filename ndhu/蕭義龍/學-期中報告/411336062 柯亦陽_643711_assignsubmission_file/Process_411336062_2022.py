# ==========================================
# 2022年度 臺指選擇權(TXO) 資料處理與隱含波動率分析腳本
# ==========================================
import pandas as pd
import numpy as np
import os
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. 定義 Black-Scholes 與隱含波動率(IV)計算函數
# ---------------------------------------------------------
def bs_price(S, K, T, r, sigma, option_type):
    """計算 Black-Scholes 選擇權理論價格"""
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if option_type == 'C' else max(0.0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'C':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calculate_iv(row):
    """利用 brentq 數值方法反推隱含波動率"""
    try:
        market_price = float(row['Market_Price'])
        S = float(row['S0'])
        K = float(row['Strike'])
        T = float(row['Maturity']) / 365.0
        r = float(row['Rf'])
        option_type = str(row['Option_Type']).strip().upper()
        
        # 排除市價低於內含價值的不合理樣本
        intrinsic_value = max(0.0, S - K) if option_type == 'C' else max(0.0, K - S)
        if market_price <= intrinsic_value: 
            return np.nan
        
        obj_fun = lambda sigma: bs_price(S, K, T, r, sigma, option_type) - market_price
        return brentq(obj_fun, 1e-4, 5.0)
    except:
        return np.nan

# ---------------------------------------------------------
# 2. 資料清理與彙整流程 (針對 2022 年資料)
# ---------------------------------------------------------
def process_data(file_path):
    print("正在讀取並清理資料...")
    df = pd.read_csv(file_path)
    df['daily'] = pd.to_datetime(df['daily'])
    
    # 嚴格篩選 2022 年區間
    df = df[(df['daily'] >= '2022-01-01') & (df['daily'] <= '2022-12-31')]
    
    # 處理期交所資料中因週選與月選多筆檔案導致的日期重複問題 (取平均)
    df_clean = df.groupby('daily').mean().reset_index()
    df_clean = df_clean.sort_values('daily')
    
    return df_clean

if __name__ == '__main__':
    input_file = 'Final_Report_Data_2022.xlsx - Sheet1.csv'
    if os.path.exists(input_file):
        final_df = process_data(input_file)
        final_df.to_csv('Processed_2022_Data.csv', index=False)
        print("資料處理完畢，已輸出 Processed_2022_Data.csv")
    else:
        print(f"找不到檔案 {input_file}，請確認。")
