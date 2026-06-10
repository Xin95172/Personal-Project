import pandas as pd
import numpy as np
import os
from google.colab import drive

# 1. 掛載雲端硬碟
drive.mount('/content/drive')

# 2. 設定檔案路徑
file_path = '/content/drive/MyDrive/金融作業/Index_411136003_2022.csv'

def check_data_quality():
    if not os.path.exists(file_path):
        print(f"❌ 錯誤：找不到檔案 {file_path}")
        return

    # 3. 讀取資料
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    print("🔍 --- 開始進行數據缺值與異常檢核 ---")
    print(f"總筆數: {len(df)} 筆\n")

    # --- (A) 缺失值檢測 (Missing Value Check) ---
    missing_values = df.isnull().sum()
    print("📌 [1. 缺失值統計]")
    if missing_values.sum() == 0:
        print("✅ 所有欄位皆無缺值。")
    else:
        print(missing_values[missing_values > 0])
    print("-" * 30)

    # --- (B) 異常值檢核 (Outlier & Logic Check) ---
    print("📌 [2. 數值邏輯與異常檢核]")
    
    # 檢核 1: 標的價格 S0 是否有負值或 0
    s0_error = df[df['S0'] <= 0]
    if not s0_error.empty:
        print(f"⚠️ 異常：發現 {len(s0_error)} 筆 S0 指數價格小於或等於 0。")
    else:
        print("✅ S0 指數價格邏輯正常（皆大於 0）。")

    # 檢核 2: 到期天數 Maturity 是否有負值
    maturity_error = df[df['Maturity'] < 0]
    if not maturity_error.empty:
        print(f"⚠️ 異常：發現 {len(maturity_error)} 筆 Maturity 為負數（日期邏輯錯誤）。")
    else:
        print("✅ Maturity 到期天數邏輯正常（皆大於或等於 0）。")

    # 檢核 3: 無風險利率 Rf 是否在合理範圍 (假設台灣利率應在 0%~5% 之間)
    rf_outlier = df[(df['Rf'] < 0) | (df['Rf'] > 0.05)]
    if not rf_outlier.empty:
        print(f"⚠️ 異常：發現 {len(rf_outlier)} 筆 Rf 利率超出合理範圍 (0%~5%)。")
    else:
        print("✅ Rf 無風險利率數值正常。")

    print("-" * 30)

    # --- (C) 數據描述統計 (快速觀察極值) ---
    print("📌 [3. 數據描述統計預覽]")
    stats = df[['S0', 'Maturity', 'Rf']].describe().loc[['min', 'max', 'mean']]
    print(stats)

# 執行檢核
try:
    check_data_quality()
except Exception as e:
    print(f"❌ 執行時發生錯誤: {e}")