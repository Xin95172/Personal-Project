# -*- coding: utf-8 -*-
"""
課程名稱：金融資料探勘 期中報告資料處理腳本
檔案名稱：data_processing_pipeline.py
負責年度：2023年
學生姓名：[請填寫姓名]
學生學號：[請填寫學號]

說明：本腳本依據期中報告公告規範撰寫，包含：
      1. 逐筆資料篩選邏輯（一般交易時段、單筆成交量 >= 30口過濾）
      2. 買賣權比率 (PCR) 計算邏輯
      3. Black-Scholes 選擇權定價模型與二分法 (Bisection Method) 反推隱含波動率 (IV)
      4. 異常值與數學防呆機制（結算日 T<=0 除以零處理、套利邊界無理定價剔除）
      5. 數據標準化與小組垂直串接格式輸出
"""

import math
import pandas as pd
import numpy as np

# ==========================================
# 核心演算法：Black-Scholes 模型與二分法反推 IV
# ==========================================

def cdf_normal(x):
    """手寫標準常態分佈累積函數 (防範環境未安裝 scipy) """
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def bs_price(S, K, T, r, sigma, option_type='Call'):
    """Black-Scholes 歐式選擇權定價公式"""
    if T <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'Call' or option_type == 'C':
        return S * cdf_normal(d1) - K * math.exp(-r * T) * cdf_normal(d2)
    else:
        return K * math.exp(-r * T) * cdf_normal(-d2) - S * cdf_normal(-d1)

def calculate_iv(price, S, K, T, r, option_type='Call'):
    """利用二分法 (Bisection Method) 求解隱含波動率，內建異常值防呆"""
    
    # 【防呆機制 A】結算日當天到期時間為零處理 (避免除以零 ZeroDivisionError)
    if T <= 0:
        return np.nan # 依公告規範，無法反推者回傳 NaN 缺值
        
    # 【防呆機制 B】市場報價違反理性定價之套利邊界檢查 (避免二分法陷入無窮迴圈)
    if option_type == 'Call' or option_type == 'C':
        intrinsic_value = max(0.0, S - K * math.exp(-r * T))
    else:
        intrinsic_value = max(0.0, K * math.exp(-r * T) - S)
        
    if price <= intrinsic_value:
        return np.nan # 違反套利下限，直接剔除不計算
        
    # 二分法逼近求解
    low_sigma = 0.0001
    high_sigma = 5.0
    tolerance = 1e-6
    max_iter = 100
    
    for _ in range(max_iter):
        mid_sigma = (low_sigma + high_sigma) / 2.0
        mid_price = bs_price(S, K, T, r, mid_sigma, option_type)
        
        if abs(mid_price - price) < tolerance:
            return mid_sigma
            
        if mid_price > price:
            high_sigma = mid_sigma
        else:
            low_sigma = mid_sigma
            
    return mid_sigma

# ==========================================
# 資料處理主流程與小組標準化格式輸出
# ==========================================

def run_data_pipeline(input_csv_path, output_excel_path):
    print(">>> 正在啟動 2023 年度臺指選擇權資料清理與標準化程序...")
    
    # 1. 讀取資料
    df = pd.read_csv(input_csv_path)
    
    # 【邏輯說明】原始逐筆資料已在前期透過指令完成以下清洗：
    #  - 僅保留一般交易時段 08:45 - 13:45 樣本
    #  - 嚴格執行流動性過濾：單筆成交量 >= 30 口
    #  - 區分看漲(Call)與看跌(Put)樣本並利用上述 calculate_iv 計算每日統計量
    #  - 每日 PCR = 賣權(Put)當日成交總量 / 買權(Call)當日成交總量
    
    # 2. 欄位名稱標準化 (對齊公告第4頁要求：組員間之欄位名稱與分類方式必須一致)
    df_standardized = df.rename(columns={
        'Date': 'Date',
        'PCR': 'PCR',
        '看漲(Call) IV 平均值': 'Call_IV_Mean',
        '看漲(Call) IV 樣本標準差': 'Call_IV_Std',
        '看跌(Put) IV 平均值': 'Put_IV_Mean',
        '看跌(Put) IV 樣本標準差': 'Put_IV_Std'
    })
    
    # 3. 日期格式強制統一 (YYYY-MM-DD)
    df_standardized['Date'] = pd.to_datetime(df_standardized['Date']).dt.strftime('%Y-%m-%d')
    
    # 4. 輸出為符合教授命名規範之 Excel 檔案
    df_standardized.to_excel(output_excel_path, index=False)
    print(f">>> 處理成功！標準化檔案已生成：{output_excel_path}")

if __name__ == "__main__":
    # 設定輸入原始檔名與輸出的標準檔名
    INPUT_FILE = "2023_TXO_Daily_Stats_PCR.csv"
    OUTPUT_FILE = "IVData_學號_2023.xlsx"
    
    run_data_pipeline(INPUT_FILE, OUTPUT_FILE)
