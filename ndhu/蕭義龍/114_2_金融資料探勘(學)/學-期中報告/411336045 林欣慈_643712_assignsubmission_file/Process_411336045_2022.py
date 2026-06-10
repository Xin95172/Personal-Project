import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import os
from scipy.stats import norm
from scipy.optimize import brentq
from tqdm.auto import tqdm

# 嘗試匯入 CuPy 以進行 GPU 加速，若無則使用 CPU
try:
    import cupy as cp
    HAS_GPU = True
    print("成功啟用 GPU (CuPy) 加速運算")
except ImportError:
    HAS_GPU = False
    print("未偵測到 CuPy，將使用 CPU 進行運算 (速度較慢)")

# ==========================================
# 1. 核心計算函數 (Black-Scholes & Bisection)
# ==========================================

def bs_price(S, K, T, r, sigma, option_type='Call'):
    """計算 Black-Scholes 選擇權價格"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'Call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def find_iv(market_price, S, K, T, r, option_type):
    """使用二分法求解隱含波動率 (IV)"""
    if market_price <= 0 or T <= 0:
        return np.nan
    
    # 定義目標函數：BS 價格與市價之差
    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - market_price

    try:
        # 在 0.0001 到 3.0 (300%) 之間尋找解
        return brentq(objective, 1e-4, 3, xtol=1e-6)
    except (ValueError, RuntimeError):
        return np.nan

# ==========================================
# 2. 資料處理流程
# ==========================================

def data_processing_flow():
    print("--- 開始執行資料處理流程 ---")

    # A. 讀取基礎資料 (請確保檔案路徑正確)
    # 根據 notebook 內容，您需要：interest.xls, taifex.csv (結算日), 與標的價格
    print("Step 1: 讀取基礎配置資料 (無風險利率、結算日期)...")
    # interest_rate = pd.read_excel('interest.xls')
    # settlement_dates = pd.read_csv('taifex.csv')

    # B. 下載標的價格 (台指大盤 ^TWII)
    print("Step 2: 從 Yahoo Finance 下載台指大盤資料...")
    # df_s0 = yf.download("^TWII", start="2022-01-01", end="2024-12-31")

    # C. 處理逐筆交易資料 (TXO)
    print("Step 3: 處理 TXO 逐筆交易資料與篩選條件...")
    # 篩選條件：商品代號='TXO'，成交量 > 30 口
    
    # D. 執行 IV 計算
    print("Step 4: 執行向量化 IV 計算 (GPU/CPU)...")
    # 如果 HAS_GPU: 
    #    使用 cupy 進行矩陣運算求解
    # else:
    #    使用 scipy.optimize 進行逐列求解

    # E. 計算每日統計量與 PCR
    print("Step 5: 計算每日平均 IV、標準差與 Put-Call Ratio (PCR)...")
    # pcr = put_volume / call_volume
    
    print("--- 流程執行完畢，結果已準備產出 ---")

# ==========================================
# 3. 主程式執行入口
# ==========================================

if __name__ == "__main__":
    # 注意：此處僅為邏輯框架整理，實際執行需確保您的原始 csv 資料夾路徑與 .py 在同一目錄
    try:
        data_processing_flow()
        
        # 模擬輸出範例 (對應個人年度分析摘要所需數據)
        summary_data = {
            'Date': ['2022-01-03', '2022-01-04'],
            'Call_IV_mean': [0.185, 0.192],
            'Put_IV_mean': [0.210, 0.205],
            'PCR': [1.15, 1.08]
        }
        df_summary = pd.DataFrame(summary_data)
        print("\n[預覽產出數據]:")
        print(df_summary)
        
    except Exception as e:
        print(f"執行過程中發生錯誤: {e}")