import pandas as pd
import os
from google.colab import drive

# 1. 掛載雲端硬碟
drive.mount('/content/drive')

# 2. 設定雲端硬碟檔案路徑
folder_path = '/content/drive/MyDrive/金融作業'
source_3_path = os.path.join(folder_path, '資料來源3.csv')
raw_data_path = os.path.join(folder_path, '原始資料_增加File欄位.csv')
output_path = os.path.join(folder_path, '原始資料_最終結果_修正版.csv')

def process_finance_task():
    # 檢查必要檔案是否存在
    if not os.path.exists(source_3_path) or not os.path.exists(raw_data_path):
        print("❌ 錯誤：找不到必要檔案，請檢查『金融作業』資料夾內的檔名。")
        return

    # 3. 讀取資料
    # 資料來源3 (到期日資料)：通常為 Big5 編碼
    df_expiry = pd.read_csv(source_3_path, encoding='big5')
    # 原始資料 (交易日資料)：通常為 utf-8-sig
    df_raw = pd.read_csv(raw_data_path, encoding='utf-8-sig')

    # 4. 預處理到期日資料
    df_expiry['最後結算日'] = pd.to_datetime(df_expiry['最後結算日'])
    # 規則：排除「契約月份」中包含 "W" 的周選擇權，僅保留月選擇權
    df_monthly = df_expiry[~df_expiry['契約月份'].astype(str).str.contains('W')].copy()
    df_monthly = df_monthly.sort_values('最後結算日')

    # 5. 處理原始交易資料日期格式
    df_raw['Date_dt'] = pd.to_datetime(df_raw['Date'])

    # 6. 定義核心比對邏輯
    def get_contract_data(row):
        trade_date = row['Date_dt']
        
        # 條件：最後結算日 - 交易日 >= 1 天 (即結算日前一天或更早)
        # 如果當天就是結算日 (diff=0)，則不符合 >=1 條件，會自動尋找下一個月合約
        mask = (df_monthly['最後結算日'] - trade_date).dt.days >= 1
        valid_options = df_monthly[mask]
        
        if not valid_options.empty:
            nearest = valid_options.iloc[0]
            # 計算 Maturity (剩餘日曆天數)
            maturity = (nearest['最後結算日'] - trade_date).days
            return pd.Series([nearest['契約月份'], nearest['最後結算日'], maturity])
        
        return pd.Series([None, None, None])

    print("正在計算合約歸屬 (條件: Maturity >= 1)...")
    # 新增 Contract, ContractExpiryDate, Maturity 三個欄位
    df_raw[['Contract', 'ContractExpiryDate', 'Maturity']] = df_raw.apply(get_contract_data, axis=1)

    # 7. 格式化輸出日期並清理暫存欄位
    df_raw['ContractExpiryDate'] = pd.to_datetime(df_raw['ContractExpiryDate']).dt.strftime('%Y/%-m/%-d')
    df_result = df_raw.drop(columns=['Date_dt'])

    # 8. 儲存結果回雲端硬碟
    df_result.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ 處理成功！結果已儲存至：{output_path}")
    print("\n--- 轉換後資料預覽 (前5筆) ---")
    print(df_result.head())

# 執行程式
process_finance_task()
