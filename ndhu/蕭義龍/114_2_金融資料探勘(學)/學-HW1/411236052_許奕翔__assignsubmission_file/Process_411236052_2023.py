import pandas as pd
import numpy as np
from pandas.tseries.offsets import WeekOfMonth

# 1. 讀取新的大盤資料 (加入關鍵的 sep 與 encoding 參數)
df = pd.read_csv('2023_Index.csv', sep='\t', encoding='utf-16')

# 2. 重新命名欄位
# 把檔案裡的 '年月日' 改為 'Date'，'收盤價(元)' 改為 'S0'
df = df.rename(columns={'年月日': 'Date', '收盤價(元)': 'S0'})

# 確保 Date 是時間格式
df['Date'] = pd.to_datetime(df['Date'])

# 3. 新增 File 欄位
df['File'] = 'OptionsDaily_' + df['Date'].dt.strftime('%Y/%m/%d') + '.csv'

# ================= 核心計算：找出結算日與合約月份 =================

# 產生涵蓋 2022 年底到 2024 年初「所有第三個星期三」的清單 (結算日清單)
expiries = pd.date_range(start='2022-12-01', end='2024-02-01', freq='WOM-3WED')

# 定義一個函數，用來判斷每一天對應的「結算日」與「合約月份」
def get_contract_info(current_date):
    for exp_date in expiries:
        # 如果當天日期 小於等於 結算日，代表還在交易這個結算日的合約
        if current_date <= exp_date:
            contract_month = exp_date.strftime('%Y%m') # 轉成 YYYYMM 格式
            return exp_date, contract_month
    return None, None

# 把函數應用到我們資料表的每一個日期上
df['ContractExpiryDate'], df['Contract'] = zip(*df['Date'].apply(get_contract_info))

# =================================================================

# 3. 計算 Maturity (距離結算日的天數)
# 也就是 ContractExpiryDate 減去 Date 的天數
df['Maturity'] = (df['ContractExpiryDate'] - df['Date']).dt.days

# 4. 新增 Rf (無風險利率)
df['Rf'] = np.where(df['Date'] >= pd.to_datetime('2023/03/27'), 0.0159, 0.01465)

# 5. 將欄位整理成跟教學檔案一模一樣的順序，並把日期格式轉回 YYYY-MM-DD
df['Date'] = df['Date'].dt.strftime('%Y/%m/%d')
df['ContractExpiryDate'] = df['ContractExpiryDate'].dt.strftime('%Y-%m-%d')

# 確保欄位順序：Date, File, S0, Maturity, Contract, ContractExpiryDate, Rf
df = df[['Date', 'File', 'S0', 'Maturity', 'Contract', 'ContractExpiryDate', 'Rf']]

# 6. 印出結果並存檔
print("----- 處理完成！資料預覽 -----")
print(df.head(10)) # 印出前10筆看看有無順利換倉

# 另存為新的 CSV 檔案
df.to_csv('Index_411236052_2023.csv', index=False, encoding='utf-8-sig')
print("\n----- 檔案已經成功儲存！檔名為: Index_411236052_2023.csv -----")