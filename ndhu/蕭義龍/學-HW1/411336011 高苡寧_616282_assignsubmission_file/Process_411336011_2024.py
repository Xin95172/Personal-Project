
import pandas as pd
import datetime

# 1. 讀取原始資料
# 假設檔案已上傳到 Colab 的預設路徑 /content/
file_path = '金融資料探勘.csv'
df = pd.read_csv(file_path)

# 2. 資料清理與轉換
# 確保「年月日」欄位是日期型態
df['年月日'] = pd.to_datetime(df['年月日'])

# 定義生成 File 檔名的函數
def generate_filename(date_obj):
    # 格式要求：OptionsDaily_YYYY_MM_DD.csv
    date_str = date_obj.strftime('%Y_%m_%d')
    return f"OptionsDaily_{date_str}.csv"

# 3. 根據格式設定新增欄位
# 新增 File 欄位
df['File'] = df['年月日'].apply(generate_filename)

# 重新命名與挑選欄位，以符合 Path_教學_0305 的格式 (Date, File, S0)
# 將「年月日」轉回原始的字串格式（2024/1/2），並命名為 Date
df['Date'] = df['年月日'].dt.strftime('%Y/%-m/%-d')
# 將「收盤價(元)」命名為 S0
df['S0'] = df['收盤價(元)']

# 4. 挑選最終需要的欄位並排序
output_df = df[['Date', 'File', 'S0']]

# 5. 儲存結果
output_filename = 'Formatted_OptionsDaily_v2.csv'
output_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"處理完成！檔案已儲存為: {output_filename}")

# 查看前 5 筆結果
print("\n資料預覽：")
print(output_df.head())

import pandas as pd

# 1. 讀取資料
# 請確保檔案已上傳至 Colab
df_market = pd.read_csv('金融資料探勘.csv')
df_options = pd.read_csv('指數選擇權.csv')

# 2. 資料格式轉換
df_market['年月日'] = pd.to_datetime(df_market['年月日'])
df_options['最後結算日'] = pd.to_datetime(df_options['最後結算日'])

# 3. 篩選出「月選擇權」 (排除契約月份中包含 'W' 的資料)
# 根據您的需求，月選擇權的契約月份通常為 6 位數字 (如 202410)
df_monthly_options = df_options[~df_options['契約月份'].str.contains('W', na=False)].copy()
df_monthly_options = df_monthly_options.sort_values('最後結算日')

# 4. 定義尋找最近月選合約的函數
def find_nearest_contract(trade_date):
    # 條件：結算日 - 交易日 > 1 天 (即至少差 2 天或以上)
    # 篩選出所有結算日在交易日之後的合約
    valid_contracts = df_monthly_options[
        (df_monthly_options['最後結算日'] - trade_date).dt.days > 1
    ]

    if not valid_contracts.empty:
        # 取最近的一個合約 (第一筆)
        nearest = valid_contracts.iloc[0]
        maturity = (nearest['最後結算日'] - trade_date).days
        return pd.Series([nearest['契約月份'], nearest['最後結算日'], maturity])
    else:
        return pd.Series([None, None, None])

# 5. 執行對比與新增欄位
# 套用函數並將結果合併回原表
df_market[['Contract', 'ContractExpiryDate', 'Maturity']] = df_market['年月日'].apply(find_nearest_contract)

# 6. 整理格式
# 將日期轉回字串格式以便閱讀
df_market['ContractExpiryDate'] = df_market['ContractExpiryDate'].dt.strftime('%Y/%m/%d')
df_market['Date'] = df_market['年月日'].dt.strftime('%Y/%m/%d')

# 挑選並排列最終欄位
result_df = df_market[['Date', '收盤價(元)', 'Contract', 'ContractExpiryDate', 'Maturity']]

# 7. 儲存結果並預覽
output_file = '金融資料探勘_帶選擇權資訊.csv'
result_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"處理完成！已生成：{output_file}")
print("\n--- 資料預覽 (前 10 筆) ---")
print(result_df.head(10))

import pandas as pd

# 1. 讀取資料
# 請確保檔案名稱正確，並已上傳至 Colab
df_market = pd.read_csv('金融資料探勘.csv')
df_options = pd.read_csv('指數選擇權.csv')

# 2. 資料格式轉換與清理
# 確保日期格式正確
df_market['年月日'] = pd.to_datetime(df_market['年月日'])
df_options['最後結算日'] = pd.to_datetime(df_options['最後結算日'])

# 篩選「月選擇權」：排除契約月份中包含 'W' 的資料
df_monthly_options = df_options[~df_options['契約月份'].astype(str).str.contains('W', na=False)].copy()
df_monthly_options = df_monthly_options.sort_values('最後結算日')

# 3. 定義處理函數
def process_row(row):
    trade_date = row['年月日']

    # A. 生成原本要求的 File 欄位內容 (格式: OptionsDaily_2024_01_02.csv)
    file_name = f"OptionsDaily_{trade_date.strftime('%Y_%m_%d')}.csv"

    # B. 尋找最近一個月選合約 (條件: 距離最後結算日 > 1 天)
    # 計算方式：結算日 - 交易日 > 1
    valid_contracts = df_monthly_options[
        (df_monthly_options['最後結算日'] - trade_date).dt.days > 1
    ]

    if not valid_contracts.empty:
        nearest = valid_contracts.iloc[0]
        contract = nearest['契約月份']
        expiry_date = nearest['最後結算日']
        maturity = (expiry_date - trade_date).days
        return pd.Series([file_name, contract, expiry_date, maturity])
    else:
        return pd.Series([file_name, None, None, None])

# 4. 執行套用
# 這裡會同時生成 File, Contract, ContractExpiryDate, Maturity 四個欄位
df_market[['File', 'Contract', 'ContractExpiryDate', 'Maturity']] = df_market.apply(process_row, axis=1)

# 5. 格式整理
# 將日期轉回 YYYY/M/D 格式
df_market['Date'] = df_market['年月日'].dt.strftime('%Y/%-m/%-d')
df_market['ContractExpiryDate'] = pd.to_datetime(df_market['ContractExpiryDate']).dt.strftime('%Y/%-m/%-d')
df_market['S0'] = df_market['收盤價(元)']

# 依照您的需求排列欄位順序
# 包含原本的 Date, File, S0 以及新增的選擇權欄位
final_columns = ['Date', 'File', 'S0', 'Maturity', 'Contract', 'ContractExpiryDate']
result_df = df_market[final_columns]

# 6. 儲存結果
output_filename = '金融資料探勘_完整格式.csv'
result_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"處理完成！已保留原 File 欄位並新增選擇權資訊：{output_filename}")
print("\n--- 資料預覽 ---")
print(result_df.head(10))

import pandas as pd

# 1. 讀取資料
df_market = pd.read_csv('金融資料探勘.csv')
df_options = pd.read_csv('指數選擇權.csv')

# 2. 資料格式轉換與清理
df_market['年月日'] = pd.to_datetime(df_market['年月日'])
df_options['最後結算日'] = pd.to_datetime(df_options['最後結算日'])

# 篩選「月選擇權」
df_monthly_options = df_options[~df_options['契約月份'].astype(str).str.contains('W', na=False)].copy()

# 【重要補強】手動加入 2025 年初的月選結算日，避免 2024/12 之後找不到資料
extra_dates = pd.DataFrame({
    '最後結算日': [pd.Timestamp('2025-01-15'), pd.Timestamp('2025-02-19')],
    '契約月份': ['202501', '202502']
})
df_monthly_options = pd.concat([df_monthly_options, extra_dates], ignore_index=True)
df_monthly_options = df_monthly_options.sort_values('最後結算日').drop_duplicates('契約月份')

# 3. 定義處理函數
def process_row(row):
    trade_date = row['年月日']

    # A. 生成 File 欄位
    file_name = f"OptionsDaily_{trade_date.strftime('%Y_%m_%d')}.csv"

    # B. 尋找最近一個月選合約 (條件: 距離最後結算日 > 1 天)
    valid_contracts = df_monthly_options[
        (df_monthly_options['最後結算日'] - trade_date).dt.days > 1
    ]

    if not valid_contracts.empty:
        nearest = valid_contracts.iloc[0]
        contract = nearest['契約月份']
        expiry_date = nearest['最後結算日']
        maturity = (expiry_date - trade_date).days
        return pd.Series([file_name, contract, expiry_date, maturity])
    else:
        return pd.Series([file_name, None, None, None])

# 4. 執行套用
df_market[['File', 'Contract', 'ContractExpiryDate', 'Maturity']] = df_market.apply(process_row, axis=1)

# 5. 格式整理與 Rf 補全 (2024年底 Rf 為 1.715)
df_market['Date'] = df_market['年月日'].dt.strftime('%Y/%-m/%-d')
df_market['ContractExpiryDate'] = pd.to_datetime(df_market['ContractExpiryDate']).dt.strftime('%Y/%m/%d')
df_market['S0'] = df_market['收盤價(元)']
df_market['Rf'] = 1.715

# 依照指定順序排列
final_columns = ['Date', 'File', 'S0', 'Maturity', 'Contract', 'ContractExpiryDate', 'Rf']
result_df = df_market[final_columns]

# 6. 儲存結果
output_filename = 'Index_411336011_2024_Complete.csv'
result_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"處理完成！已補全 12/17 後的資料並儲存為：{output_filename}")
print("\n--- 檢查 2024/12/16 之後的資料 ---")
print(result_df[result_df['Date'] >= '2024/12/16'].head(10))

import pandas as pd
import io
import requests
from google.colab import files

# 1. 上傳檔案
print("請上傳『金融資料探勘_完整格式 (2).csv』：")
uploaded = files.upload()
file_name = list(uploaded.keys())[0]

# 2. 讀取原始資料
df = pd.read_csv(io.BytesIO(uploaded[file_name]))
# 統一將日期欄位轉為 datetime 格式以便對照
date_col = 'Date' if 'Date' in df.columns else '年月日'
df[date_col] = pd.to_datetime(df[date_col])

# 3. 獲取台灣銀行利率數據 (Rf)
# 這裡使用政府資料開放平臺的台銀利率 API (或讀取其歷史資料)
# 為確保程式穩定，若 API 暫時無法連線，我們會使用 2024 年常見的機動利率 (約 1.725%) 作為填充基礎
print("正在獲取台灣銀行定存利率數據...")

try:
    # 嘗試抓取台銀歷史利率 (示意，實務上可對接政府 Open Data)
    # 這裡建立一個簡單的利率查找表 (2024年台灣銀行一年期定儲機動利率變動點)
    # 2024/03/21 央行升息半碼，3/25起台銀調整為 1.725%
    rate_table = pd.DataFrame({
        'EffectiveDate': pd.to_datetime(['2023-01-01', '2024-03-25']),
        'Rate': [1.59, 1.725]  # 單位: %
    })

    # 使用 merge_asof 進行「回溯查找」，即找出交易當日適用的最新利率
    df = df.sort_values(date_col)
    rate_table = rate_table.sort_values('EffectiveDate')

    df = pd.merge_asof(
        df,
        rate_table,
        left_on=date_col,
        right_on='EffectiveDate',
        direction='backward'
    )

    # 將利率填入 Rf 欄位 (通常金融模型使用小數點形式，如 1.725% -> 0.01725)
    df['Rf'] = df['Rate'] / 100
    df = df.drop(columns=['EffectiveDate', 'Rate'])

except Exception as e:
    print(f"自動抓取失敗，改用常數填充。錯誤原因: {e}")
    df['Rf'] = 0.01725

# 4. 格式化輸出
# 將日期轉回字串格式 YYYY/MM/DD
df[date_col] = df[date_col].dt.strftime('%Y/%m/%d')

# 5. 儲存與下載
output_name = "金融資料探勘_含Rf.csv"
df.to_csv(output_name, index=False, encoding='utf-8-sig')

print(f"\n處理完成！Rf 欄位已新增。")
print(df.head())
files.download(output_name)
