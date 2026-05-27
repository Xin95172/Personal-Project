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
