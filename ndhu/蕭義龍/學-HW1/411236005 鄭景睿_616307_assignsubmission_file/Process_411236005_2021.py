Python 3.14.4 (tags/v3.14.4:23116f9, Apr  7 2026, 14:10:54) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
import pandas as pd

# 1. 讀取 Excel 檔案 (請確保檔名與左側列表完全一致)
# 如果檔名有變動，請稍微修改下方字串
file_index = '加權指數.xlsx' 
file_expiry = '資料來源3.xlsx'

try:
    # 讀取指數：skiprows=1 為了跳過第一行 Y9999
    df_main = pd.read_excel(file_index, skiprows=1)
    # 讀取到期日
    df_expiry = pd.read_excel(file_expiry)
    print("✅ 檔案讀取成功！")
except Exception as e:
    print(f"❌ 讀取失敗，請檢查檔名是否為 {file_index} 與 {file_expiry}")
    print(f"錯誤訊息: {e}")

# 2. 處理到期日資料 (排除週選 W)
df_expiry['最後結算日'] = pd.to_datetime(df_expiry['最後結算日'])
... # 只要「契約月份」裡面有 'W' 就拿掉，只留月選
... df_monthly = df_expiry[~df_expiry['契約月份'].astype(str).str.contains('W')].copy()
... df_monthly = df_monthly.sort_values('最後結算日').reset_index(drop=True)
... 
... # 3. 處理原始資料日期
... df_main['Date'] = pd.to_datetime(df_main['年月日'])
... 
... # 4. 核心邏輯：交易日距離結算日 >= 1
... def find_contract(trade_date):
...     # 條件：結算日 - 交易日 >= 1
...     # 若當天是結算日(差0天)，會自動找下一個月
...     mask = (df_monthly['最後結算日'] - trade_date).dt.days >= 1
...     valid = df_monthly[mask]
...     
...     if not valid.empty:
...         target = valid.iloc[0]
...         expiry = target['最後結算日']
...         return pd.Series([target['契約月份'], expiry, (expiry - trade_date).days])
...     return pd.Series([None, None, None])
... 
... # 5. 合成欄位
... print("🚀 正在執行合成邏輯...")
... df_main[['Contract', 'ContractExpiryDate', 'Maturity']] = df_main['Date'].apply(find_contract)
... 
... # 格式化日期顯示為字串 (YYYY-MM-DD)
... df_main['ContractExpiryDate'] = pd.to_datetime(df_main['ContractExpiryDate']).dt.strftime('%Y-%m-%d')
... 
... # 6. 存檔
... output_name = '合成結果_最終版.csv'
... df_main.to_csv(output_name, index=False, encoding='utf-8-sig')
... 
... print(f"✨ 完成！結果已儲存為：{output_name}")
... # 預覽前 10 筆
... df_main.head(10)
SyntaxError: multiple statements found while compiling a single statement
