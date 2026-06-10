import pandas as pd
import os
from google.colab import drive

# 1. 掛載 Google Drive
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# 2. 設定資料夾路徑
base_path = '/content/drive/My Drive/金融資料探勘'
raw_data_path = os.path.join(base_path, '原始資料_增加File欄位.csv')
expiry_data_path = os.path.join(base_path, '資料來源3.csv')

# 3. 讀取並清理資料的函數
def load_and_fix_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到檔案：{path}")
    
    # 嘗試不同編碼
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(path, encoding='cp950')
    
    # 清理標題：移除可能導致 KeyError 的換行符與空白
    df.columns = [str(c).replace('\n', '').replace('\r', '').strip() for c in df.columns]
    
    # 針對「資料來源3」可能的雙行標題進行欄位校正
    mapping = {'最後結算日': '最後結算日', '契約月份': '契約月份', 'Date': 'Date'}
    for col in df.columns:
        if '最後' in col and '結算' in col: df.rename(columns={col: '最後結算日'}, inplace=True)
        if '契約' in col or '月份' in col: 
            if 'W' not in str(df[col].iloc[0]) or '月份' in col: # 避開資料列誤認
                df.rename(columns={col: '契約月份'}, inplace=True)
                
    return df

# 執行讀取
print("正在讀取雲端硬碟檔案...")
raw_df = load_and_fix_csv(raw_data_path)
expiry_df = load_and_fix_csv(expiry_data_path)

# 4. 日期格式轉換
raw_df['Date'] = pd.to_datetime(raw_df['Date'])
expiry_df['最後結算日'] = pd.to_datetime(expiry_df['最後結算日'])

# 5. 過濾「月選擇權」（契約月份不含 'W'）
monthly_expiry = expiry_df[~expiry_df['契約月份'].astype(str).str.contains('W')].copy()
monthly_expiry = monthly_expiry.sort_values('最後結算日')

# 6. 比對邏輯：距離最後結算日 >= 1
def find_contract_info(trade_date):
    # 條件：最後結算日 - 交易日 >= 1
    # 這代表最後結算日可以是交易日的隔天或更晚
    valid_contracts = monthly_expiry[monthly_expiry['最後結算日'] >= (trade_date + pd.Timedelta(days=1))]
    
    if not valid_contracts.empty:
        # 取得最近的一個月選合約
        target = valid_contracts.iloc[0]
        expiry_date = target['最後結算日']
        return pd.Series({
            'Contract': target['契約月份'],
            'ContractExpiryDate': expiry_date.strftime('%Y/%m/%d'),
            'Maturity': (expiry_date - trade_date).days
        })
    else:
        return pd.Series([None, None, None], index=['Contract', 'ContractExpiryDate', 'Maturity'])

# 7. 應用邏輯並新增欄位
print("正在計算合約與到期天數...")
new_info = raw_df['Date'].apply(find_contract_info)
raw_df = pd.concat([raw_df, new_info], axis=1)

# 8. 儲存結果回雲端
output_path = os.path.join(base_path, '原始資料_處理完成.csv')
raw_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"--- 處理完畢 ---")
print(f"結果已儲存至：{output_path}")
print(raw_df[['Date', 'Contract', 'ContractExpiryDate', 'Maturity']].head())