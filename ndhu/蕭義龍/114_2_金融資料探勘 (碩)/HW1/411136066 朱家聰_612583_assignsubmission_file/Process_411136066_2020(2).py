import pandas as pd
import os
from google.colab import drive

# 1. 掛載 Google Drive
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# 2. 設定精確路徑
# 根據你的圖片，檔案都在「金融資料探勘」資料夾下
base_path = '/content/drive/My Drive/金融資料探勘'
raw_data_path = os.path.join(base_path, '原始資料_增加File欄位.csv')
expiry_data_path = os.path.join(base_path, '資料來源3.csv')

# 3. 讀取並修正標題列的函數
def load_financial_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到檔案：{path}")
    
    # 讀取時跳過第一行拆分的標題，並手動指定正確的編碼
    try:
        # 如果你的 CSV 預覽顯示標題被拆分，我們嘗試合併前兩行或直接清理
        df = pd.read_csv(path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(path, encoding='cp950')
    
    # 清理欄位名稱：移除換行符號與空白
    df.columns = [c.replace('\n', '').replace('\r', '').strip() for c in df.columns]
    
    # 如果讀進來的欄位名稱還是不對（例如變成「最後」），我們手動重新命名
    rename_dict = {
        '最後結算日': '最後結算日',
        '契約月份': '契約月份'
    }
    # 模糊比對：只要欄位名稱包含「結算」或「月份」就更正
    for col in df.columns:
        if '結算' in col: df.rename(columns={col: '最後結算日'}, inplace=True)
        if '月份' in col: df.rename(columns={col: '契約月份'}, inplace=True)
        if 'Date' in col: df.rename(columns={col: 'Date'}, inplace=True)
            
    return df

# 執行讀取
print("正在讀取雲端硬碟檔案...")
raw_df = load_financial_data(raw_data_path)
expiry_df = load_financial_data(expiry_data_path)

# 4. 資料轉換
raw_df['Date'] = pd.to_datetime(raw_df['Date'])
expiry_df['最後結算日'] = pd.to_datetime(expiry_df['最後結算日'])

# 5. 過濾「月選擇權」（不含 W）
monthly_expiry = expiry_df[~expiry_df['契約月份'].astype(str).str.contains('W')].copy()
monthly_expiry = monthly_expiry.sort_values('最後結算日')

# 6. 比對邏輯：距離結算日 > 1 天
def find_nearest_contract(trade_date):
    # 找出所有結算日比（交易日+1天）還要晚的合約
    mask = monthly_expiry['最後結算日'] > (trade_date + pd.Timedelta(days=1))
    valid = monthly_expiry[mask]
    
    if not valid.empty:
        target = valid.iloc[0]
        return pd.Series({
            'Contract': target['契約月份'],
            'ContractExpiryDate': target['最後結算日'].strftime('%Y/%m/%d'),
            'Maturity': (target['最後結算日'] - trade_date).days
        })
    return pd.Series([None, None, None], index=['Contract', 'ContractExpiryDate', 'Maturity'])

# 7. 執行並儲存
print("計算中...")
results = raw_df['Date'].apply(find_nearest_contract)
raw_df = pd.concat([raw_df, results], axis=1)

output_file = os.path.join(base_path, '原始資料_處理完成_v2.csv')
raw_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"成功！檔案已存至：{output_file}")
raw_df[['Date', 'Contract', 'ContractExpiryDate', 'Maturity']].head()