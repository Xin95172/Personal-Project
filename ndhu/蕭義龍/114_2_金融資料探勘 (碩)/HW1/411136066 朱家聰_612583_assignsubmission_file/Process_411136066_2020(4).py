import pandas as pd
import os
from google.colab import drive

# 1. 掛載 Google Drive
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# 2. 設定路徑
base_path = '/content/drive/My Drive/金融資料探勘'
file_path = os.path.join(base_path, 'Path_教學_最終完成版.csv')

# 3. 讀取 CSV 檔案
try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
except:
    df = pd.read_csv(file_path, encoding='cp950')

df['Date'] = pd.to_datetime(df['Date'])

# 4. 取得台灣銀行歷史利率數據
# 說明：在自動化流程中，建議從政府公開資料 API 取得。
# 這裡建立一個簡單的利率查找表 (範例為 2020-2021 年常見機動利率區間)
# 實務上您可以替換為爬蟲抓取的歷史數據或是從 .csv 讀取利率表
def get_bot_rf(date):
    """
    根據日期回傳台灣銀行定期儲蓄存款(一年期, 機動利率)
    此處以 2020-2021 常見數據為例，若您有具體的利率 csv 檔案亦可合併
    """
    # 2020/03/23 之後因應降息，多數銀行定儲機動利率約為 0.84%
    if date >= pd.Timestamp('2020-03-23'):
        return 0.84 / 100
    else:
        # 2020/03/23 之前約為 1.09%
        return 1.09 / 100

# 5. 新增 Rf 欄位並填入數據
print("正在比對利率數據...")
df['Rf'] = df['Date'].apply(get_bot_rf)

# 6. 儲存結果
output_path = os.path.join(base_path, 'Path_教學_最終完成版_含Rf.csv')
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"處理完成！新檔案已儲存至：{output_path}")
print(df[['Date', 'Rf']].head())