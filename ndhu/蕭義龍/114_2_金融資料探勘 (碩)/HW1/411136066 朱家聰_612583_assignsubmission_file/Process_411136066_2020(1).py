import pandas as pd
import os
from google.colab import drive

# 1. 掛載 Google Drive
drive.mount('/content/drive', force_remount=True)

# 2. 設定路徑 (請確保資料夾名稱正確)
folder_path = '/content/drive/MyDrive/金融資料探勘'
input_file = os.path.join(folder_path, '原始資料.csv')
output_file = os.path.join(folder_path, 'Path_Converted_New.csv')

def load_csv_with_encoding(path):
    # 嘗試常見的台灣中文編碼
    encodings = ['utf-8-sig', 'cp950', 'big5', 'utf-8']
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, Exception):
            continue
    raise Exception("無法識別檔案編碼，請確認檔案格式是否正確。")

try:
    # 3. 使用修正後的讀取函數
    df = load_csv_with_encoding(input_file)
    print(f"✅ 成功讀取檔案！偵測到的欄位有：{list(df.columns)}")
    
    # 4. 核心邏輯：日期格式化與 File 欄位生成
    # 處理日期欄位名稱 (有些檔案會有空白，先去除)
    df.columns = df.columns.str.strip()
    
    # 將「年月日」轉為日期格式
    df['dt_temp'] = pd.to_datetime(df['年月日'])
    
    # 生成：OptionsDaily_YYYY_MM_DD.csv
    df['File'] = df['dt_temp'].dt.strftime('OptionsDaily_%Y_%m_%d.csv')
    
    # 5. 整理成 Path_教學_0305.csv 的格式
    # 欄位對應：年月日 -> Date, 收盤價(元) -> S0
    result_df = pd.DataFrame({
        'Date': df['年月日'],
        'File': df['File'],
        'S0': df['收盤價(元)']
    })
    
    # 6. 輸出結果至雲端硬碟
    # 使用 utf-8-sig 存檔，確保 Excel 開啟不會有亂碼
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print("\n--- 處理完成！前 5 筆資料預覽 ---")
    print(result_df.head())
    print(f"\n📂 檔案已存於：{output_file}")

except Exception as e:
    print(f"❌ 發生錯誤：{e}")