import pandas as pd
import os
from google.colab import drive

# 1. 掛載雲端硬碟 (如果您已經掛載過，這行會顯示已掛載)
drive.mount('/content/drive')

# 2. 設定路徑 (請確保資料夾名稱正確，MyDrive 之間通常沒有空格)
folder_path = '/content/drive/MyDrive/金融作業'
input_path = os.path.join(folder_path, '原始資料.csv')
output_path = os.path.join(folder_path, 'Converted_Path_Data.csv')

if os.path.exists(input_path):
    try:
        # 3. 讀取原始資料 (針對台灣 Excel 檔案，加入 encoding='big5')
        # 如果 big5 仍報錯，可以嘗試 'cp950'
        df = pd.read_csv(input_path, encoding='big5')
        
        # 4. 轉換日期並生成 File 欄位
        df['年月日'] = pd.to_datetime(df['年月日'])
        
        # 依照要求：File 為 OptionDaily_2022_01_03.csv 格式
        df['File'] = df['年月日'].dt.strftime('OptionDaily_%Y_%m_%d.csv')
        
        # 依照要求：年月日改為 Date (格式 YYYY/M/D)
        df['Date'] = df['年月日'].dt.strftime('%Y/%-m/%-d')
        
        # 依照要求：收盤價(元)改為 S0
        df = df.rename(columns={'收盤價(元)': 'S0'})
        
        # 5. 整理最終欄位順序
        output_df = df[['Date', 'File', 'S0']]
        
        # 6. 儲存結果 (加上 utf-8-sig 確保輸出的檔案在 Excel 開啟不亂碼)
        output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print("✅ 轉換成功！")
        print(f"檔案位置：{output_path}")
        print("\n資料預覽：")
        print(output_df.head())
        
    except Exception as e:
        print(f"❌ 讀取時發生錯誤: {e}")
        print("提示：請確認檔案編碼是否為 Big5，或嘗試將 encoding 改為 'cp950'。")
else:
    print(f"❌ 找不到檔案：{input_path}")
