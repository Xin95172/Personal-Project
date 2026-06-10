import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta, WE
import io
from google.colab import files

# 1. 上傳原始 Excel 檔案
print("請點擊下方按鈕上傳您的 2023.xlsx")
uploaded = files.upload()
if not uploaded:
    print("未選擇檔案，請重新執行。")
else:
    file_name = list(uploaded.keys())[0]

    def get_third_wednesday(year, month):
        """計算台灣期權每個月的第三個星期三"""
        first_day = datetime(year, month, 1)
        first_wed = first_day + relativedelta(weekday=WE(1))
        third_wed = first_wed + timedelta(weeks=2)
        return third_wed

    # 2. 讀取 Excel 資料
    # 讀取第一個分頁，並強制轉換第一欄為日期，第二欄為數值
    df_raw = pd.read_excel(io.BytesIO(uploaded[file_name]))
    
    df = pd.DataFrame()
    df['Date'] = pd.to_datetime(df_raw.iloc[:, 0]) 
    df['Close'] = df_raw.iloc[:, 1]               

    # 依照日期排序（由舊到新，確保邏輯判斷正確）
    df = df.sort_values('Date').reset_index(drop=True)

    results = []
    for index, row in df.iterrows():
        # 過濾掉日期為空值的列（如果有空白列的話）
        if pd.isnull(row['Date']):
            continue
            
        curr_date = row['Date']
        
        # A. 找出當月的結算日
        expiry_this_month = get_third_wednesday(curr_date.year, curr_date.month)
        
        # B. 判斷契約月份
        if curr_date.date() > expiry_this_month.date():
            target_month_date = curr_date + relativedelta(months=1)
            expiry_date = get_third_wednesday(target_month_date.year, target_month_date.month)
        else:
            expiry_date = expiry_this_month
            
        # C. 計算 Maturity
        maturity = (expiry_date.date() - curr_date.date()).days
        
        results.append({
            'Date': curr_date.strftime('%Y/%m/%d'),
            'File': file_name,
            'SO': row['Close'],
            'Contract': expiry_date.strftime('%Y%m'),
            'ContractExpiry Date': expiry_date.strftime('%Y/%m/%d'),
            'Maturity': maturity,
            'Rf': 0.012
        })

    # 3. 產出最終完整表格
    final_df = pd.DataFrame(results)

    # --- 驗證區：顯示總筆數 ---
    print("\n" + "="*30)
    print(f"檔案處理成功！")
    print(f"總筆數 (列數): {len(final_df)} 筆")
    print(f"日期範圍: {final_df['Date'].min()} 到 {final_df['Date'].max()}")
    print("="*30)

    # 4. 存檔並自動發送下載請求
    output_filename = 'Full_Year_2023_Options.csv'
    final_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print(f"\n正在準備下載完整檔案：{output_filename} ...")
    files.download(output_filename)
