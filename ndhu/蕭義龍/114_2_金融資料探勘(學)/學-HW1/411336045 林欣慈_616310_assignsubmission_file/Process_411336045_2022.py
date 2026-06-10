# ==========================================
# 作業名稱：運用AI協助建構一年份選擇權分析索引檔
# 負責年度：2022
# 檔案用途：資料處理流程實作檔 (Process_學號_年度.py)
# ==========================================

import pandas as pd
import datetime
import os

def construct_option_index(input_file, student_id, year_val):
    print(f"開始處理 {year_val} 年度資料...")

    # 1. 讀入原始資料並確認年度範圍與資料筆數 [cite: 24]
    df = pd.read_csv(input_file)
    print(f"原始資料筆數: {len(df)}")
    
    # 2. 整理日期格式並統一欄位名稱 [cite: 25]
    df['Date'] = pd.to_datetime(df['Date'])
    df.rename(columns={'Close': 'SO'}, inplace=True) # 將收盤價改名為 SO [cite: 26, 71]

    # 3. 建立每日檔名 File [cite: 26]
    # 格式範例：OptionsDaily_2020_01_02.csv [cite: 64]
    df['File'] = df['Date'].dt.strftime('OptionsDaily_%Y_%m_%d.csv')

    # 4. 建立近月契約 Contract 與契約到期日 ContractExpiry Date [cite: 27]
    # 定義台指選擇權結算日邏輯：每月第三個星期三
    def get_third_wednesday(y, m):
        first_day = datetime.date(y, m, 1)
        # weekday: 0=Mon, 2=Wed
        first_wed = (2 - first_day.weekday() + 7) % 7
        third_wed = first_day + datetime.timedelta(days=first_wed + 14)
        return pd.to_datetime(third_wed)

    def get_contract_logic(row):
        tx_date = row['Date']
        # 當月結算日
        current_expiry = get_third_wednesday(tx_date.year, tx_date.month)
        
        # 判定邏輯：若交易日超過當月結算日，則歸屬下月契約 [cite: 32, 70]
        if tx_date > current_expiry:
            next_month = tx_date.month + 1
            y = tx_date.year
            if next_month > 12:
                next_month = 1
                y += 1
            expiry_date = get_third_wednesday(y, next_month)
            contract = f"{y}{next_month:02d}"
        else:
            expiry_date = current_expiry
            contract = f"{tx_date.year}{tx_date.month:02d}"
        return pd.Series([contract, expiry_date])

    df[['Contract', 'ContractExpiry Date']] = df.apply(get_contract_logic, axis=1)

    # 5. 計算 Maturity，並清楚註明距離到期天數採用日曆日 
    df['Maturity'] = (df['ContractExpiry Date'] - df['Date']).dt.days

    # 6. 補入 Rf，說明資料來源與合併方式 [cite: 29]
    # 註：此處 Rf 採用參考範例之 0.0109 [cite: 64]
    df['Rf'] = 0.0109

    # 7. 整理最終年度索引檔欄位順序 [cite: 30, 71]
    # 順序：Date, File, SO, Contract, ContractExpiry Date, Maturity, Rf
    final_cols = ['Date', 'File', 'SO', 'Contract', 'ContractExpiry Date', 'Maturity', 'Rf']
    df_final = df[final_cols].copy()

    # 統一日期輸出格式為 yyyy-mm-dd [cite: 69]
    df_final['Date'] = df_final['Date'].dt.strftime('%Y-%m-%d')
    df_final['ContractExpiry Date'] = df_final['ContractExpiry Date'].dt.strftime('%Y-%m-%d')

    # 8. 輸出最終年度索引檔 [cite: 30]
    output_name = f"Index_{student_id}_{year_val}.xlsx"
    df_final.to_excel(output_name, index=False)
    
    print(f"處理完成！產出檔案：{output_name}")
    print("\n--- 最終索引檔前10筆資料展示 --- [cite: 48]")
    print(df_final.head(10))

# 執行區 (請修改學號)
if __name__ == "__main__":
    MY_ID = "您的學號"  # <--- 請修改這裡
    YEAR = "2022"
    INPUT = "2022.csv"
    
    if os.path.exists(INPUT):
        construct_option_index(INPUT, MY_ID, YEAR)
    else:
        print(f"找不到檔案 {INPUT}，請確認檔案已上傳至正確路徑。")