import pandas as pd
import os
from google.colab import drive

# 1. 掛載雲端硬碟
drive.mount('/content/drive')

# 2. 設定路徑
folder_path = '/content/drive/MyDrive/金融作業'
input_path = os.path.join(folder_path, 'Path_教學_最終完成版.csv')
output_path = os.path.join(folder_path, 'Path_教學_最終完成版_含利率.csv')

def process_rf_final_correct():
    if not os.path.exists(input_path):
        print(f"❌ 找不到檔案：{input_path}")
        return

    # 3. 讀取主資料
    df_main = pd.read_csv(input_path, encoding='utf-8-sig')
    df_main['Date_dt'] = pd.to_datetime(df_main['Date'])
    
    # 4. 【精確版】2022 年台銀一年期定儲機動利率變動表
    # 嚴格遵循台銀公告之「生效日」
    rates_data = [
        ['2022/01/01', 0.840],  # 年初利率
        ['2022/03/21', 1.090],  # 3/21 第一次升息生效
        ['2022/06/20', 1.215],  # 6/20 第二次升息生效
        ['2022/09/26', 1.340],  # 9/26 第三次升息生效 (修正：9/22-9/25仍為1.215)
        ['2022/12/19', 1.465]   # 12/19 第四次升息生效
    ]
    
    df_rf = pd.DataFrame(rates_data, columns=['EffectiveDate', 'Rf_percent'])
    df_rf['EffectiveDate'] = pd.to_datetime(df_rf['EffectiveDate'])
    
    # 5. 排序
    df_main = df_main.sort_values('Date_dt')
    df_rf = df_rf.sort_values('EffectiveDate')
    
    # 6. 使用 merge_asof 進行回溯比對 (backward)
    df_final = pd.merge_asof(df_main, df_rf, 
                             left_on='Date_dt', 
                             right_on='EffectiveDate', 
                             direction='backward')

    # 7. 數值換算 (1.215 -> 0.01215)
    df_final['Rf'] = df_final['Rf_percent'] / 100
    
    # 8. 清理並還原格式
    df_final = df_final.drop(columns=['Date_dt', 'EffectiveDate', 'Rf_percent'])
    
    # 9. 儲存結果
    df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ 利率資料已完全修正並儲存。")
    print("💡 驗證結果：")
    
    # 驗證您提到的爭議日期
    check_dates = ['2022/09/21', '2022/09/22', '2022/09/23', '2022/09/26']
    verification = df_final[df_final['Date'].isin(check_dates)]
    print(verification)

# 執行
try:
    process_rf_final_correct()
except Exception as e:
    print(f"❌ 執行失敗: {e}")
