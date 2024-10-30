import numpy as np
import pandas as pd
from math import sqrt, exp, log
from scipy.stats import norm
import os

def main():
    ave, std, Date = [], [], []
    year = "2021"
    file = "加權指數.xlsx"
    path = "/Users/xinc./Documents/GitHub/desktop-tutorial/ndhu/midterm/"
    excel_path = os.path.join(path, file)

    DD = pd.read_excel(excel_path, sheet_name=year)

    for i in range(len(DD)):
        if pd.isna(DD['Path'][i]) or pd.isna(DD['File'][i]):
            print(f"跳過第 {i} 行，因為 Path 或 File 欄位為空。")
            continue

        file_path = os.path.join(str(DD['Path'][i]), str(DD['File'][i]))

        try:
            D = pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            print(f"讀取檔案 {file_path} 時發生錯誤：{e}")
            continue

        D.columns = D.columns.str.strip()
        D = D.dropna().drop('開盤集合競價', axis=1, errors='ignore')

        mapping = {
            '成交日期': 'Date',
            '商品代號': 'ID',
            '履約價格': 'E',
            '到期月份(週別)': 'Maturity_contract',
            '買賣權別': 'Call_Put',
            '成交時間': 'Time',
            '成交價格': 'Pr',
            '成交數量(B or S)': 'Num',
        }
        D = D.rename(mapping, axis=1)

        S, T, R = DD['S0'][i], DD['Maturity'][i] / 365, 0.01465
        Contract, Call_Put, Volume = 'TXO', 'C', 30

        IV = []
        m = 0

        for index, row in D.iterrows():
            print(f"Processing row {index}: {row}")  # 印出每筆資料

            if row['ID'].strip() == Contract and row['Call_Put'].strip() == Call_Put:
                K, Price = row["E"], row["Pr"]
                sol = solve(row['Call_Put'].strip(), S, K, R, T, Price)

                if sol < 0.03:
                    continue

                IV.append(sol)
                print(f"{m:3d} \t {Price:7.2f} \t {K:.2f} \t {row['Num']} \t {IV[m]*100:.4f}% \t row: {index}")
                m += 1

        if IV:
            ave_value = np.average(IV)
            std_value = np.std(IV, ddof=1)
        else:
            print(f"{DD['Date'][i]} 無符合條件的資料")
            ave_value = float('nan')
            std_value = float('nan')

        ave.append(ave_value)
        std.append(std_value)
        Date.append(DD['Date'][i])

        print(f"{Date[-1]} \t ave = {ave_value*100:.4f}% \t std.s = {std_value:.4f}")
        print('**************************************************************')

    with open(f"Finalresult_{year}_{Call_Put}.txt", "w") as fp:
        for i in range(len(Date)):
            print(f"{Date[i]} \t {ave[i]:.6f} \t {std[i]:.6f}", file=fp)

def BS(X, Sigma, S, K, R, T, Price):
    d1 = (log(S / K) + (R + Sigma**2 / 2) * T) / (Sigma * sqrt(T))
    d2 = d1 - Sigma * sqrt(T)

    if X == "C":
        return S * norm.cdf(d1) - K * exp(-R * T) * norm.cdf(d2) - Price
    if X == "P":
        return K * exp(-R * T) * norm.cdf(-d2) - S * norm.cdf(-d1) - Price

def solve(X, S, K, R, T, Price):
    a, b = 0.00001, 1.00001
    for _ in range(4096):
        c = (b + a) / 2
        if b - c < 1e-6:
            break
        if BS(X, b, S, K, R, T, Price) * BS(X, c, S, K, R, T, Price) < 0:
            a = c
        else:
            b = c
    return c

if __name__ == "__main__":
    main()