import csv
from datetime import datetime, date, timedelta

s=[["Date","File","S0","Contract","ContractExpiryDate","Maturity","Rf"]]

ss=[]

# 開啟 CSV 檔案
with open('2018~2024臺指選擇權結算.csv', newline='') as csvfile:

    # 讀取 CSV 檔案內容
    rows = csv.reader(csvfile)

    # 以迴圈輸出每一列
    for row in rows:
        if "W" not in row[1] and row[1].isdigit():
            #print(row)
            ss.append(row)
    ss=ss[::-1]
    #print(ss)

# 開啟 CSV 檔案
with open('2018~2024收盤價.csv', newline='') as csvfile:

    # 讀取 CSV 檔案內容
    rows = csv.reader(csvfile)

    # 以迴圈輸出每一列
    for row in rows:
        y=['2020']#選擇年分
        for j in y:
            if j in row[0]:
                row.append(row[1])
                row[1]="OptionsDaily_"+row[0].split("/")[0]+"_"+row[0].split("/")[1]+"_"+row[0].split("/")[2]+".csv"
                d1 = date(int(row[0].split("/")[0]), int(row[0].split("/")[1]),int(row[0].split("/")[2]))
                for i in ss:
                    d2=date(int(i[0].split("/")[0]), int(i[0].split("/")[1]), int(i[0].split("/")[2]))
                    delta = d2-d1
                    if delta.days>0:
                        row.append(i[1])
                        row.append(i[0])
                        #print(d1,d2)
                        #print(f"相差天數: {delta.days} 天")
                        row.append(delta.days)
                        row.append(row[2])
                        del row[2]
                        break

                s.append(row)
sss=''
for i in y:
    sss=sss+'_'+i
with open('Index_411111220'+sss+'.csv', 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerows(s)
