import yfinance as yf
import pandas as pd
df = yf.download("^TWII", start="2020-01-01", end="2020-12-31")
df = df[["Close"]].copy()
df.columns = ["S0"]
df["File"] = df.index.strftime("OptionsDaily_%Y_%m_%d.csv")
def third_wednesday(year, month):
    day = pd.Timestamp(year, month, 1)
    while day.weekday() != 2:
        day += pd.Timedelta(days=1)
    return day + pd.Timedelta(days=14)

def get_expiry(date):
    expiry = third_wednesday(date.year, date.month)
    # 如果今天已經到了到期日，就換下個月
    if date >= expiry:
        # 算下個月
        if date.month == 12:
            expiry = third_wednesday(date.year + 1, 1)
        else:
            expiry = third_wednesday(date.year, date.month + 1)
    return expiry

df["ContractExpiryDate"] = df.index.map(get_expiry)
df["Contract"] = df["ContractExpiryDate"].dt.strftime("%Y%m").astype(int)
df["Maturity"] = (df["ContractExpiryDate"] - df.index).dt.days

# 2020年3月19日央行降息1碼，臺灣銀行於3月25日起全面調降存款利率
# 降息前，一年期定存固定利率為 1.04%
df["Rf"] = 0.0104  
# 降息後，一年期定存固定利率調降至 0.79% 左右（機動為 0.84%）
df.loc[df.index >= "2020-03-25", "Rf"] = 0.0079  

df = df.reset_index()
df = df[["Date", "File", "S0", "Maturity", "Contract", "ContractExpiryDate", "Rf"]]
df["Date"] = df["Date"].dt.date
df["ContractExpiryDate"] = df["ContractExpiryDate"].dt.date
df.to_excel("Index_611336036_2020.xlsx", index=False)
print("完成！檔案已儲存")
