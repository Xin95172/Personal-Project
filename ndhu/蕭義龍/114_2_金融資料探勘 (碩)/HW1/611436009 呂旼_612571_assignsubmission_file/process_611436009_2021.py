import yfinance as yf
import pandas as pd
df = yf.download("^TWII", start="2021-01-01", end="2021-12-31")
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

# 2021年央行全年沒有升降息的動作（維持自2020年3月降息後的水準）。
# 臺灣銀行一年期定存固定利率全年都在歷史低點的 0.79% 左右。
df["Rf"] = 0.0079  

df = df.reset_index()
df = df[["Date", "File", "S0", "Maturity", "Contract", "ContractExpiryDate", "Rf"]]
df["Date"] = df["Date"].dt.date
df["ContractExpiryDate"] = df["ContractExpiryDate"].dt.date
df.to_excel("Index_611436009_2021.xlsx", index=False)
print("完成！檔案已儲存")
