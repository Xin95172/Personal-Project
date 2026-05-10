"""
台股投資宇宙定義。

提供預設的大型股清單，也可自行傳入。
"""

# ── 預設：台股大型股約 50 檔（TWSE .TW） ──
TW_LARGE_CAP = [
    # 半導體 & 電子
    "2330.TW",  # 台積電
    "2454.TW",  # 聯發科
    "2303.TW",  # 聯電
    "3711.TW",  # 日月光投控
    "2379.TW",  # 瑞昱
    "3034.TW",  # 聯詠
    "3037.TW",  # 欣興
    "2327.TW",  # 國巨
    "8069.TW",  # 元太
    "6669.TW",  # 緯穎
    # 電腦 & 周邊
    "2317.TW",  # 鴻海
    "2382.TW",  # 廣達
    "2357.TW",  # 華碩
    "2345.TW",  # 智邦
    "3231.TW",  # 緯創
    "4938.TW",  # 和碩
    "2301.TW",  # 光寶科
    "2308.TW",  # 台達電
    "2395.TW",  # 研華
    # 金融
    "2881.TW",  # 富邦金
    "2882.TW",  # 國泰金
    "2891.TW",  # 中信金
    "2886.TW",  # 兆豐金
    "2884.TW",  # 玉山金
    "2880.TW",  # 華南金
    "2892.TW",  # 第一金
    "5880.TW",  # 合庫金
    # 傳產 & 塑化
    "1301.TW",  # 台塑
    "1303.TW",  # 南亞
    "1326.TW",  # 台化
    "6505.TW",  # 台塑化
    "2002.TW",  # 中鋼
    "1101.TW",  # 台泥
    # 航運
    "2603.TW",  # 長榮
    "2615.TW",  # 萬海
    "2609.TW",  # 陽明
    # 食品 & 零售
    "2912.TW",  # 統一超
    "1216.TW",  # 統一
    # 通訊
    "2412.TW",  # 中華電
    "4904.TW",  # 遠傳
    # 汽車 & 機械
    "2207.TW",  # 和泰車
    "2049.TW",  # 上銀
    # 其他
    "5871.TW",  # 中租-KY
    "2633.TW",  # 台灣高鐵
    "9910.TW",  # 豐泰
    "3443.TW",  # 創意
    "2474.TW",  # 可成
]


def get_universe(
    tickers: list[str] | None = None,
    industry: str | None = None,
    twse_csv_path: str = "twse.csv"
) -> list[str]:
    """
    回傳投資宇宙股票代碼清單。
    
    參數:
    - tickers: 自訂清單，若提供則直接回傳。
    - industry: 指定產業名稱（如 '半導體'、'金融保險'）或 '全部' (所有股票)。若為 None 且沒提供 tickers，預設回傳 TW_LARGE_CAP。
    - twse_csv_path: twse.csv 的檔案路徑。
    """
    if tickers is not None:
        return tickers
        
    if industry is not None:
        import pandas as pd
        try:
            df = pd.read_csv(twse_csv_path)
            # 過濾出股票且有 yfinance_symbol
            df = df.dropna(subset=["yfinance_symbol"])
            df = df[df["security_type_guess"] == "股票"]
            
            if industry != "全部":
                df = df[df["industry"] == industry]
                
            return df["yfinance_symbol"].tolist()
        except Exception as e:
            print(f"讀取 {twse_csv_path} 或過濾產業失敗: {e}")
            return list(TW_LARGE_CAP)

    return list(TW_LARGE_CAP)
