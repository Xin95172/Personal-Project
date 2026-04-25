import os
import json
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from config.patterns import (
    JTITLE_PATTERNS,
    MAIN_PATTERNS,
    JTYPE_PATTERNS,
    JRESULT_PATTERNS,
    MANUAL_LABELING,
)
from utils import (
    classify_cases,
    extract_fact,
    remove_blank,
)

def _process_unified_case(args):
    jid, input_folder, SPECIAL_CASES = args
    import os
    import json
    import re
    from config.patterns import (
        JTITLE_PATTERNS, JTYPE_PATTERNS, MAIN_PATTERNS, JRESULT_PATTERNS, MANUAL_LABELING
    )
    from utils import (
        ip_law_check, j_type_check, extract_main_clause, j_result_check,
        extract_fact, map_manual_verdict
    )
    from utils.role_extractor import extract_role_features

    file_path = os.path.join(input_folder, f"{jid}.json")
    try:
        if os.path.getsize(file_path) == 0: return None
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except: return None

    jtitle = data.get("JTITLE", "")
    jcase = data.get("JCASE", "")
    jfull = data.get("JFULL", "")

    # 核心特徵提取 (僅保留判定勝負所需的欄位)
    ip_law = ip_law_check(JTITLE_PATTERNS, jtitle, jcase)
    j_type = j_type_check(JTYPE_PATTERNS, jcase, jfull, ip_law, jid)
    main_clause = extract_main_clause(MAIN_PATTERNS, jfull)
    role_f = extract_role_features(jfull, j_type)

    return {
        "JID": jid,
        "JYEAR": data.get("JYEAR", ""),
        "JCASE": jcase,
        "JDATE": data.get("JDATE", ""),
        "JTITLE": jtitle,
        "IP Law": ip_law,
        "JTYPE": j_type,
        "main_clause": main_clause,
        "JFULL": jfull,
        **role_f
    }

# 定位專案根目錄 (textual data analysis/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def run_preprocessing(input_folder=None, output_folder=None,
                      artifacts_folder=None, cache_folder=None, n_jobs=-1,
                      parquet_folder=None):
    # ─── 自動對齊絕對路徑 (解決路徑亂竄問題) ───
    if input_folder is None: input_folder = os.path.join(PROJECT_ROOT, "data/raw_json")
    if output_folder is None: output_folder = os.path.join(PROJECT_ROOT, "data/IP_Law_processed")
    if artifacts_folder is None: artifacts_folder = os.path.join(PROJECT_ROOT, "artifacts/reports")
    if cache_folder is None: cache_folder = os.path.join(PROJECT_ROOT, "artifacts/cache")

    input_folder = os.path.abspath(input_folder)
    artifacts_folder = os.path.abspath(artifacts_folder)
    cache_folder = os.path.abspath(cache_folder)
    output_folder = os.path.abspath(output_folder)
    """
    優化後的資料前處理管線：
    1. 優先從 Parquet 緩存載入資料。
    2. 若提供 parquet_folder，從使用者的 Parquet 檔案讀取（不需 raw_json）。
    3. 若無緩存，則一次性讀取 JSON 並提取所有欄位（不重複讀取）。
    4. 支援快速重新標籤（只掃描變動的 Regex）。

    parquet_folder: 含有 raw_extracted.parquet + full_text_backup.parquet 的資料夾路徑。
                    若指定此參數，即使 artifacts/cache 中沒有緩存，也不需要 raw_json 資料夾。
                    注意：指定 parquet_folder 時，JTYPE 會重新以修正後的 j_type_check() 計算。
    """
    import time
    from utils.verdict_utils import apply_verdict_rules
    start_time = time.time()

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(artifacts_folder, exist_ok=True)
    os.makedirs(cache_folder, exist_ok=True)

    cache_path = os.path.join(cache_folder, "raw_extracted.parquet")
    full_backup_cache = os.path.join(cache_folder, "full_text_backup.parquet")
    df = None

    # ─── 嘗試載入 Parquet 緩存（parquet_folder 未指定時才走 cache）───────
    # 若明確指定 parquet_folder，跳過 cache，強制重新計算 JTYPE
    if parquet_folder is None and os.path.exists(cache_path):
        print(f"📦 偵測到 Parquet 緩存，正在從 {cache_path} 載入...")
        try:
            df = pd.read_parquet(cache_path)
            print(f"✅ 載入成功！共 {len(df)} 筆資料。")
        except Exception as e:
            print(f"⚠️ 緩存載入失敗 ({e})，將重新讀取原始資料。")

    # ─── 若指定 parquet_folder，從使用者 Parquet 讀取並重新計算 JTYPE ─
    if df is None and parquet_folder is not None:
        parquet_folder = os.path.abspath(parquet_folder)
        raw_ext_path = os.path.join(parquet_folder, "raw_extracted.parquet")
        full_text_path = os.path.join(parquet_folder, "full_text_backup.parquet")
        print(f"📂 從使用者 Parquet 讀取資料 ({parquet_folder})...")
        df = pd.read_parquet(raw_ext_path)
        df_full = pd.read_parquet(full_text_path)
        # 合併 JFULL（重新計算 JTYPE 需要）
        df = df.merge(df_full[["JID", "JFULL"]], on="JID", how="left")
        print(f"✅ 載入成功！共 {len(df)} 筆資料。")

        # 重新計算 JTYPE（修正 RULING 誤判問題）
        print("🔄 重新計算 JTYPE（修正 RULING 位置判斷）...")
        from utils.verdict_utils import j_type_check, ip_law_check
        def _recompute_jtype(row):
            jfull_val = row["JFULL"] if isinstance(row.get("JFULL"), str) else ""
            return j_type_check(JTYPE_PATTERNS, row["JCASE"], jfull_val, row["IP Law"], row["JID"])
        df["JTYPE"] = df.apply(_recompute_jtype, axis=1)
        print("✅ JTYPE 重新計算完成。")

        # 重新計算 main_clause（確保與新 JTYPE 一致）
        from utils.verdict_utils import extract_main_clause
        print("🔄 重新萃取 main_clause...")
        df["main_clause"] = df.apply(
            lambda row: extract_main_clause(MAIN_PATTERNS, row["JFULL"])
            if isinstance(row.get("JFULL"), str) else None, axis=1
        )

        # 儲存全文本備份到 cache
        print(f"📦 正在快速存儲「全文本備份」資料 -> {full_backup_cache}")
        df[["JID", "JFULL"]].to_parquet(full_backup_cache, index=False, compression="snappy")

        # 儲存輕量化核心特徵快取（不含 JFULL）
        # 用 inplace drop 避免建立副本導致 OOM（JFULL 已存入 full_text_backup，可安全釋放）
        import gc
        df.drop(columns=["JFULL"], inplace=True)
        gc.collect()
        df_no_jfull = df
        print(f"⚡️ 正在存儲「輕量化核心特徵」快取 -> {cache_path}")
        df_no_jfull.to_parquet(cache_path, index=False, compression="snappy")

    # ─── 若無緩存且無 parquet_folder，執行 Cold Start (一次讀取 9 萬筆) ─
    if df is None:
        print(f"🚀 緩存不存在，啟動 Cold Start (讀取自 {input_folder})...")
        files = [f.replace(".json", "") for f in os.listdir(input_folder) if f.endswith(".json")]
        SPECIAL_CASES = ["CHDM,101,智附民, 2,20131223,1", "CHDM,102,智簡上, 1,20130510,1"]
        items = [(jid, input_folder, SPECIAL_CASES) for jid in files]
        max_workers = max(1, multiprocessing.cpu_count() - 1) if n_jobs == -1 else n_jobs
        results = []
        with multiprocessing.Pool(processes=max_workers) as pool:
            for res in tqdm(pool.imap_unordered(_process_unified_case, items, chunksize=100),
                            total=len(items), desc="Extracting Raw Data (Unified)"):
                if res is not None:
                    results.append(res)
        df = pd.DataFrame(results)

        # 儲存全文本備份
        print(f"📦 正在快速存儲「全文本備份」資料 -> {full_backup_cache}")
        df[["JID", "JFULL"]].to_parquet(full_backup_cache, index=False, compression="snappy")

        # 儲存輕量化核心特徵快取（不含 JFULL）
        df = df.drop(columns=["JFULL"])
        print(f"⚡️ 正在存儲「輕量化核心特徵」快取 -> {cache_path}")
        df.to_parquet(cache_path, index=False, compression="snappy")

    # ─── CJK 空白字元修正（修正從 cache 載入的 main_clause OCR 問題）──────
    import re as _re
    df["main_clause"] = df["main_clause"].apply(
        lambda mc: _re.sub(r"(?<=[^\x00-\x7F])[ \t]+(?=[^\x00-\x7F])", "", mc)
        if isinstance(mc, str) else mc
    )

    # ─── 執行標籤判定 (Pattern Matching) ───────────────────
    print("⚖️ 正在執行標籤判定 (Pattern Matching)...")
    df = apply_verdict_rules(df, MAIN_PATTERNS, JRESULT_PATTERNS)

    # ─── 套用人工標籤 (MANUAL_LABELING) ────────────────────
    from utils.verdict_utils import map_manual_verdict
    for jid, m in MANUAL_LABELING.items():
        if not jid:
            continue
        if jid not in df["JID"].values:
            continue
        idx = df[df["JID"] == jid].index[0]
        df.at[idx, "JTYPE"] = m.get("j_type", df.at[idx, "JTYPE"])
        if "j_result" in m:
            jfull_val = df.at[idx, "JFULL"] if "JFULL" in df.columns else ""
            if not isinstance(jfull_val, str):
                jfull_val = ""
            df.at[idx, "VERDICT"] = map_manual_verdict(m["j_result"], jfull_val)

    # ─── valid_mask 僅用於 fact_removed_blank 萃取 ─────────
    valid_jtypes = ("ADMINISTRATIVE", "CIVIL", "CRIMINAL", "CWC")
    valid_mask = (
        (df["IP Law"] == True)
        & df["JTYPE"].isin(valid_jtypes)
        & ~df["VERDICT"].isin(["未知", "不重要"])
    )
    # judgment_labels 輸出全部 9 萬筆（不過濾 IP Law）
    # 移除恆為空值/False 的無效欄位（零資訊量）
    _ZERO_INFO_COLS = [
        "victim_names", "victim_is_company", "victim_is_government",
        "company_as_victim_only", "prosecutor_is_company",
        "civil_plaintiff_is_government", "受刑人_is_government",
    ]
    df_labels_out = df.drop(
        columns=["main_clause", "Fact", "JFULL"] + _ZERO_INFO_COLS,
        errors="ignore"
    )

    # ─── 儲存 judgment_labels ──────────────────────────────
    labels_csv = os.path.join(artifacts_folder, "judgment_labels.csv")
    df_labels_out.to_csv(labels_csv, index=False, encoding="utf-8-sig")

    print("📊 正在產出 Excel 報表...")
    excel_tasks = [
        ("Labels", df_labels_out, "judgment_labels.xlsx"),
    ]
    try:
        import shutil
        # 自動選擇可用的 Excel engine（xlsxwriter 優先，fallback 到 openpyxl）
        try:
            import xlsxwriter
            engine = "xlsxwriter"
            engine_kwargs = {"options": {"strings_to_urls": False}}
        except ImportError:
            engine = "openpyxl"
            engine_kwargs = {}
        print(f"  (使用 Excel engine: {engine})")
        for name, data, filename in excel_tasks:
            target = os.path.join(artifacts_folder, filename)
            tmp = os.path.join(os.path.dirname(target), f"~tmp_{filename}")
            print(f"  - 正在產出 {filename}...")
            with pd.ExcelWriter(tmp, engine=engine, engine_kwargs=engine_kwargs) as writer:
                data.to_excel(writer, index=False)
            shutil.move(tmp, target)
        print("✅ Excel 報表產出完成！")
    except Exception as e:
        print(f"⚠️ Excel 產出失敗 ({e})，請使用 CSV 或 Parquet。")

    # ─── 產出 fact_removed_blank.xlsx（供 build_features Step 2 使用）─
    # 若 JFULL 不在 df，從 full_text_backup.parquet 補充載入
    if "JFULL" not in df.columns and os.path.exists(full_backup_cache):
        print("📖 載入 JFULL 進行事實萃取...")
        df_jfull = pd.read_parquet(full_backup_cache)
        df = df.merge(df_jfull[["JID", "JFULL"]], on="JID", how="left")

    fact_removed_blank_path = os.path.join(artifacts_folder, "fact_removed_blank.xlsx")
    if "JFULL" in df.columns:
        SPECIAL_CASES = ["CHDM,101,智附民, 2,20131223,1", "CHDM,102,智簡上, 1,20130510,1"]
        print("✂️ 正在萃取犯罪事實 / 事實及理由文字...")
        fact_records = []
        for _, row in tqdm(df[valid_mask].iterrows(),
                           total=int(valid_mask.sum()), desc="Extracting Facts"):
            jid = row["JID"]
            jfull = row.get("JFULL", "")
            if not isinstance(jfull, str):
                jfull = ""
            fact = extract_fact(jid, jfull, SPECIAL_CASES)
            if fact and fact not in ("No next line found", "No end match found"):
                fact_records.append({"JID": jid, "Text": fact})
        df_fact = pd.DataFrame(fact_records).set_index("JID")
        df_fact = remove_blank(df_fact)
        df_fact.to_excel(fact_removed_blank_path)
        print(f"✅ fact_removed_blank.xlsx 儲存完成，共 {len(df_fact)} 筆。")
    else:
        print("⚠️ 找不到 JFULL，無法產出 fact_removed_blank.xlsx。"
              "請確認 full_text_backup.parquet 存在於 cache_folder。")

    df = df.drop(columns=["JFULL"], errors="ignore")
    return df


if __name__ == "__main__":
    run_preprocessing()
