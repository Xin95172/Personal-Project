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
        "JYEAR": data.get("JYEAR",""), 
        "JCASE": jcase, 
        "JDATE": data.get("JDATE",""),
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
                      artifacts_folder=None, cache_folder=None, n_jobs=-1):
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
    2. 若無緩存，則一次性讀取 JSON 並提取所有欄位（不重複讀取）。
    3. 支援快速重新標籤（只掃描變動的 Regex）。
    """
    import time
    from utils.verdict_utils import apply_verdict_rules
    start_time = time.time()
    
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(artifacts_folder, exist_ok=True)
    os.makedirs(cache_folder, exist_ok=True)
    
    cache_path = os.path.join(cache_folder, "raw_extracted.parquet")
    df = None

    # ─── 嘗試載入 Parquet 緩存 ───────────────────────────
    if os.path.exists(cache_path):
        print(f"📦 偵測到 Parquet 緩存，正在從 {cache_path} 載入...")
        try:
            df = pd.read_parquet(cache_path)
            print(f"✅ 載入成功！共 {len(df)} 筆資料。")
        except Exception as e:
            print(f"⚠️ 緩存載入失敗 ({e})，將重新讀取原始資料。")

    # ─── 若無緩存，執行 Cold Start (一次讀取 9 萬筆) ────────
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
                if res: results.append(res)
        
        df = pd.DataFrame(results)
        
        # ─── 數據整理與備份策略 (恢復高速分離存儲) ───
        full_backup_path = os.path.join(cache_folder, "full_text_backup.parquet")
        print(f"📦 正在快速存儲「全文本備份」資料 -> {full_backup_path}")
        df[["JID", "JFULL"]].to_parquet(full_backup_path, index=False, compression="snappy")
        
        # 移除肥大欄位，保留核心快取
        df = df.drop(columns=["JFULL"])
        print(f"⚡️ 正在存儲「輕量化核心特徵」快取 -> {cache_path}")
        df.to_parquet(cache_path, index=False, compression="snappy")

    # ─── 執行標籤判定 (暖啟動：極快) ──────────────────────
    print("⚖️ 正在執行標籤判定 (Pattern Matching)...")
    df = apply_verdict_rules(df, MAIN_PATTERNS, JRESULT_PATTERNS)
    
    # 處理手動標籤覆蓋
    for jid, m in MANUAL_LABELING.items():
        if jid in df["JID"].values:
            idx = df[df["JID"] == jid].index[0]
            df.at[idx, "JTYPE"] = m.get("j_type", df.at[idx, "JTYPE"])
            if "j_result" in m:
                from utils.verdict_utils import map_manual_verdict
                # 這裡需要傳入 jfull
                jfull_val = df.at[idx, "JFULL"] if "JFULL" in df.columns else ""
                df.at[idx, "VERDICT"] = map_manual_verdict(m["j_result"], jfull_val)

    # 目前已不再回傳 Fact 欄位以節省空間，故跳過事實清洗邏輯
    # print("🧹 執行事實清洗 (remove_blank)...")
    # df["Fact"] = df["Fact"].apply(lambda x: "".join(str(x).split()) if pd.notnull(x) else x)
    
    # ─── 儲存標籤報表 (Labels) ───────────────────────────
    valid_jtypes = ("ADMINISTRATIVE", "CIVIL", "CRIMINAL", "CWC")
    valid_mask = (df["IP Law"] == True) & (df["JTYPE"].isin(valid_jtypes)) & (~df["VERDICT"].isin(["未知", "不重要"]))
    df_labels_out = df[valid_mask].drop(columns=["main_clause", "Fact", "JFULL"], errors="ignore")
    
    labels_csv = os.path.join(artifacts_folder, "judgment_labels.csv")
    df_labels_out.to_csv(labels_csv, index=False, encoding='utf-8-sig')
    
    # ─── 產出 Excel (分開產出以防超時) ──────────────────────
    print("📊 正在產出 Excel 報表...")
    try:
        excel_tasks = [
            ("Labels", df_labels_out, "judgment_labels.xlsx"),
            ("Raw_Extracted", df, "raw_extracted.xlsx")
        ]
        import shutil
        for name, data, filename in excel_tasks:
            target = os.path.join(artifacts_folder, filename)
            tmp = f"/tmp/{filename}"
            print(f"  - 正在產出 {filename}...")
            # 注意：這裡會將 Facts 與 JFULL 全量寫入 raw_extracted.xlsx
            with pd.ExcelWriter(tmp, engine='xlsxwriter', engine_kwargs={'options': {'strings_to_urls': False}}) as writer:
                data.to_excel(writer, index=False)
            shutil.move(tmp, target)
        print("✅ Excel 報表產出完成！")
    except Exception as e:
        print(f"⚠️ Excel 產出失敗 ({e})，請使用 CSV 或 Parquet。")

    total_time = (time.time() - start_time) / 60
    print(f"✨ 全流程耗時: {total_time:.2f} 分鐘")
    return df

if __name__ == "__main__":
    run_preprocessing()

if __name__ == "__main__":
    df = run_preprocessing()
