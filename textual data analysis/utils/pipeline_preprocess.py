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

def _process_single_fact_in_memory(args):
    jid, SPECIAL_CASES, input_folder = args
    import os
    import json
    from utils import extract_fact
    
    # READ JSON DIRECTLY IN WORKER TO BYPASS IPC PIPE BOTTLENECKS
    file_path = os.path.join(input_folder, f"{jid}.json")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        jfull = data.get("JFULL", "")
    except Exception:
        jfull = ""
        
    text = extract_fact(jid, jfull, SPECIAL_CASES)
    return jid, text

def run_preprocessing(input_folder="../data/raw_json", output_folder="../data/IP_Law_cases", artifacts_folder="../artifacts/reports", n_jobs=-1):
    """
    執行資料前處理管線，包含：判決書分類、萃取犯罪事實、移除空資料。
    """
    print("1. 執行 classify_cases 分類判決書 (純記憶體精簡版，無視 macOS I/O 限制)...")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(artifacts_folder, exist_ok=True)
    
    # 已經修復正則運算 (END_PATTERNS) 的重大效能重疊問題，
    # 這裡可恢復原本的 n_jobs 設定以啟用平行運算，大幅加速整體處理時間。
    df_labels = classify_cases(
        input_folder,
        output_folder,
        JTITLE_PATTERNS,
        JTYPE_PATTERNS,
        MAIN_PATTERNS,
        JRESULT_PATTERNS,
        MANUAL_LABELING,
        n_jobs=n_jobs
    )
    
    # Filter valid cases
    valid_jtypes = ("ADMINISTRATIVE", "CIVIL", "CRIMINAL", "CWC")
    valid_cases_df = df_labels[
        (df_labels["IP Law"] == True) & 
        (df_labels["JTYPE"].isin(valid_jtypes)) & 
        (~df_labels["VERDICT"].isin(["未知", "不重要"]))
    ]
    
    # Save labels
    labels_path = os.path.join(artifacts_folder, "judgment_labels.xlsx")
    df_labels.to_excel(labels_path, index=False)
    print(f"判決標籤已儲存至: {labels_path}")

    print("2. 執行 extract_fact 萃取判決事實 (從原始 JSON 讀取，零 IPC 序列化開銷)...")
    SPECIAL_CASES = ["CHDM,101,智附民, 2,20131223,1", "CHDM,102,智簡上, 1,20130510,1"]
    
    # Prepare arguments for map (only pass JID, completely bypass 64KB pipe limit)
    items_to_process = [(row["JID"], SPECIAL_CASES, input_folder) for _, row in valid_cases_df.iterrows()]
    
    jids = []
    texts = []
    
    if n_jobs == 1:
        for args in tqdm(items_to_process, mininterval=0.5, desc="Extracting Facts"):
            jid, text = _process_single_fact_in_memory(args)
            if text:
                jids.append(jid)
                texts.append(text)
    else:
        max_workers = max(1, multiprocessing.cpu_count() - 1) if n_jobs == -1 else n_jobs
        with multiprocessing.Pool(processes=max_workers) as pool:
            for jid, text in tqdm(pool.imap_unordered(_process_single_fact_in_memory, items_to_process, chunksize=50), total=len(items_to_process), mininterval=0.5, desc="Extracting Facts"):
                if text:
                    jids.append(jid)
                    texts.append(text)

    df_fact = pd.DataFrame({"JID": jids, "Text": texts})
    fact_path = os.path.join(artifacts_folder, "fact.xlsx")
    df_fact.to_excel(fact_path)

    print("3. 執行 remove_blank 移除空資料...")
    df = pd.read_excel(fact_path)
    df.set_index("JID", inplace=True)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
        
    df_removed_blank = remove_blank(df)
    clean_fact_path = os.path.join(artifacts_folder, "fact_removed_blank.xlsx")
    df_removed_blank.to_excel(clean_fact_path)
    
    print(f"前處理完成！清理後的資料儲存於: {clean_fact_path}")
    return df_removed_blank

if __name__ == "__main__":
    df = run_preprocessing()
