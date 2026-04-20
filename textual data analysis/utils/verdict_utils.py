import os
import re
import json
import shutil
import concurrent.futures
import multiprocessing
import pandas as pd
from tqdm import tqdm

# 舊的 VERDICT 到新 VERDICT 類別的映射
VERDICT_MAPPING = {
    "勝訴": "Win",
    "敗訴": "Lose",
    "部分勝訴/敗訴": "Mixed",
    "和解成功": None,  
    "部分和解": "Settlement_Partial",
    "和解失敗": "Settlement_Failure",
    "發回更審": "Remand",
    "不受理/程序駁回": "Other",
    "停止訴訟": "Other",
    "更正判決": "Other",
    "移送": "Other",
    "免訴": "Other",
    "補充判決": "Other",
    "程序撤銷": "Other",
    # 加入已經是英文的目標，讓動態加載的 verdict_mismatches_minimal 可以直接直通
    "Win": "Win",
    "Lose": "Lose",
    "Mixed": "Mixed",
    "Remand": "Remand",
    "Settlement_Success": "Settlement_Success",
    "Settlement_Partial": "Settlement_Partial",
    "Settlement_Failure": "Settlement_Failure",
    "Other": "Other"
}

def map_manual_verdict(manual_result: str, jfull: str) -> str:
    from config.patterns import MAIN_PATTERNS
    if manual_result == "和解成功":
        main_clause = extract_main_clause(MAIN_PATTERNS, jfull)
        if main_clause and re.search(r"和解成立|調解成立", main_clause):
            return "Settlement_Success"
        return "Other"
    return VERDICT_MAPPING.get(manual_result, "Other")

def ip_law_check(JTITLE_PATTERN: re.Pattern, jtitle: str, jcase: str):
    return JTITLE_PATTERN.search(jtitle) is not None or "智" in jcase

def j_type_check(JTYPE_PATTERNS: dict, jcase: str, jfull: str, ip_law: bool, jid):
    start = re.search(r"主\s*文", jfull)
    if start is not None:
        start_index = start.start()
    else:
        return "不重要"
    jfull = jfull[:start_index]
    jfull = re.sub(r"\s+", "", jfull)

    if JTYPE_PATTERNS["RULING_PATTERNS"].search(jfull):
        return "RULING"
    if (JTYPE_PATTERNS["CWC_PATTERNS"].search(jcase)) or ("刑事附帶民事訴訟" in jfull):
        return "CWC"
    if (JTYPE_PATTERNS["OTHERS_PATTERNS"].search(jfull)) or (ip_law is not True):
        return "不重要"
    elif JTYPE_PATTERNS["CIVIL_PATTERNS"].search(jfull):
        return "CIVIL"
    elif JTYPE_PATTERNS["CRIMINAL_PATTERNS"].search(jfull):
        return "CRIMINAL"
    elif JTYPE_PATTERNS["ADMINISTRATIVE_PATTERNS"].search(jfull):
        return "ADMINISTRATIVE"
    else:
        return "未知"

def extract_main_clause(MAIN_PATTERNS: dict, jfull: str):
    if jfull is None or not isinstance(jfull, str):
        return None
        
    START_PATTERNS = MAIN_PATTERNS["START_PATTERNS"]
    END_PATTERNS = MAIN_PATTERNS["END_PATTERNS"]
    start = re.search(START_PATTERNS, jfull)
    if not start:
        start = re.search(r"判決如左", jfull)
        if not start: return None
    start_index = start.end()
    
    end = re.search(END_PATTERNS, jfull[start_index:])
    if end:
        main_clause = jfull[start_index:start_index + end.start()]
    else:
        # 當找不到明確的理由或事實結尾時，設定硬性長度上限 (1500字)，避免將整篇超過幾萬字的判決書全部送出
        main_clause = jfull[start_index:start_index + 1500]
        
    main_clause = main_clause.strip().replace("\r\n", "").replace("\n", "")
    # 移除 Excel 所無法接受的不可見 XML 亂碼字元（這會導致 fact_removed_blank 損毀並觸發修復動作）
    main_clause = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", main_clause)
    
    return main_clause

def j_result_check(MAIN_PATTERNS, JRESULT_PATTERNS, jfull, j_type, ip_law, jid, main_clause=None):
    if main_clause is None or (isinstance(main_clause, float) and pd.isna(main_clause)):
        if jfull is not None:
            main_clause = extract_main_clause(MAIN_PATTERNS, jfull)
        else:
            return "Other"
    
    if main_clause is None: return "Other"
    
    SPECIAL_PATTERNS = JRESULT_PATTERNS.get("SPECIAL_PATTERNS", {})
    if re.search(SPECIAL_PATTERNS.get("Remand", r"$^"), main_clause): return "Remand"

    if j_type == "CIVIL":
        if re.search(SPECIAL_PATTERNS.get("Settlement_Success", r"$^"), main_clause): return "Settlement_Success"
        if re.search(SPECIAL_PATTERNS.get("Settlement_Partial", r"$^"), main_clause): return "Settlement_Partial"
        if re.search(SPECIAL_PATTERNS.get("Settlement_Failure", r"$^"), main_clause): return "Settlement_Failure"

    for k, v in SPECIAL_PATTERNS.items():
        if k not in ["Settlement_Success", "Settlement_Partial", "Settlement_Failure", "Remand"] and re.search(v, main_clause):
            return "Other"

    if j_type in ["RULING", "不重要"] or not ip_law: return "Other"
    if j_type not in JRESULT_PATTERNS: return "Other"

    patterns = JRESULT_PATTERNS[j_type]
    if re.search(patterns.get("PARTIAL_PATTERNS", r"$^"), main_clause): return "Mixed"
    if re.search(patterns.get("WIN_PATTERNS", r"$^"), main_clause): return "Win"
    if re.search(patterns.get("LOSS_PATTERNS", r"$^"), main_clause): return "Lose"
    return "Other"

def apply_verdict_rules(df, MAIN_PATTERNS, JRESULT_PATTERNS):
    """
    對 DataFrame 進行批次標籤判定，主要用於暖啟動（從緩存載入時）。
    """
    def _row_check(row):
        return j_result_check(
            MAIN_PATTERNS, JRESULT_PATTERNS, 
            jfull=None, # 當提供 main_clause 時不需要 jfull
            j_type=row["JTYPE"], 
            ip_law=row["IP Law"], 
            jid=row["JID"], 
            main_clause=row["main_clause"]
        )
    
    df["VERDICT"] = df.apply(_row_check, axis=1)
    return df

def _process_single_case(file_path, output_folder):
    # 局部 import 利用快取，且避開多進程序列化問題
    from config.patterns import JTITLE_PATTERNS, JTYPE_PATTERNS, MAIN_PATTERNS, JRESULT_PATTERNS, MANUAL_LABELING
    from utils.role_extractor import extract_role_features
    try:
        if os.path.getsize(file_path) == 0: return None
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except: return None

    jid, jtitle, jcase, jfull = data.get("JID", ""), data.get("JTITLE", ""), data.get("JCASE", ""), data.get("JFULL", "")
    ip_law = ip_law_check(JTITLE_PATTERNS, jtitle, jcase)
    j_type = j_type_check(JTYPE_PATTERNS, jcase, jfull, ip_law, jid)
    main_clause = extract_main_clause(MAIN_PATTERNS, jfull)
    j_result = j_result_check(MAIN_PATTERNS, JRESULT_PATTERNS, jfull, j_type, ip_law, jid)

    if jid in MANUAL_LABELING:
        m = MANUAL_LABELING[jid]
        j_type = m.get("j_type", j_type)
        if "j_result" in m: j_result = map_manual_verdict(m["j_result"], jfull)

    role_f = extract_role_features(jfull, j_type)
    return {
        "JID": jid, "JYEAR": data.get("JYEAR",""), "JCASE": jcase, "JDATE": data.get("JDATE",""),
        "JTITLE": jtitle, "JPDF": data.get("JPDF",""), "IP Law": ip_law, "JTYPE": j_type,
        "VERDICT": str(j_result), "main_clause": main_clause,
        **{k: role_f.get(k, False) for k in ["plaintiff_is_company", "defendant_is_company", "appellant_is_company", "appellee_is_company", "complainant_is_company", "victim_is_company", "prosecutor_present", "company_vs_company", "company_involved", "company_vs_individual", "individual_vs_company", "company_as_victim_only", "company_as_defendant_only", "is_civil", "is_pure_criminal", "is_attached_civil", "is_admin", "is_appeal", "is_first_instance", "is_summary_case", "claim_damages", "claim_injunction", "claim_destroy_goods", "claim_validity_review", "claim_admin_cancellation"]}
    }

def classify_cases(
    input_folder: str,
    output_folder: str,
    JTITLE_PATTERNS=None,
    JTYPE_PATTERNS=None,
    MAIN_PATTERNS=None,
    JRESULT_PATTERNS=None,
    MANUAL_LABELING=None,
    n_jobs: int = -1,
):
    """
    判決書分類。Patterns 參數保留以相容呼叫端，
    但實際由 _process_single_case 內部 import 取得（利用 sys.modules 快取）。
    """
    os.makedirs(output_folder, exist_ok=True)
    files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".json")]

    # 針對大量小檔案 I/O 密集型任務，ThreadPoolExecutor 表現更優
    max_workers = 32 if n_jobs == -1 else n_jobs
    from functools import partial
    func = partial(_process_single_case, output_folder=output_folder)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(func, files), total=len(files), mininterval=1, desc="Classifying Cases"))

    return pd.DataFrame([r for r in results if r])
