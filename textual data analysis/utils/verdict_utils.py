import os
import re
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import pandas as pd
from tqdm import tqdm

def ip_law_check(JTITLE_PATTERN: re.Pattern, jtitle: str, jcase: str):
    """
    檢查是否為ip案件
    """
    return JTITLE_PATTERN.search(jtitle) is not None or "智" in jcase


def j_type_check(JTYPE_PATTERNS: dict, jcase: str, jfull: str, ip_law: bool, jid):
    """
    檢查他是哪種類型的案件，民事、形式、行政、刑附民
    若他不是ip_law，或是他為裁定則標記為不重要
    """
    start = re.search(r"主\s*文", jfull)
    if start is not None:
        start_index = start.start()
    else:
        return "不重要"
    jfull = jfull[:start_index]
    jfull = re.sub(r"\s+", "", jfull)

    CWC_PATTERNS = JTYPE_PATTERNS["CWC_PATTERNS"]
    CIVIL_PATTERNS = JTYPE_PATTERNS["CIVIL_PATTERNS"]
    CRIMINAL_PATTERNS = JTYPE_PATTERNS["CRIMINAL_PATTERNS"]
    ADMINISTRATIVE_PATTERNS = JTYPE_PATTERNS["ADMINISTRATIVE_PATTERNS"]
    RULING_PATTERNS = JTYPE_PATTERNS["RULING_PATTERNS"]
    OTHERS_PATTERNS = JTYPE_PATTERNS["OTHERS_PATTERNS"]

    # debug 用
    # if jid == "SCDM,99,審智簡,4,20100628,1":
    #     is_ruling = RULING_PATTERNS.search(jfull)
    #     is_cwc = CWC_PATTERNS.search(jcase)
    #     is_civil = CIVIL_PATTERNS.search(jfull)
    #     is_criminal = CRIMINAL_PATTERNS.search(jfull)
    #     is_administrative = ADMINISTRATIVE_PATTERNS.search(jfull)
    #     is_others = OTHERS_PATTERNS.search(jfull)

    #     jfull_preview = jfull.replace('\n', '\\n').replace('\r', '\\r')
    #     print(f"\n--- Debug JID: {jid} ---")
    #     print(f"Original JFULL start : {jfull_preview}")
    #     print(f"RULING_PATTERNS match: {bool(is_ruling)}")
    #     print(f"CWC_PATTERNS match (jcase): {bool(is_cwc)}")
    #     print(f"CIVIL_PATTERNS match: {bool(is_civil)}")
    #     print(f"CRIMINAL_PATTERNS match: {bool(is_criminal)}")
    #     print(f"ADMINISTRATIVE_PATTERNS match: {bool(is_administrative)}")
    #     print(f"OTHERS_PATTERNS match: {bool(is_others)}")
    #     print(f"IP Law check result: {ip_law}")

    if RULING_PATTERNS.search(jfull):
        return "RULING"
    if (CWC_PATTERNS.search(jcase)) or ("刑事附帶民事訴訟" in jfull):
        return "CWC"

    if (OTHERS_PATTERNS.search(jfull)) or (ip_law is not True):
        return "不重要"
    elif CIVIL_PATTERNS.search(jfull):
        return "CIVIL"
    elif CRIMINAL_PATTERNS.search(jfull):
        return "CRIMINAL"
    elif ADMINISTRATIVE_PATTERNS.search(jfull):
        return "ADMINISTRATIVE"
    else:
        return "未知"


def extract_main_clause(MAIN_PATTERNS: dict, jfull: str):
    """
    提取主文
    """
    START_PATTERNS = MAIN_PATTERNS["START_PATTERNS"]
    END_PATTERNS = MAIN_PATTERNS["END_PATTERNS"]
    start = re.search(START_PATTERNS, jfull)
    if not start:
        start = re.search(r"判決如左", jfull)
        if not start:
            return None
    start_index = start.end()

    end = re.search(END_PATTERNS, jfull[start_index:])
    if end:
        end_index = start_index + end.start()
        main_clause = jfull[start_index:end_index]
    else:
        main_clause = jfull[start_index:]

    main_clause = main_clause.strip().replace("\r\n", "")
    return main_clause


def j_result_check(
    MAIN_PATTERNS: dict, JRESULT_PATTERNS: dict, jfull: str, j_type: str, ip_law: bool, jid
):
    """
    檢查結果是勝訴、敗訴、部分勝訴/敗訴
    若不是要取的樣本則標記為不重要
    """
    if j_type in ["RULING", "不重要"] or ip_law is False:
        return "不重要"

    main_clause = extract_main_clause(MAIN_PATTERNS, jfull)
    if main_clause is None:
        return "不重要"

    # # debug 用
    # if jid == "TPBA,94,訴,444,20060427,2":
    #     print(f"\n--- Debug JID: {jid} in j_result_check ---")
    #     print(f"JTYPE: {j_type}")
    #     print(f"IP Law: {ip_law}")
    #     main_clause_preview = main_clause.replace('\n', '\\n').replace('\r', '\\r')
    #     print(f"Extracted main_clause : {main_clause_preview}")

    SPECIAL_PATTERNS = JRESULT_PATTERNS.get("SPECIAL_PATTERNS", {})
    for key, value in SPECIAL_PATTERNS.items():
        pattern = re.compile(value)
        if pattern.search(main_clause):
            return key

    if j_type == "未知":
        return "j_type == 未知"

    WIN_PATTERNS = JRESULT_PATTERNS[j_type]["WIN_PATTERNS"]
    LOSS_PATTERNS = JRESULT_PATTERNS[j_type]["LOSS_PATTERNS"]
    PARTIAL_PATTERNS = JRESULT_PATTERNS[j_type]["PARTIAL_PATTERNS"]

    if PARTIAL_PATTERNS.search(main_clause):
        return "部分勝訴/敗訴"
    elif WIN_PATTERNS.search(main_clause):
        return "勝訴"
    elif LOSS_PATTERNS.search(main_clause):
        return "敗訴"
    else:
        return "未知"


def _process_single_case(file_path, output_folder):
    from config.patterns import JTITLE_PATTERNS, JTYPE_PATTERNS, MAIN_PATTERNS, JRESULT_PATTERNS, MANUAL_LABELING
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    jid = data.get("JID", "")
    jyear = data.get("JYEAR", "")
    jcase = data.get("JCASE", "")
    jdate = data.get("JDATE", "")
    jtitle = data.get("JTITLE", "")
    jfull = data.get("JFULL", "")
    jpdf = data.get("JPDF", "")

    ip_law = ip_law_check(JTITLE_PATTERNS, jtitle, jcase)
    j_type = j_type_check(JTYPE_PATTERNS, jcase, jfull, ip_law, jid)
    j_result = j_result_check(MAIN_PATTERNS, JRESULT_PATTERNS, jfull, j_type, ip_law, jid)
    
    if j_result == "未知":
        try:
            j_type = MANUAL_LABELING.get(jid, {}).get("j_type", j_type)
            j_result = MANUAL_LABELING.get(jid, {}).get("j_result", j_result)
        except Exception:
            pass

    return {
        "JID": jid,
        "JYEAR": jyear,
        "JCASE": jcase,
        "JDATE": jdate,
        "JTITLE": jtitle,
        "JPDF": jpdf,
        "IP Law": ip_law,
        "JTYPE": j_type,
        "VERDICT": j_result,
        # NO JFULL HERE. Zero IPC overhead!
    }
def classify_cases(
    input_folder: str,
    output_folder: str,
    JTITLE_PATTERNS: re.Pattern,
    JTYPE_PATTERNS: dict,
    MAIN_PATTERNS: dict,
    JRESULT_PATTERNS: dict,
    MANUAL_LABELING: dict,
    n_jobs: int = -1,
):
    """
    得到分類的 excel，且將 IP_law 的案件複製到 output_folder (平行運算版)
    """
    os.makedirs(output_folder, exist_ok=True)
    label_file = "judgment_labels.xlsx"
    result_list = []
    
    files_to_process = [f for f in os.listdir(input_folder) if f.endswith(".json")]
    
    # 決定 CPU 核心數
    # 改用 ThreadPoolExecutor 來重疊 macOS 硬碟 I/O 延遲，同時達到零 IPC 通訊成本
    if n_jobs == 1:
        for file_name in tqdm(files_to_process, mininterval=0.5, desc="Classifying Cases"):
            file_path = os.path.join(input_folder, file_name)
            result = _process_single_case(file_path, output_folder)
            result_list.append(result)
    else:
        max_workers = 32 if n_jobs == -1 else n_jobs * 2
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            from functools import partial
            func = partial(_process_single_case, output_folder=output_folder)
            file_paths = [os.path.join(input_folder, f) for f in files_to_process]
            
            for res in tqdm(executor.map(func, file_paths, chunksize=50), total=len(file_paths), mininterval=0.5, desc="Classifying Cases"):
                result_list.append(res)

    df = pd.DataFrame(result_list)
    return df
