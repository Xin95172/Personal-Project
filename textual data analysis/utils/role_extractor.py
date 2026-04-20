"""
角色導向抽取模組 (Role-Oriented Extraction Module)

從判決書標頭中精確辨識程序主體（原告、被告等），
並判斷各角色是否為公司法人，以產出 company_vs_company 等分類變數。

設計原則：
1. 只從標頭區段（主文之前）抽取角色，避免全文關鍵字誤判。
2. 區分「程序主體」與「文本提及主體」。
3. 產出嚴格 (company_vs_company) 與寬鬆 (company_involved) 兩種分類。
"""

import re
from typing import Optional


# ─── 公司法人判定規則 ─────────────────────────────────────
COMPANY_SUFFIXES = (
    "股份有限公司",
    "有限公司",
    "有限責任公司",
    "合夥企業",
    "合作社",
    "企業社",
    "商行",
    "工業社",
    "實業社",
    # 外國公司常見寫法
    "Co., Ltd.",
    "Co.,Ltd.",
    "Corp.",
    "Corporation",
    "Inc.",
    "LLC",
    "L.L.C.",
    "Ltd.",
    "B.V.",
    "GmbH",
    "S.A.",
    "K.K.",
    "Pty Ltd",
)

COMPANY_PREFIXES = (
    "日商", "美商", "英商", "德商", "法商", "韓商",
    "新加坡商", "瑞士商", "荷蘭商", "香港商", "澳商",
    "加拿大商", "義大利商", "瑞典商", "丹麥商", "盧森堡商",
    "開曼群島商", "英屬維京群島商", "薩摩亞商",
    "馬來西亞商", "泰商", "愛爾蘭商", "百慕達商", "巴哈馬商",
)

GOV_KEYWORDS = (
    "經濟部智慧財產局", "智慧財產局", "經濟部", "財政部",
    "行政院", "檢察署", "檢察官", "公訴人",
    "公所", "縣政府", "市政府", "鄉公所", "鎮公所", "區公所",
)


def is_company(name: str) -> bool:
    """
    判斷一個名稱是否為公司法人。
    排除政府機關、自然人、檢察官等。
    """
    if not name:
        return False

    name_clean = re.sub(r"\s+", "", name)

    # 過長的字串不可能是純名稱，應該是解析錯誤
    if len(name_clean) > 80:
        return False

    # 先排除政府機關
    for kw in GOV_KEYWORDS:
        if kw in name_clean:
            return False

    # 排除律師
    if "律師" in name_clean:
        return False

    # 檢查公司後綴
    for suffix in COMPANY_SUFFIXES:
        if suffix in name_clean:
            return True

    # 檢查外商前綴
    for prefix in COMPANY_PREFIXES:
        if name_clean.startswith(prefix):
            return True

    # 包工業、工程行等也算企業
    if re.search(r"(包工業|工程行|企業行|營造廠|工廠)$", name_clean):
        return True

    return False


def is_government(name: str) -> bool:
    """判斷一個名稱是否為政府機關"""
    if not name:
        return False
    name_clean = re.sub(r"\s+", "", name)
    for kw in GOV_KEYWORDS:
        if kw in name_clean:
            return True
    return False


# ─── 角色抽取正則表達式 ─────────────────────────────────────
# 核心概念：判決書標頭的「角色行」有固定格式
# 1. 行首為角色標籤（可能有全形空白分隔）
# 2. 標籤後接空白，再接名稱
# 3. 名稱通常不超過 40 個字元
# 4. 非角色行（如「上列當事人間因...」）的特徵是：
#    - 角色標籤不在行首或接近行首
#    - 標籤後面是完整句子而非名稱

# 角色標籤：只匹配「行首」出現的標籤
# 使用寬鬆的空白匹配（全形+半形）
ROLE_PATTERNS = {
    # ─ 民事核心對立方（含再審）─
    "plaintiff":       re.compile(r"^[\s　]*(?:再\s*審)?\s*原\s*告[\s　]"),
    "defendant":       re.compile(r"^[\s　]*(?:再\s*審)?\s*被\s*告[\s　]"),
    # ─ 上訴/抗告（含再抗告）─
    "appellant":       re.compile(r"^[\s　]*(?:再)?\s*上\s*訴\s*人[\s　]"),
    "appellee":        re.compile(r"^[\s　]*(?:再)?\s*被\s*上?\s*訴?\s*人[\s　]"),
    "抗告人":          re.compile(r"^[\s　]*(?:再)?\s*抗\s*告\s*人[\s　]"),
    "相對人":          re.compile(r"^[\s　]*相\s*對\s*人[\s　]"),
    # ─ 聲請 ─
    "applicant":       re.compile(r"^[\s　]*聲\s*請\s*人[\s　]"),
    # ─ 刑事 ─
    "prosecutor":      re.compile(r"^[\s　]*公\s*訴\s*人[\s　]"),
    # ─ 自訴 ─
    "civil_plaintiff": re.compile(r"^[\s　]*自\s*訴\s*人[\s　]"),
    # ─ 告訴人/被害人 ─
    "complainant":     re.compile(r"^[\s　]*告\s*訴\s*人[\s　]"),
    "victim":          re.compile(r"^[\s　]*被\s*害\s*人[\s　]"),
    # ─ 其他 ─
    "participant":     re.compile(r"^[\s　]*參\s*加\s*人[\s　]"),
    "受刑人":          re.compile(r"^[\s　]*受\s*刑\s*人[\s　]"),
}

# 需要排除的「段落敘述行」：含有角色關鍵字但不是當事人列表
# 特徵：行首不是角色標籤，而是敘述性文字
NOISE_PATTERNS = re.compile(
    r"(上列|右列|前列|左列|上開).*?(間|因|之|等)|"
    r"(本院.*?判決|裁定如下|裁定如左|言詞辯論終結|判決如下|判決如左)|"
    r"(經檢察官|經自訴人|經原告|提起上訴|提起附帶|提起自訴|提起行政)|"
    r"(聲請.*?簡易判決|聲請.*?沒收|聲請.*?處刑|聲請.*?定|聲請.*?宣告)|"
    r"(第一審|第二審|更為審判)"
)


def _extract_header(jfull: str) -> str:
    """取得主文之前的標頭段落。"""
    match = re.search(r"主\s*文", jfull)
    if match:
        return jfull[: match.start()]
    return ""


def _is_role_line(line: str) -> bool:
    """
    判斷一行是否為「角色列表行」而非「段落敘述行」。

    角色行的特徵：
    - 行首（可含空白）直接接角色標籤
    - 標籤後接空白再接名稱
    - 行長度通常不超過 80 個字元（含空白）

    非角色行的特徵：
    - 行內有完整的句子結構（如「間」「因」「之」等連接詞）
    - 含有案件描述用語（如「本院判決如下」）
    """
    line_clean = re.sub(r"\s+", "", line)

    # 太長的行不太可能是角色行
    if len(line_clean) > 60:
        return False

    return True


def _parse_role_name_v2(line: str, role_pattern: re.Pattern) -> Optional[str]:
    """
    從一行中，在角色標籤後面抽取名稱。
    例如：「原　　　告　耐斯企業股份有限公司」→ 回傳「耐斯企業股份有限公司」

    改進版：增加名稱合理性檢查。
    """
    match = role_pattern.search(line)
    if not match:
        return None

    # 取得標籤之後的文字（pattern 已包含尾部空白）
    after_label = line[match.end():]

    # 移除前導空白（全形+半形）
    after_label = re.sub(r"^[\s　]+", "", after_label)

    if not after_label:
        return None

    name = after_label.strip()

    # 用多個空白分割，取第一段作為名稱（後面通常是地址、身分證等）
    name = re.split(r"\s{2,}|　{2,}", name)[0]

    # 移除括號中的職稱
    name = re.sub(r"[（(].{0,20}?[）)]", "", name)

    # 移除住所資訊
    name = re.sub(r"住同[上右左]", "", name)
    name = re.sub(r"住.+$", "", name)

    name = name.strip()

    # ─── 名稱合理性檢查 ───────────────
    # 太短：可能是解析錯誤
    if len(name) < 2:
        return None

    # 太長：可能是段落文字被誤抓
    name_clean = re.sub(r"\s+", "", name)
    if len(name_clean) > 50:
        return None

    # 包含句子結構特徵：不是名稱
    if re.search(r"(間|因|之|等案件|案件|事件，|提起|不服|對於|本院|判決)", name_clean):
        return None

    # 包含年度字號：不是名稱
    if re.search(r"\d+年度", name_clean):
        return None

    return name


def extract_roles(jfull: str) -> dict:
    """
    從判決書全文中抽取標頭段落的角色資訊。

    回傳格式：包含所有角色名稱、公司判定、關係型變數等。
    """
    header = _extract_header(jfull)
    if not header:
        return _empty_result()

    # 統一換行符
    header = header.replace("\r\n", "\n").replace("\r", "\n")
    lines = header.split("\n")

    # 第一階段：抽取所有角色的名稱
    roles = {key: [] for key in ROLE_PATTERNS}

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # 先檢查這行是否像角色列表行
        if not _is_role_line(line):
            continue

        # 跳過噪音行
        line_clean = re.sub(r"\s+", "", line_stripped)
        if NOISE_PATTERNS.search(line_clean):
            continue

        for role_key, role_pattern in ROLE_PATTERNS.items():
            if role_pattern.search(line):
                name = _parse_role_name_v2(line, role_pattern)
                if name and name not in roles[role_key]:
                    roles[role_key].append(name)

    # ─── 聲請人為檢察官時，歸入 prosecutor ──────────
    applicant_names = roles.get("applicant", [])
    prosecutor_from_applicant = []
    real_applicants = []
    for name in applicant_names:
        if "檢察" in name:
            prosecutor_from_applicant.append(name)
        else:
            real_applicants.append(name)

    if prosecutor_from_applicant:
        roles["prosecutor"].extend(prosecutor_from_applicant)
        roles["applicant"] = real_applicants

    # 第二階段：判斷各角色是否為公司
    result = {}
    for role_key in ROLE_PATTERNS:
        result[f"{role_key}_names"] = roles[role_key]
        result[f"{role_key}_is_company"] = any(
            is_company(n) for n in roles[role_key]
        )
        result[f"{role_key}_is_government"] = any(
            is_government(n) for n in roles[role_key]
        )

    # 第三階段：建立核心對立關係
    prosecution_side_keys = [
        "plaintiff", "appellant", "applicant",
        "civil_plaintiff", "complainant", "抗告人",
    ]
    defense_side_keys = ["defendant", "appellee", "相對人"]

    prosecution_names = []
    for key in prosecution_side_keys:
        prosecution_names.extend(roles.get(key, []))

    defense_names = []
    for key in defense_side_keys:
        defense_names.extend(roles.get(key, []))

    prosecution_has_company = any(is_company(n) for n in prosecution_names)
    defense_has_company = any(is_company(n) for n in defense_names)
    prosecution_has_gov = any(is_government(n) for n in prosecution_names)
    defense_has_gov = any(is_government(n) for n in defense_names)

    # 檢察官是否出現（刑事案件指標）
    prosecutor_present = bool(roles.get("prosecutor", []))

    # ─── 關係型變數 ─────────────────────────────
    result["company_vs_company"] = (
        prosecution_has_company and defense_has_company
    )
    all_important_names = prosecution_names + defense_names
    all_important_names.extend(roles.get("complainant", []))
    all_important_names.extend(roles.get("victim", []))
    result["company_involved"] = any(is_company(n) for n in all_important_names)

    result["company_vs_individual"] = (
        prosecution_has_company and not defense_has_company and not defense_has_gov
    )
    result["individual_vs_company"] = (
        not prosecution_has_company
        and not prosecution_has_gov
        and defense_has_company
    )
    result["company_as_victim_only"] = (
        any(is_company(n) for n in roles.get("complainant", []))
        or any(is_company(n) for n in roles.get("victim", []))
    ) and not prosecution_has_company
    result["company_as_defendant_only"] = (
        defense_has_company and not prosecution_has_company
    )

    # ─── 程序型變數 ─────────────────────────────
    result["prosecutor_present"] = prosecutor_present
    result["prosecution_names"] = prosecution_names
    result["defense_names"] = defense_names

    return result


def _empty_result() -> dict:
    """回傳空的結果字典"""
    result = {}
    for role_key in ROLE_PATTERNS:
        result[f"{role_key}_names"] = []
        result[f"{role_key}_is_company"] = False
        result[f"{role_key}_is_government"] = False
    result["company_vs_company"] = False
    result["company_involved"] = False
    result["company_vs_individual"] = False
    result["individual_vs_company"] = False
    result["company_as_victim_only"] = False
    result["company_as_defendant_only"] = False
    result["prosecutor_present"] = False
    result["prosecution_names"] = []
    result["defense_names"] = []
    return result


def extract_role_features(jfull: str, j_type: str) -> dict:
    """
    從判決書全文與案件類型中，產出完整的角色特徵變數。
    這是外部呼叫的主要介面。
    """
    roles = extract_roles(jfull)

    # ─── 程序型變數 ─────────────────────────────
    roles["is_civil"] = j_type == "CIVIL"
    roles["is_pure_criminal"] = j_type == "CRIMINAL"
    roles["is_attached_civil"] = j_type == "CWC"
    roles["is_admin"] = j_type == "ADMINISTRATIVE"

    # 審級判斷
    header = _extract_header(jfull)
    header_clean = re.sub(r"\s+", "", header)

    roles["is_appeal"] = bool(
        re.search(r"(上訴|上字|抗告|再審)", header_clean)
    )
    roles["is_first_instance"] = not roles["is_appeal"]
    roles["is_summary_case"] = bool(re.search(r"(簡易|簡字)", header_clean))

    # ─── 救濟型變數（從主文判斷）─────────────────────
    main_match = re.search(r"主\s*文", jfull)
    if main_match:
        main_start = main_match.end()
        end_match = re.search(
            r"(\n\s+|\s+)(事\s*實|理\s*由|附\s*錄|事實及理由)",
            jfull[main_start:],
        )
        if end_match:
            main_clause = jfull[main_start: main_start + end_match.start()]
        else:
            main_clause = jfull[main_start: main_start + 2000]

        mc = re.sub(r"\s+", "", main_clause)

        roles["claim_damages"] = bool(re.search(r"(給付|賠償|損害)", mc))
        roles["claim_injunction"] = bool(
            re.search(r"(應停止|不得.{0,30}?使用|排除侵害|禁止)", mc)
        )
        roles["claim_destroy_goods"] = bool(re.search(r"(銷毀|沒收|回收銷毀)", mc))
        roles["claim_validity_review"] = bool(
            re.search(r"(撤銷|廢棄|原處分|訴願決定)", mc)
        )
        roles["claim_admin_cancellation"] = bool(
            re.search(r"(註冊應予撤銷|廢止|評定)", mc)
        )
    else:
        roles["claim_damages"] = False
        roles["claim_injunction"] = False
        roles["claim_destroy_goods"] = False
        roles["claim_validity_review"] = False
        roles["claim_admin_cancellation"] = False

    return roles
