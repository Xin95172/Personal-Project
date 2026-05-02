import re
import unicodedata
import pandas as pd

def normalize_zh(text: str) -> str:
    t = unicodedata.normalize('NFKC', text)
    t = t.replace('\u3000', ' ').replace('\xa0', ' ')
    t = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1F\x7F]', '', t)
    return t


def start_for_special_cases(jid: str, text: str) -> re.Match | None:
    """
    處理特殊案例
    """
    text = normalize_zh(text)
    if jid == 'CHDM,101,智附民,2,20131223,1':
        start = re.search(r'(?m)^\s*三、', text)
    elif jid == 'CHDM,102,智簡上,1,20130510,1':
        start = re.search(r'(?m)^\s*二、上訴意旨略以:', text)

    return start if start else None


def extract_fact(jid: str, text: str, SPECIAL_CASES: list) -> str | None:
    """
    抓事實、理由
    """
    text = normalize_zh(text)
    match = None
    next_top = None

    UPPER_NUM_PATTERNS = r'[壹貳參肆伍陸柒捌玖]'
    U_L_NUM_PATTERNS = r'[一二三四五六七八九壹貳參肆伍陸柒捌玖]'
    SPECIAL_PATTERNS = r'\([一二三四]\)'
    NEXT_TOP = r'(實體|認事論罪|事實(部[份分]|概要|及理由)|再審原告主張|原告主張|兩造(聲明及陳述|的主張及抗辯)|關於自訴\(告訴\)權部分)|得心證[之的]理由|被告方面|本訴部分|撤銷改判之理由與檢察官上訴理由之審酌'

    start = re.search(
        r'(\r?\n)\s*'
        r'(?:\d+\s*)?'
        r'({U_L_NUM_PATTERNS}、)?((犯\s*罪)?(1信股)?\s*事\s*實\s*及\s*理\s*由'
        r'|(犯\s*罪)?\s*事\s*實(要旨|概要)?|理\s*由)'
        r'|事理及理由'
        r'|判決事實及理由要領|訴訟標的及理由要領'
        r'[:：]?'.format(U_L_NUM_PATTERNS=U_L_NUM_PATTERNS),
        text,
    )

    if jid == 'NHEM,101,湖簡,346,20121220,1':
        start = re.search(r'\s?二、按行為後法律有變更者，適用行為時之法律；但行為後之法\s?', text)

    # # debug
    # if jid == 'CHDM,101,智訴,1,20120712,1':
    #     print(start)

    if not start:
        return None
    start_idx = start.end()

    proc_head = re.search(
        r'(?m)'
        r'^\s*{UPPER_NUM_PATTERNS}、\s*程序(部[分份]|方面)'
        r'([\(\（].*?[\)\）])?'
        r'[：:]?'
        r'(\r?\n|$)'.format(UPPER_NUM_PATTERNS=UPPER_NUM_PATTERNS),
        text[start_idx:],
    )

    # # debug
    # if jid == 'CHDM,100,智重附民,1,20110322,1':
    #     print(proc_head)

    if proc_head:
        proc_head_idx = proc_head.end()
        if proc_head_idx < start_idx or jid == 'CHDM,101,智訴,1,20120712,1':
            after_proc_idx = start_idx + proc_head_idx
            next_top = re.search(
                r'(?:\r?\n\s*)?{U_L_NUM_PATTERNS}、{NEXT_TOP}[^：:\n]*[：:]?'.format(
                    U_L_NUM_PATTERNS=U_L_NUM_PATTERNS, NEXT_TOP=NEXT_TOP
                ),
                text[after_proc_idx:],
                flags=re.DOTALL,
            )
            if next_top:
                start = next_top
                start_idx = after_proc_idx + next_top.end()
            else:
                return None

    if jid in SPECIAL_CASES:
        start = start_for_special_cases(jid, text)
        start_idx = start.end() if start else None

    # # debug
    # if jid == 'CHDM,101,智訴,1,20120712,1':
    #     print(start)

    if start:
        next_line = re.search(r'(?:\r?\n[^\r\n]*){4}', text[start_idx:])
        if next_line is None:
            return 'No next line found'

        next_line_idx = next_line.end()
        end = re.search(
            r'(?m)'
            r'\s?{U_L_NUM_PATTERNS}、[\s\S]*?[。:：]\r?\n'.format(
                U_L_NUM_PATTERNS=U_L_NUM_PATTERNS
            ),
            text[start_idx + next_line_idx :],
            flags=re.DOTALL,
        )
        if end:
            end_idx = end.start()
            match = text[start_idx : start_idx + next_line_idx + end_idx].strip()
        else:
            end = re.search(
                r'(?m)^\s*{SPECIAL_PATTERNS}[\s\S]*?[。:：](\r?\n|$)'.format(
                    SPECIAL_PATTERNS=SPECIAL_PATTERNS
                ),
                text[start_idx + next_line_idx :],
                flags=re.DOTALL,
            )
            if end:
                end_idx = end.start()
                match = text[start_idx : start_idx + next_line_idx + end_idx].strip()
            else:
                return 'No end match found'

    # # debug用
    # if jid == 'CHDM,102,智簡上,1,20130510,1':
    #     print("Debug Info:")
    #     print("JID:", jid)
    #     print("Proc Head:", proc_head.group()) if proc_head else print("No Proc Head found")
    #     print("Next Top:", next_top.group()) if next_top else print("No Next Top found")
    #     print("Start", start.group())
    #     print("Next line:", next_line.group()) if next_line else print("No Next line found")
    #     print("End", end.group()) if end else print("No end match found")
    #     print("match:", match)

    if not match:
        return None

    return match

def remove_blank(df: pd.DataFrame) -> pd.DataFrame:
    """
    remove blank from fact.xlsx
    """
    for jid, fact in df.iterrows():
        fact = fact['Text']
        fact_cleaned = ' '.join(fact.split())
        fact_cleaned = re.sub(r'[(\r?\n) ]', '', fact_cleaned)
        df.at[jid, 'Text'] = fact_cleaned

        # # debug
        # if jid == 'CHDM,100,智易,1,20111026,1':
        #     print(f"Original: {fact}")
        #     print(f"Cleaned: {fact_cleaned}")

    return df
