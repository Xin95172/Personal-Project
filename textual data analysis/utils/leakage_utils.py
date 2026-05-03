import re
from pathlib import Path

import pandas as pd


def load_leakage_terms(leakage_terms_path: str | Path) -> pd.DataFrame:
    terms = pd.read_csv(leakage_terms_path, encoding='utf-8-sig')
    required = {'term', 'pattern_type', 'action', 'enabled'}
    missing = required - set(terms.columns)
    if missing:
        raise KeyError(f"leakage_terms 缺少必要欄位: {sorted(missing)}")

    enabled = terms['enabled']
    if enabled.dtype != bool:
        enabled = enabled.astype(str).str.lower().isin({'true', '1', 'yes', 'y'})
    terms = terms[enabled].copy()
    terms['term'] = terms['term'].fillna('').astype(str)
    terms = terms[terms['term'].str.len() > 0]
    terms = terms.assign(_term_len=terms['term'].str.len()).sort_values('_term_len', ascending=False)
    terms = terms.drop(columns=['_term_len'])
    return terms


def remove_leakage_from_text(text: str, leakage_terms: pd.DataFrame) -> str:
    if not isinstance(text, str):
        return ''

    cleaned = text
    for _, row in leakage_terms.iterrows():
        term = row['term']
        pattern_type = str(row.get('pattern_type', 'literal')).lower()
        action = str(row.get('action', 'remove')).lower()

        if action == 'review':
            continue

        if pattern_type == 'regex':
            pattern = term
        else:
            pattern = re.escape(term)

        if action == 'remove_section':
            cleaned = re.sub(pattern + r'.*?(?=(?:[一二三四五六七八九十]+、|[0-9]+[.、]|$))', ' ', cleaned)
        else:
            cleaned = re.sub(pattern, ' ', cleaned)

    return ' '.join(cleaned.split())


def remove_leakage_from_series(text_series: pd.Series, leakage_terms_path: str | Path) -> pd.Series:
    leakage_terms = load_leakage_terms(leakage_terms_path)
    return text_series.fillna('').astype(str).apply(
        lambda text: remove_leakage_from_text(text, leakage_terms)
    )
