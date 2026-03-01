import re
import unicodedata
from typing import Any, Literal
from collections.abc import Callable
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEXICON_DIR = PROJECT_ROOT / "lexicon_resources"

def custom_tokenizer(text: str) -> list[str]:
    PUNCS_TO_SPACE = r"[（）\(\)【】「」『』《》〈〉〔〕［］—–‐\-~·•…、，。．；：？！?!]"
    URL_PAT = r"(https?:\/\/|www\.|\w+\.\w+)"
    text = re.sub(PUNCS_TO_SPACE, " ", unicodedata.normalize("NFKC", text).lower())
    tokens = text.split()
    clean_tokens = []
    for t in tokens:
        if re.match(r"^[\W_]+$", t):
            continue
        if re.match(r"^&[A-Za-z0-9#]+;?$", t):
            continue
        if re.match(r"^\d+(\.\d+)?$", t):
            continue
        if re.match(r"^\$[\d,]+(\.\d+)?$", t):
            continue
        if re.match(r"^[^\w]*\d+[^\w]*$", t):
            continue
        if re.match(rf"^{URL_PAT}", t):
            continue
        if re.match(r"^[A-Za-z0-9]{10,}$", t):
            continue
        if not re.search(r"[A-Za-z\u4e00-\u9fff]", t):
            continue
        clean_tokens.append(t)
    return clean_tokens

def get_dtm_BoW(
    df: pd.DataFrame,
    custom_tokenizer: Callable[[str], list[str]] = custom_tokenizer,
    stop_word: list | None = None,
    delete_word: list | None = None,
    *,
    model: Literal["BoW", "TF-IDF"] = "BoW",  # 選 BoW 或 TF-IDF
    strip_phrase: bool = False,  # 移除stop words 的片語，eg: 「公司 登記」或「著作 財產 權」
) -> tuple[pd.DataFrame, Any, np.ndarray, np.ndarray, set]:
    """
    get document-term matrix using Bag-of-Words model from word_seg.xlsx
    """

    def _strip_phrases_series(series: pd.Series, phrases: list[str]) -> pd.Series:
        """
        把包含空白的片語先從文本移除（大小寫與全半形規範後）
        """
        s = series.astype(str).copy()
        for p in phrases:
            if not isinstance(p, str):
                continue
            p_norm = unicodedata.normalize("NFKC", p).lower().strip()
            if not p_norm or " " not in p_norm:
                continue
            pat = re.compile(re.escape(p_norm))
            s = s.apply(lambda x: pat.sub(" ", unicodedata.normalize("NFKC", str(x)).lower()))
        return s

    if "Word Segmentation" not in df.columns:
        raise KeyError('DataFrame 必須包含 "Word Segmentation" 欄位')

    # regularize delete_set for stop words & delete words
    def unified_normalizer(token: str) -> str:
        return unicodedata.normalize("NFKC", token).lower().strip()

    def make_analyzer(tok: Callable[[str], list[str]], banned: set[str]):
        def analyzer(doc: str) -> list[str]:
            tokens = [unified_normalizer(t) for t in tok(doc)]
            return [t for t in tokens if t not in banned]

        return analyzer

    # 建立 analyzer：段詞後過濾stop_set
    delete_set = {unified_normalizer(t) for t in (delete_word or []) if t}
    analyzer = make_analyzer(custom_tokenizer, delete_set)

    # 移除 stop words 中的片語
    texts = df["Word Segmentation"]
    if strip_phrase and delete_set:
        phrase = []
        for lst in (stop_word or []), (delete_word or []):
            for w in lst:
                if isinstance(w, str) and " " in w:
                    phrase.append(w)
        if phrase:
            texts = _strip_phrases_series(texts, phrase)

    # 建立 vectorizer
    if model == "BoW":
        vectorizer = CountVectorizer(
            analyzer=analyzer, preprocessor=None, token_pattern=None, min_df=2, max_df=0.98
        )
    elif model == "TF-IDF":
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            norm="l2",
            preprocessor=None,
            token_pattern=None,  # type: ignore
            min_df=2,
            max_df=0.98,
        )
    else:
        raise ValueError("model must be either 'BoW' or 'TF-IDF'")

    # vectorize
    dtm_sparse = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    dtm = pd.DataFrame.sparse.from_spmatrix(dtm_sparse, index=df.index, columns=vocab)

    # save files
    LEXICON_DIR.mkdir(parents=True, exist_ok=True)
    if model == "BoW":
        sp.save_npz(str(LEXICON_DIR / "dtm_csr_BoW.npz"), dtm_sparse)
    elif model == "TF-IDF":
        sp.save_npz(str(LEXICON_DIR / "dtm_csr_TF_IDF.npz"), dtm_sparse)

    # # debug
    # with open("forced_delete.txt", "w", encoding="utf-8") as f:
    #     f.write("\n".join(sorted(delete_set)))
    # with open("dtm_columns.txt", "w", encoding="utf-8") as f:
    #     f.write("\n".join(dtm.columns.tolist()))
    # residual = set(dtm.columns) & set(delete_set)
    # with open("residual_stopwords.txt", "w", encoding="utf-8") as f:
    #     f.write("\n".join(sorted(residual)))

    doc_ids = df.index.to_numpy()

    # assert that the ordinary is consistency
    assert dtm.shape[0] == dtm_sparse.shape[0], "Row count mismatch"
    assert dtm.shape[1] == dtm_sparse.shape[1], "Column count mismatch"
    assert np.array_equal(dtm.columns.to_numpy(), vocab), "Column order mismatch"
    assert np.array_equal(dtm.index.to_numpy(), df.index.to_numpy()), "Row order mismatch"
    assert all(dtm.index == doc_ids)

    return dtm, dtm_sparse, doc_ids, vocab, delete_set

def get_verdict_results(doc_ids: np.ndarray, labels: pd.DataFrame) -> pd.DataFrame:
    return labels.loc[doc_ids, ["JRESULT"]]
