import os
import re
import pandas as pd
from tqdm import tqdm

try:
    from ckip_transformers.nlp import CkipWordSegmenter
except ImportError:
    pass # Expected to be available in the user's environment

from utils import (
    convert_dict_to_vocab_list,
    custom_tokenizer,
    get_dtm,
    get_verdict_results,
    word_seg,
)
from utils.leakage_utils import remove_leakage_from_series

def build_features(df_clean_path='../artifacts/reports/fact_removed_blank.xlsx', 
                   artifacts_folder='../artifacts/reports',
                   lexicon_folder='../resources/lexicons',
                   dictionary_folder='../resources/dictionaries',
                   features_folder='../artifacts/features/dtm',
                   force_recompute_seg=False,
                   ckip_device=0,
                   target_jtypes=None,
                   target_verdicts=None,
                   dataset_name=None,
                   remove_leakage=False,
                   leakage_terms_path=None):
    """
    執行斷詞與特徵矩陣工程管線，包含：
    1. CKIP 斷詞並過濾停用詞
    2. 生成 Document-Term Matrix
    3. 取得判決結果 Labels
    """
    def _as_list(value):
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        return list(value)

    def _safe_slug(value):
        value = re.sub(r'[^A-Za-z0-9_.-]+', '_', value.strip().lower())
        return value.strip('_') or 'subset'

    target_jtypes = _as_list(target_jtypes)
    target_verdicts = _as_list(target_verdicts)

    if dataset_name is None and (target_jtypes or target_verdicts):
        parts = []
        if target_jtypes:
            parts.append('jtype_' + '_'.join(_safe_slug(v) for v in target_jtypes))
        if target_verdicts:
            parts.append('verdict_' + '_'.join(_safe_slug(v) for v in target_verdicts))
        dataset_name = '__'.join(parts)

    leakage_variant = 'no_leakage' if remove_leakage else 'with_leakage'
    dataset_slug = _safe_slug(dataset_name) if dataset_name else None
    if dataset_slug:
        dataset_features_folder = os.path.join(features_folder, dataset_slug, leakage_variant)
        dataset_artifacts_folder = os.path.join(artifacts_folder, dataset_slug, leakage_variant)
    elif remove_leakage:
        dataset_features_folder = os.path.join(features_folder, leakage_variant)
        dataset_artifacts_folder = os.path.join(artifacts_folder, leakage_variant)
    else:
        dataset_features_folder = features_folder
        dataset_artifacts_folder = artifacts_folder
    os.makedirs(dataset_artifacts_folder, exist_ok=True)

    print("1. 載入判決事實與標籤，建立 Step 2 資料集...")
    df_removed_blank = pd.read_excel(df_clean_path)
    labels_path = os.path.join(artifacts_folder, 'judgment_labels.xlsx')
    labels_all = pd.read_excel(labels_path)

    label_cols = ['JID', 'JTYPE', 'VERDICT']
    missing_cols = [col for col in label_cols if col not in labels_all.columns]
    if missing_cols:
        raise KeyError(f"judgment_labels 缺少必要欄位: {missing_cols}")

    df_meta = df_removed_blank.merge(labels_all[label_cols], on='JID', how='inner')
    if target_jtypes:
        df_meta = df_meta[df_meta['JTYPE'].isin(target_jtypes)]
    if target_verdicts:
        df_meta = df_meta[df_meta['VERDICT'].isin(target_verdicts)]
    if df_meta.empty:
        raise ValueError(
            f"篩選後沒有資料。target_jtypes={target_jtypes}, "
            f"target_verdicts={target_verdicts}"
        )

    print(f"資料集名稱: {dataset_name or 'all'}")
    print(f"Leakage variant: {leakage_variant}")
    print(f"資料筆數: {len(df_meta)}")
    print("JTYPE 分布:")
    print(df_meta['JTYPE'].value_counts().to_string())
    print("VERDICT 分布:")
    print(df_meta['VERDICT'].value_counts().to_string())

    df_removed_blank = df_meta[['JID', 'Text']].set_index('JID')
    labels = labels_all.set_index('JID')

    if remove_leakage:
        if leakage_terms_path is None:
            leakage_terms_path = os.path.join(lexicon_folder, 'leakage_terms.csv')
        print(f"套用 leakage 清理: {leakage_terms_path}")
        df_removed_blank['Text'] = remove_leakage_from_series(
            df_removed_blank['Text'], leakage_terms_path
        )
        cleaned_text_path = os.path.join(dataset_artifacts_folder, 'fact_removed_blank.xlsx')
        df_removed_blank.to_excel(cleaned_text_path)
        print(f"leakage-cleaned text 已儲存為 {cleaned_text_path}")

    print("2. 準備 CKIP 斷詞模型與自定義字典...")
    dictionary_path = os.path.join(dictionary_folder, 'articut_user_defined_dict.json')
    vocab_list = convert_dict_to_vocab_list(dictionary_path)
    
    word_seg_path = os.path.join(dataset_artifacts_folder, 'word_seg.xlsx')
    legacy_word_seg_path = os.path.join(artifacts_folder, 'word_seg_with_leakage.xlsx')
    old_legacy_word_seg_path = os.path.join(artifacts_folder, 'word_seg.xlsx')
    
    if os.path.exists(word_seg_path) and not force_recompute_seg:
        print(f"找到已有的斷詞結果: {word_seg_path}，直接載入！")
        df_word_seg = pd.read_excel(word_seg_path)
        df_word_seg.set_index('JID', inplace=True)
    elif (
        not dataset_name
        and not remove_leakage
        and os.path.exists(old_legacy_word_seg_path)
        and not force_recompute_seg
    ):
        print(f"找到舊版全資料斷詞結果: {old_legacy_word_seg_path}，轉存為 with_leakage cache。")
        df_word_seg = pd.read_excel(old_legacy_word_seg_path)
        df_word_seg.set_index('JID', inplace=True)
        df_word_seg = df_word_seg.loc[df_removed_blank.index]
        df_word_seg.to_excel(word_seg_path)
        print(f"全資料 with_leakage 斷詞結果已儲存為 {word_seg_path}")
    elif (
        dataset_name
        and not remove_leakage
        and os.path.exists(legacy_word_seg_path)
        and not force_recompute_seg
    ):
        print(f"找到全資料斷詞結果: {legacy_word_seg_path}，篩選為目前資料集後另存。")
        df_word_seg = pd.read_excel(legacy_word_seg_path)
        df_word_seg.set_index('JID', inplace=True)
        df_word_seg = df_word_seg.loc[df_removed_blank.index]
        df_word_seg.to_excel(word_seg_path)
        print(f"子資料集斷詞結果已儲存為 {word_seg_path}")
    elif (
        dataset_name
        and not remove_leakage
        and os.path.exists(old_legacy_word_seg_path)
        and not force_recompute_seg
    ):
        print(f"找到舊版全資料斷詞結果: {old_legacy_word_seg_path}，篩選為目前資料集後另存。")
        df_word_seg = pd.read_excel(old_legacy_word_seg_path)
        df_word_seg.set_index('JID', inplace=True)
        df_word_seg = df_word_seg.loc[df_removed_blank.index]
        df_word_seg.to_excel(word_seg_path)
        print(f"子資料集斷詞結果已儲存為 {word_seg_path}")
    else:
        print("初始化 CkipWordSegmenter...")
        ws_driver = CkipWordSegmenter(model='albert-base', device=ckip_device)

        df_word_seg = df_removed_blank.copy()
        print("開始執行 CKIP 斷詞迴圈...")
        for jid, fact in tqdm(df_removed_blank.iterrows(), total=len(df_removed_blank)):
            ws_result = word_seg(fact['Text'], ws_driver, vocab_list, show_progress=False)
            df_word_seg.at[jid, 'Word Segmentation'] = str(ws_result)
        df_word_seg.to_excel(word_seg_path)
        print(f"斷詞完成，結果已儲存為 {word_seg_path}")

    print("3. 載入停用詞與刪除詞庫...")
    stop_word_path = os.path.join(lexicon_folder, 'stopwords-ch-jiebar-zht.txt')
    with open(stop_word_path, 'r', encoding='utf-8') as f:
        stop_word = f.read().splitlines()

    delete_vocab_path = os.path.join(lexicon_folder, 'delete_vocab.txt')
    with open(delete_vocab_path, 'r', encoding='utf-8') as f:
        delete_word = f.read().splitlines()

    print("4. 生成 Document-Term Matrix (BoW, TF & TF-IDF)...")
    dtm_bow, dtm_csr_bow, doc_ids, vocab_bow, _ = get_dtm(
        df_word_seg, custom_tokenizer, stop_word, delete_word,
        strip_phrase=True, model='BoW', output_dir=dataset_features_folder
    )
    dtm_tf, dtm_csr_tf, doc_ids_tf, vocab_tf, _ = get_dtm(
        df_word_seg, custom_tokenizer, stop_word, delete_word,
        strip_phrase=True, model='TF', output_dir=dataset_features_folder
    )
    dtm_tfidf, dtm_csr_tfidf, doc_ids_tfidf, vocab_tfidf, _ = get_dtm(
        df_word_seg, custom_tokenizer, stop_word, delete_word,
        strip_phrase=True, model='TF-IDF', output_dir=dataset_features_folder
    )
    assert (doc_ids == doc_ids_tf).all(), "TF doc_ids 與 BoW doc_ids 不一致"
    assert (doc_ids == doc_ids_tfidf).all(), "TF-IDF doc_ids 與 BoW doc_ids 不一致"

    print("取得並儲存 verdict 判決結果陣列...")
    verdict_results = get_verdict_results(doc_ids, labels)
    verdict_out = os.path.join(dataset_artifacts_folder, 'verdict_results.xlsx')
    verdict_results.to_excel(verdict_out)

    doc_ids_out = os.path.join(dataset_artifacts_folder, 'doc_ids.csv')
    df_meta.set_index('JID').loc[doc_ids, ['JTYPE', 'VERDICT']].to_csv(
        doc_ids_out, encoding='utf-8-sig'
    )

    print("特徵萃取完成！特徵結果與標籤集已儲存/更新。")
    return dtm_csr_bow, dtm_csr_tf, dtm_csr_tfidf

if __name__ == '__main__':
    build_features()
