import os
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

def build_features(df_clean_path='../artifacts/reports/fact_removed_blank.xlsx', 
                   artifacts_folder='../artifacts/reports',
                   lexicon_folder='../resources/lexicons',
                   dictionary_folder='../resources/dictionaries',
                   features_folder='../artifacts/features/dtm',
                   force_recompute_seg=False,
                   ckip_device=-1):
    """
    執行斷詞與特徵矩陣工程管線，包含：
    1. CKIP 斷詞並過濾停用詞
    2. 生成 Document-Term Matrix
    3. 取得判決結果 Labels
    """
    print("1. 準備 CKIP 斷詞模型與自定義字典...")
    dictionary_path = os.path.join(dictionary_folder, 'articut_user_defined_dict.json')
    vocab_list = convert_dict_to_vocab_list(dictionary_path)
    
    word_seg_path = os.path.join(artifacts_folder, 'word_seg.xlsx')
    
    if os.path.exists(word_seg_path) and not force_recompute_seg:
        print(f"找到已有的斷詞結果: {word_seg_path}，直接載入！")
        df_word_seg = pd.read_excel(word_seg_path)
        df_word_seg.set_index('JID', inplace=True)
    else:
        print("初始化 CkipWordSegmenter...")
        ws_driver = CkipWordSegmenter(model='albert-base', device=ckip_device)

        print("讀取乾淨的判決事實...")
        df_removed_blank = pd.read_excel(df_clean_path)
        df_removed_blank.set_index('JID', inplace=True)

        df_word_seg = df_removed_blank.copy()
        print("開始執行 CKIP 斷詞迴圈...")
        for jid, fact in tqdm(df_removed_blank.iterrows(), total=len(df_removed_blank)):
            ws_result = word_seg(fact['Text'], ws_driver, vocab_list, show_progress=False)
            df_word_seg.at[jid, 'Word Segmentation'] = str(ws_result)
        df_word_seg.to_excel(word_seg_path)
        print(f"斷詞完成，結果已儲存為 {word_seg_path}")

    print("2. 載入停用詞與刪除詞庫...")
    stop_word_path = os.path.join(lexicon_folder, 'stopwords-ch-jiebar-zht.txt')
    with open(stop_word_path, 'r', encoding='utf-8') as f:
        stop_word = f.read().splitlines()

    delete_vocab_path = os.path.join(lexicon_folder, 'delete_vocab.txt')
    with open(delete_vocab_path, 'r', encoding='utf-8') as f:
        delete_word = f.read().splitlines()

    print("載入判決標籤 (judgment_labels)...")
    labels_path = os.path.join(artifacts_folder, 'judgment_labels.xlsx')
    labels = pd.read_excel(labels_path)
    labels.set_index('JID', inplace=True)

    print("3. 生成 Document-Term Matrix (BoW, TF & TF-IDF)...")
    dtm_bow, dtm_csr_bow, doc_ids, vocab_bow, _ = get_dtm(
        df_word_seg, custom_tokenizer, stop_word, delete_word,
        strip_phrase=True, model='BoW', output_dir=features_folder
    )
    dtm_tf, dtm_csr_tf, doc_ids_tf, vocab_tf, _ = get_dtm(
        df_word_seg, custom_tokenizer, stop_word, delete_word,
        strip_phrase=True, model='TF', output_dir=features_folder
    )
    dtm_tfidf, dtm_csr_tfidf, doc_ids_tfidf, vocab_tfidf, _ = get_dtm(
        df_word_seg, custom_tokenizer, stop_word, delete_word,
        strip_phrase=True, model='TF-IDF', output_dir=features_folder
    )
    assert (doc_ids == doc_ids_tf).all(), "TF doc_ids 與 BoW doc_ids 不一致"
    assert (doc_ids == doc_ids_tfidf).all(), "TF-IDF doc_ids 與 BoW doc_ids 不一致"

    print("取得並儲存 verdict 判決結果陣列...")
    verdict_results = get_verdict_results(doc_ids, labels)
    verdict_out = os.path.join(artifacts_folder, 'verdict_results.xlsx')
    verdict_results.to_excel(verdict_out)

    print("特徵萃取完成！特徵結果與標籤集已儲存/更新。")
    return dtm_csr_bow, dtm_csr_tf, dtm_csr_tfidf

if __name__ == '__main__':
    build_features()
