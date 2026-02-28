import re
import json
from ckip_transformers.nlp import CkipWordSegmenter

def convert_dict_to_vocab_list(input_dict_path, output_vocab_path=None) -> str:
    with open(input_dict_path, "r", encoding="utf-8") as f:
        user_dict = json.load(f)

    vocab_list = list(user_dict.keys())

    if output_vocab_path:
        with open(output_vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_list, f, ensure_ascii=False, indent=4)

    vocab_list_sorted = sorted(vocab_list, key=len, reverse=True)
    vocab_pattern = "|".join(re.escape(word) for word in vocab_list_sorted)

    return vocab_pattern

def word_seg(text: str, ws_driver: CkipWordSegmenter, vocab_pattern: str | None = None) -> str:
    if vocab_pattern:
        escaped = [re.escape(w) for w in vocab_pattern]
        pattern = "|".join(escaped)
        text = re.sub(f"({pattern})", r" \1 ", text)

    preprocessed_text = " ".join(text.split())
    ws_result = ws_driver([preprocessed_text])

    ws_result = " ".join(ws_result[0])
    return ws_result
