import re
import json

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

def word_seg(text: str, ws_driver: 'Any', vocab_pattern: str | None = None) -> str:
    # vocab_pattern 已是由 convert_dict_to_vocab_list 回傳的完整 regex pattern 字串
    # 直接套用即可，不需再逐字元 escape
    if vocab_pattern:
        text = re.sub(f"({vocab_pattern})", r" \1 ", text)

    preprocessed_text = " ".join(text.split())
    ws_result = ws_driver([preprocessed_text])

    ws_result = " ".join(ws_result[0])
    return ws_result
