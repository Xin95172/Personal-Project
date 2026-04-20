import os
import json
import pandas as pd
from tqdm import tqdm
import multiprocessing

def _get_full_text(jid_args):
    jid, input_folder = jid_args
    file_path = os.path.join(input_folder, f"{jid}.json")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {"JID": jid, "JFULL": data.get("JFULL", "")}
    except:
        return None

def main():
    input_folder = os.path.abspath("./data/raw_json")
    output_path = os.path.abspath("./artifacts/cache/full_text_backup.parquet")
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"📦 啟動獨立備份任務：從 {input_folder} 提取所有全文...")
    files = [f.replace(".json", "") for f in os.listdir(input_folder) if f.endswith(".json")]
    items = [(jid, input_folder) for jid in files]
    
    results = []
    with multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count() - 1)) as pool:
        for res in tqdm(pool.imap_unordered(_get_full_text, items, chunksize=100), 
                       total=len(items), desc="Full Text Backup"):
            if res:
                results.append(res)
    
    print(f"💾 正在儲存全文備份至 {output_path}...")
    df = pd.DataFrame(results)
    df.to_parquet(output_path, index=False, compression="snappy")
    print("✅ 備份完成！這份檔案之後可以獨立讀取，且不會拖慢您的主管線速度。")

if __name__ == "__main__":
    main()
