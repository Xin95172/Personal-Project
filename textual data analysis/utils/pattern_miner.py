import os
import json
import csv
import re
from collections import Counter

def mine_patterns(input_csv, json_folder, output_json, max_samples_per_type=200):
    mismatches = []
    try:
        with open(input_csv, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mismatches.append(row)
    except Exception as e:
        print(f"Read CSV Failed: {e}")
        return

    # Group by mismatch type
    grouped = {}
    for m in mismatches:
        key = f"{m.get('JTYPE')}_{m.get('VERDICT')}_TO_{m.get('own_verdict')}"
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(m.get('JID'))

    print(f"Found {len(grouped)} distinct mismatch categories.")
    results = {}
    total_processed = 0
    
    count = 0
    for key, jids in sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True):
        if len(jids) < 5 or count >= 10: 
            break # Skip rare outliers or stop if beyond top 10 categories
            
        print(f"Mining top {count+1} category [{key}] with {len(jids)} cases...")
        count += 1
        texts = []
        for jid in jids[:100]: # max 100 samples per class
            fpath = os.path.join(json_folder, f"{jid}.json")
            if os.path.exists(fpath):
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        text = data.get("JFULL", "")
                        match = re.search(r'主\s*文\s*(.*?)(事實及理由|理\s*由|中\s*華\s*民\s*國|中華民國|\n\n)', text, re.DOTALL)
                        mc = match.group(1) if match else text[:300]
                        mc = re.sub(r'[^\w\s]', '', mc)
                        mc = re.sub(r'\s+', '', mc)
                        if mc:
                            texts.append(mc)
                            total_processed += 1
                except:
                    pass

        # Extract strict NLP n-grams (3 to 6 characters)
        ngrams = []
        for text in texts:
            for n in [3, 4, 5, 6]:
                for i in range(len(text)-n+1):
                    token = text[i:i+n]
                    if not re.match(r'^[0-9一二三四五六七八九十萬元年月日]+$', token):
                        ngrams.append(token)

        counter = Counter(ngrams)
        results[key] = {
            "total_samples": len(jids),
            "analyzed": len(texts),
            "top_patterns": [word for word, count in counter.most_common(20)]
        }
        
        # Incremental save
        out_folder = os.path.dirname(output_json)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Pattern mining complete! Analyzed {total_processed} cases.")
    print(f"Results saved to {output_json}")

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "reports", "verdict_mismatches_minimal.csv")
    json_path = os.path.join(os.path.dirname(__file__), "..", "data", "IP_Law_cases")
    out_path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "reports", "pattern_clues.json")
    
    mine_patterns(csv_path, json_path, out_path)
