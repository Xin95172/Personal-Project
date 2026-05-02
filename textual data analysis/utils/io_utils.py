import os
import json
from tqdm.notebook import tqdm

def get_ip_data(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    keywords = ["智慧財產", "專利", "商標", "著作權", "營業秘密"]

    matched_files = []

    for file in tqdm(os.listdir(input_folder)):
        if file.endswith('.json'):
            file_path = os.path.join(input_folder, file)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                jfull = data.get('JFULL', '')
                jcase = data.get('JCASE', '')
                jtitle = data.get('JTITLE', '')

                jfull = jfull if isinstance(jfull, str) else ''
                jcase = jcase if isinstance(jcase, str) else ''
                jtitle = jtitle if isinstance(jtitle, str) else ''

                if (
                    any(kw in jcase for kw in keywords)
                    or any(kw in jtitle for kw in keywords)
                    or any(kw in jfull for kw in keywords)
                ):

                    jfull_cleaned = ' '.join(jfull.split())
                    jfull_cleaned = jfull_cleaned.replace('\r\n', '')
                    jfull_cleaned = jfull_cleaned.replace('\n', '')
                    jfull_cleaned = jfull_cleaned.replace(' ', '')

                    data['JFULL'] = jfull_cleaned
                    new_path = os.path.join(output_folder, file)

                    with open(new_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)

                    matched_files.append(file_path)

            except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
                print(f"錯誤（{type(e).__name__}）：{file_path}")
