import os
import re
import json
import shutil
import requests
import subprocess
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def clean_filename(filename):
    """移除檔名中的非法字元"""
    return re.sub(r'[\\/:*?"<>|]', '', filename)

def generate_date_range(start_year, start_month, end_year, end_month):
    """生成 YYYY-MM 格式的日期列表"""
    dates = []
    current_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 1)
    
    while current_date <= end_date:
        year_month = current_date.strftime('%Y-%m')
        dates.append(year_month)

        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)
    
    return dates

def _process_single_json(file_info):
    """
    多進程執行的輔助函式：處理單一 JSON 檔案的讀取與關鍵字檢查
    :param file_info: (file_path, output_dir, keyword)
    :return: True if matched, False otherwise
    """
    file_path, output_dir, keyword = file_info
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 檢查內容是否符合關鍵字 (通常在 JFULL 欄位)
        if 'JFULL' in data and keyword in data['JFULL']:
            f_name = os.path.basename(file_path)
            new_path = os.path.join(output_dir, f_name)
            shutil.copy2(file_path, new_path)
            return True

        # 處理完不刪除，由主進程在確保所有子進程結束後統一清理或逐步清理
        return False
    except:
        return False

class JudicialLoader:
    def __init__(self, headers):
        """
        初始化下載器
        :param headers: 包含 User-Agent 與 authorization Bearer token 的字典
        """
        self.headers = headers

    def download_rar_files(self, url_range, year_months, download_folder='../data/rar_files'):
        """
        從司法院 Open Data API 下載 RAR 資源，具備進度條顯示
        """
        os.makedirs(download_folder, exist_ok=True)
        
        print(f"--- 開始下載原始 RAR 資源 (共 {len(url_range)} 個月份) ---")
        for n in range(min(len(url_range), len(year_months))):
            url = f'https://opendata.judicial.gov.tw/api/FilesetLists/{url_range[n]}/file'
            file_name = f'{year_months[n]}裁判書.rar'
            file_path = os.path.join(download_folder, file_name)

            if os.path.exists(file_path):
                print(f"Skipping: {file_name} (Already exists)")
                continue

            try:
                response = requests.get(url, headers=self.headers, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(file_path, 'wb') as f, tqdm(
                    desc=f"Downloading {file_name}",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=32768):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))

            except requests.exceptions.RequestException as e:
                print(f"\n{file_name} 下載失敗，錯誤：{e}")

    def extract_and_filter_json(self, 
                               rar_dir='../data/rar_files', 
                               unrar_base_dir='../data/unrar_files', 
                               output_dir='../data/raw_json', 
                               keyword='智慧財產',
                               seven_zip_path=r'C:\Program Files\7-Zip\7z.exe',
                               cleanup_rar=True,
                               max_workers=None):
        """
        解壓縮 RAR 並使用多進程平行過濾包含關鍵字的 JSON
        """
        os.makedirs(unrar_base_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        rar_files = [f for f in os.listdir(rar_dir) if f.endswith('.rar')]
        
        print(f"\n--- 開始解析與過濾 JSON (總共 {len(rar_files)} 個 RAR) ---")
        for file in rar_files:
            rar_file = os.path.join(rar_dir, file)
            print(f"\n[月份處理] {file}")

            try:
                # 1. 執行解壓縮
                subprocess.run(
                    [seven_zip_path, 'x', rar_file, f'-o{unrar_base_dir}', '-y'],
                    check=True, stdout=subprocess.DEVNULL
                )
            except Exception as e:
                print(f"  解壓縮失敗: {e}")
                continue

            # 2. 收集所有解壓後的 JSON 檔案路徑
            json_files = []
            for root, _, files in os.walk(unrar_base_dir):
                for f_name in files:
                    if f_name.endswith('.json'):
                        json_files.append((os.path.join(root, f_name), output_dir, keyword))

            # 3. 使用多進程進行平行搜尋
            matched_count = 0
            if json_files:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # 使用 list 封裝以觸發進度條
                    futures = [executor.submit(_process_single_json, info) for info in json_files]
                    for future in tqdm(as_completed(futures), total=len(json_files), desc=f"  Searching '{keyword}'", leave=False):
                        if future.result():
                            matched_count += 1
            
            # 4. 清理暫存解壓目錄內容
            # 為了效能，我們直接整批刪除剛剛解壓出來的內容，而不是一個個刪
            for root, dirs, files in os.walk(unrar_base_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))

            print(f"  處理完成: 找到 {matched_count} 筆符合檔案")
            
            # 5. 自動清理原始 RAR
            if cleanup_rar and os.path.exists(rar_file):
                os.remove(rar_file)

    def run_all(self, url_range, year_months, **kwargs):
        """一鍵式完整流程"""
        self.download_rar_files(
            url_range, year_months, 
            download_folder=kwargs.get('rar_dir', '../data/rar_files')
        )
        self.extract_and_filter_json(
            rar_dir=kwargs.get('rar_dir', '../data/rar_files'),
            unrar_base_dir=kwargs.get('unrar_dir', '../data/unrar_files'),
            output_dir=kwargs.get('output_dir', '../data/raw_json'),
            keyword=kwargs.get('keyword', '智慧財產'),
            seven_zip_path=kwargs.get('7z_path', r'C:\Program Files\7-Zip\7z.exe'),
            cleanup_rar=kwargs.get('cleanup_rar', True),
            max_workers=kwargs.get('max_workers', None)
        )
