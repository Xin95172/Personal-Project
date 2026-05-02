# 專案資料夾分類

這份文件記錄各資料夾用途，避免清理檔案時誤刪 pipeline 需要的資料。

## 核心程式

- `src/`: 資料下載與原始資料載入。
- `utils/`: 前處理、斷詞、DTM、模型流程的 pipeline 工具。
- `algos/`: 模型與評估演算法實作。
- `config/`: 案件類型、勝敗判斷、例外規則等 pattern 設定。
- `notebooks/`: 主要執行入口，目前主檔是 `main.ipynb`。

## 資料與產物

- `data/raw_parquet/`: Step 1 可用的 parquet 輸入來源。`raw_extracted.parquet` 與 `full_text_backup.parquet` 不應隨意刪除。
- `data/raw_json/`: 若從原始 JSON cold start，預設會從這裡讀取。
- `data/processed/`: 前處理後但不屬於報表的資料輸出。
- `resources/dictionaries/`: Step 2 斷詞需要的使用者字典。
- `resources/lexicons/`: Step 2 使用的停用詞與刪除詞。
- `artifacts/cache/`: Step 1 產生的快取。`PARQUET_FOLDER = None` 時會從這裡讀取。
- `artifacts/reports/`: pipeline 輸出的報表與中間資料，例如 `judgment_labels.xlsx`、`fact_removed_blank.xlsx`、`verdict_results.xlsx`。
- `artifacts/features/dtm/`: Step 2 產生的 BoW、TF、TF-IDF 稀疏矩陣。
- `artifacts/features/`: MNIR 或其他模型特徵輸出。
- `artifacts/models/`: 訓練完成的模型檔。

## 可清理項目

以下檔案可安全刪除，程式會自動重建或不需要保留：

- `__pycache__/`
- `*.pyc`
- `.ipynb_checkpoints/`
- `~$*.xlsx`
- `~$*.docx`

## 清理前要確認

- `artifacts/cache/*.parquet`: 是快取，可重建；但刪除後若 `PARQUET_FOLDER = None`，Step 1 會找不到快取，需改用 `PARQUET_FOLDER = "../data/raw_parquet"` 重跑。
- `artifacts/features/dtm/dtm_csr_*.npz`: 是 Step 2 產物；刪除後需重跑 Step 2。
- `artifacts/features/*.npy`: 是 Step 3/MNIR 產物；刪除後需重跑模型流程。
- `artifacts/models/*.pt`: 是訓練好的模型；除非確定不需要該模型，否則保留。
