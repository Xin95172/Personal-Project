# 操作與資料流

## 資料如何產生

```text
configs/*.json
  → generators（展開 stack、位置、範圍、牌面、路線）
  → manifest（待訓練的 packs）
  → MCCFR trainer（每個 pack 反覆迭代）
  → checkpoint / 品質報告 / SQLite strategy database
  → FastAPI 依精確資訊集查詢策略
```

「完整」是指完整遍歷設定檔所描述的離散抽象，不是窮盡真實無限注德州撲克的所有連續下注尺寸、所有手牌範圍與所有牌局歷史。增加抽象粒度會增加訓練量與資料庫大小。

## 日常操作

安裝：

```powershell
python -m pip install -e ".[dev]"
```

建立並訓練指定設定：

```powershell
poker-train configs\preflop_solution_grid.json
poker-train configs\heads_up_solution_grid.json
poker-train configs\multiway_solution_grid.json
```

`poker-train` 是日常指令，會自動處理 generator 與 manifest。`poker-build-db` 的輸入才是 manifest，僅供進階分批訓練。只想產生並檢查 pack，使用 `poker-train <設定檔> --generate-only`。建議第一次使用先在設定檔限制 `max_canonical_boards` 或 `max_preflop_routes_per_stack`，確認資料庫與 checkpoint 都可寫入後，再改回 `null` 跑完整設定。若中途停止，保留 `artifacts/`；下次執行同一個設定時，程式會載入既有 checkpoint 並再訓練設定的迭代數。

條件子局的設定與訓練：

```powershell
python -m poker_solver.generators.conditional_subgames configs\conditional_subgames.json
python -m poker_solver.cli.build_strategy_db artifacts\generated\conditional_subgames_manifest.json
```

`conditional_subgames.json` 預設是空清單；先新增一個明確的子局規格才會產生訓練包。

## API

啟動服務：

```powershell
poker-serve artifacts\checkpoints\river.pkl --strategy-db artifacts\data\river_strategies.sqlite3
```

服務啟動後，至 `http://127.0.0.1:8000/docs` 查看每個欄位與範例。通用端點是 `POST /v1/decision`：它只會回傳策略庫中完全相符的資訊集；若查不到，會回傳 404，而不是假裝提供 GTO 答案。舊版 `/v1/river/strategy` 端點仍可用於載入的 river checkpoint。

每個匯出的動作也包含下列訓練信賴統計：`samples`、`ev_mean`、`ev_stddev`、`ev_stderr`、`ci95_low`、`ci95_high`。這些是 solver traversal 中收集的 action-value 統計；它們用來判斷抽樣誤差，不能單獨證明整個策略已達 GTO 收斂。若兩個動作的 95% 區間大幅重疊，應增加相關子局的訓練量，而不是只依平均 EV 排名。

## 檢查與清理

```powershell
python -m pytest
python -m pytest --cov=poker_solver --cov-report=term-missing
poker-clean --cache --empty
```

`poker-clean --cache --empty` 只移除快取、coverage、egg-info 與空舊目錄。`poker-clean --all` 會移除全部可再生訓練產物；這包含策略資料庫與 checkpoint。
