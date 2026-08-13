# Poker Heck Solver

這是一個以 MCCFR（Monte Carlo Counterfactual Regret Minimization）訓練的德州撲克策略資料庫專案。它不是「精確求解整個 8-Max 無限注德州撲克」的引擎；而是在明確定義的牌局、範圍、牌面與下注尺寸抽象下，逐個子局逼近納許均衡策略，並將結果放入 SQLite 供 API 即時查詢。

目前涵蓋 preflop、heads-up postflop，以及 2 至 8 人的 multiway postflop 抽象。每個訓練包都由設定檔與產生器建立，因此訓練範圍可擴充，不依賴手寫 demo 牌局。

## 快速開始

在此目錄安裝開發環境：

```powershell
python -m pip install -e ".[dev]"
```

先從設定檔產生訓練包，再將 manifest 交給訓練器：

```powershell
poker-train configs\preflop_solution_grid.json
poker-train configs\heads_up_solution_grid.json
poker-train configs\multiway_solution_grid.json
```

每個命令都會自動產生 manifest，再依其順序逐包訓練。只想檢查會產生哪些 pack 時，加上 `--generate-only`。先只做小規模驗證時，請暫時在設定檔設定 `max_canonical_boards` 或 `max_preflop_routes_per_stack`，完成驗證後再改回 `null`。訓練會將 checkpoint、品質報告與策略寫到 `artifacts/`；此目錄是可再生輸出，不納入版控。

啟動查詢 API：

```powershell
poker-serve artifacts\checkpoints\river.pkl --strategy-db artifacts\data\river_strategies.sqlite3
```

開啟 `http://127.0.0.1:8000/docs`，可直接從 Swagger 測試 `/health`、`/v1/decision` 與舊版 river 查詢端點。

## 專案結構

```text
configs/                 可修改的訓練範圍與抽象設定
docs/                    操作與資料流說明
src/poker_solver/
  api/                   FastAPI 服務
  cli/                   訓練、建庫、啟動與清理命令
  engine/                遊戲狀態、牌局規則、底池與攤牌
  generators/            從設定產生完整訓練包
  solver_core/           MCCFR、範圍、checkpoint、SQLite 儲存
tests/                   單元、整合與端對端測試
artifacts/               訓練結果；可刪除並重新產生
```

## 文件入口

- [操作與資料流](docs/OPERATIONS.md)
- [所有可調整參數](configs/README.md)

## 驗證與清理

```powershell
python -m pytest
python -m pytest --cov=poker_solver --cov-report=term-missing
poker-clean --cache --empty
```

若要刪除訓練產物，使用 `poker-clean --all`。它會刪除整個 `artifacts/`，包含 checkpoint 與 SQLite 策略庫，請只在確定不再需要它們時執行。
