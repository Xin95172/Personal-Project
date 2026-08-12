# Poker Heck Solver

這是一個用 MCCFR 建立 8-Max No-Limit Texas Hold'em 近似策略庫的專案。它不是精確閉式 GTO solver：策略品質取決於遊戲樹、range、下注尺寸 abstraction 與訓練迭代數。

## 專案入口

```powershell
python -m pip install -e ".[dev]"

# 產生與訓練 preflop、heads-up postflop、multiway postflop 網格
poker-generate-preflop-packs configs\preflop_solution_grid.json
poker-build-db artifacts\generated\preflop_manifest.json
```

其餘訓練指令、API 與清理方式請看 [操作手冊](docs/OPERATIONS.md)。

## 目錄

| 位置 | 用途 |
|---|---|
| `src/poker_solver/` | 引擎、MCCFR 核心、API 與 CLI。 |
| `configs/` | 正式的訓練參數網格。 |
| `artifacts/` | checkpoint、資料庫與產生的 pack；不納入 Git。 |
| `tests/` | unit 與 integration 測試。 |
| `docs/` | 操作與架構說明。 |

## 核心概念

- preflop 可使用完整 1,326 combo 空間，或依位置選擇前幾%牌力範圍。
- postflop 使用花色同構化的 canonical board，避免重複訓練等價牌面。
- 下注尺寸採 pot-ratio abstraction：33%、50%、75%、100%、150%、200% 與 all-in。
- API 只回傳已訓練／已存入策略庫的資訊集；查不到時應重新求解或回傳找不到，而不是冒充精確策略。
