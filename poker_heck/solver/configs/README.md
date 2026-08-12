# 訓練設定說明

這是所有訓練設定檔的唯一參數手冊。修改 JSON 前先查本檔；程式不會自行補上未列出的訓練參數。

| 檔案 | 用途 |
|---|---|
| `preflop_solution_grid.json` | 8-Max preflop 訓練網格。 |
| `heads_up_solution_grid.json` | 單挑 flop／turn／river 訓練網格。 |
| `multiway_solution_grid.json` | 多人 flop 起始、跨街完整樹訓練網格。 |
| `conditional_subgames.json` | 已知前史的 turn／river 局部重算工作。 |

最常調整的參數：

| 欄位 | 可選設定 |
|---|---|
| `range_spec.kind` | `all_combos`、`top_percent` |
| `range_spec.percent` | `0 < percent <= 100` |
| `stack_bb` | 任意正 BB 數列，例如 `[40, 60, 100]` |
| `board_source` | `canonical`、`explicit` |
| `max_canonical_boards` | 正整數或 `null`（不限） |
| `max_preflop_routes_per_stack` | 正整數或 `null`（不限） |
| `max_raises`／`max_re_raises` | 非負整數或 `null`（不限） |
| `bet_sizes`／`raise_sizes` | 正數 pot-ratio 陣列，例如 `[0.33, 0.5, 1.0]` |
