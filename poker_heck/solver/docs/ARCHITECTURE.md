# 專案架構

```text
configs → generators → pack／manifest → MCCFR → checkpoint／SQLite → API
```

| 位置 | 用途 |
|---|---|
| `engine/` | 撲克規則、合法行動、底池與攤牌。 |
| `solver_core/` | MCCFR、range spec、checkpoint、策略庫。 |
| `generators/` | 依 configs 展開訓練工作。 |
| `configs/` | 正式訓練規格。 |
| `artifacts/` | 可再生 pack、checkpoint、SQLite 與匯出。 |

## 街道策略

preflop 是獨立訓練範圍。postflop 不是三個互不相干的遊戲：flop pack 必須一路 traverses 到 turn 與 river，才能估計 flop 行動的未來價值。

策略庫仍會把訓練到的資訊集依 `flop`、`turn`、`river` 分街存放，API 因此能按目前街道查詢。turn／river 的額外求解使用 conditional subgame：輸入必須帶前序公共牌、行動 history、有效籌碼與條件 range，而非脫離 flop 獨立訓練。

## Range 與行動 abstraction

所有 solver 使用宣告式 `range_spec`：`all_combos` 或 `top_percent`。下注是有限的 pot-ratio action abstraction；它能完整展開所選尺寸的合法樹，但不是連續任意下注尺寸下的精確 GTO。
