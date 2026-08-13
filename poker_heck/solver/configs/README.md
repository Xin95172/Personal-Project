# 訓練設定參數

這個目錄的 JSON 是訓練範圍的唯一入口。產生器會依設定展開 packs；請不要手動編輯 `artifacts/generated/` 內的 manifest，因為下一次訓練會重新產生它。

所有金額都以 BB 表示；下注尺寸則以「當下底池比例」表示。例如 `0.75` 是 75% pot，`2.0` 是 200% pot。`all_in` 是否可用由 `include_all_in` 控制。

## 共用欄位

| 欄位 | 用途 |
| --- | --- |
| `strategy_db` | SQLite 策略資料庫的輸出位置。 |
| `output_dir` | manifest 與品質報告的位置。 |
| `manifest` | 產生後由訓練命令讀取的 pack 清單。 |
| `stack_bb` | 要遍歷的有效籌碼深度。 |
| `iterations_per_pack` | 每一個 pack 的 MCCFR 迭代數。 |
| `checkpoint_every` | 每隔多少迭代寫一次 checkpoint；必須大於 0。 |
| `checkpoint_dir` | checkpoint 位置。 |
| `export_dir` | CSV 與品質報告輸出位置。 |
| `sizing_policy.bet_sizes` | 未面對下注時允許的下注尺寸比例。 |
| `sizing_policy.raise_sizes` | 面對下注時允許的加注尺寸比例。 |
| `sizing_policy.include_all_in` | 是否把 all-in 加入可用動作。 |
| `sizing_policy.max_re_raises` | 每條街最多連續 re-raise 次數。 |

## 範圍規格 `range_spec`

```json
{ "mode": "all_combos" }
```

代表使用所有未被已知牌阻擋的兩張手牌組合。若只要牌力前幾%，使用：

```json
{ "mode": "top_percent", "percent": 15.0 }
```

`percent` 必須介於 0 與 100；值越小，訓練的起手牌範圍越窄。Heads-up 設定以 `oop_range_spec` 與 `ip_range_spec` 分別設定兩位玩家。Multiway 設定的 `range_spec` 套用到所有存活玩家，直到未來加入位置別範圍為止。

目前三份主要 grid 的預設值均為 `top_percent: 20`，用於先集中訓練較強、較常進入有意義底池的手牌。這是「縮小訓練遊戲」而非單純排程優先順序：範圍外的手牌不會得到策略。要恢復全範圍，改回 `{ "kind": "all_combos" }`，並同步將 preflop `continuation.range_profile_id` 改為 `all_combos`。

## Preflop：`preflop_solution_grid.json`

`stack_bb`、`range_spec` 與 `sizing_policy` 控制抽象；產生器會遍歷有效座位與開局／3-bet／4-bet 等合法路線。`open_sizes_bb`、`raise_sizes_bb` 若出現在設定中，則是 preflop 專用的 BB 尺寸清單；postflop 的比例尺寸不會替代它們。

可選的 `continuation` 會將非棄牌的 preflop 終端接到 multiway postflop 子局：

```json
{
  "enabled": true,
  "subgame_iterations": 100,
  "value_rollouts": 4,
  "max_cached_subgames": 500,
  "strategy_db": "../artifacts/data/multiway_strategies.sqlite3",
  "range_profile_id": "all_combos",
  "solver_version": "multiway-postflop-grid-v1",
  "bet_sizes": [0.33, 0.5, 0.75, 1.0, 1.5, 2.0],
  "raise_sizes": [0.33, 0.5, 0.75, 1.0, 1.5, 2.0],
  "include_all_in": true,
  "max_re_raises": 1
}
```

它會從該路線推導後驗 range、快取已求解的 flop 子局，再將 postflop 平均策略 EV 回傳 preflop。`subgame_iterations` 與 `value_rollouts` 越高越穩定也越慢；`max_cached_subgames` 是記憶體上限，設為 `null` 代表不限制。

## Heads-up postflop：`heads_up_solution_grid.json`

| 欄位 | 用途 |
| --- | --- |
| `streets` | 要訓練的街別，可選 `flop`、`turn`、`river`。 |
| `board_source` | `canonical` 代表用同構牌面代表；`explicit` 代表使用 `boards`。 |
| `max_canonical_boards` | 限制每條街的 canonical 牌面數；`null` 代表不限制。 |
| `pot_profiles` | 進入 postflop 時的底池與已投入籌碼情境。 |
| `oop_range_spec` / `ip_range_spec` | 兩方各自的手牌範圍。 |

## Multiway postflop：`multiway_solution_grid.json`

| 欄位 | 用途 |
| --- | --- |
| `player_counts` | 要遍歷的人數，例如 `[2, 3, 4, 5, 6, 7, 8]`。 |
| `solve_scope` | 目前僅支援 `flop_full_tree`。 |
| `preflop_route_policy` | 產生進入翻牌圈的合法 preflop 路線規則。 |
| `preflop_route_offset_per_stack` | 從每個 stack 的第幾條路線開始；用於分批。 |
| `max_routes_per_stack` | 每個 stack 最多處理幾條路線；`null` 是全部。 |
| `board_source` / `max_canonical_boards` | multiway 翻牌牌面來源與上限。 |

目前預設是每個 stack 最多 20 條 preflop 路線與 50 個 canonical flop，作為可完成的第一批訓練。確認訓練、SQLite 與 API 正常後，再逐步提高；要完整遍歷時，將兩者改為 `null`。

設定中的 `preflop_templates`、固定 ranges、固定 boards 都不應作為完整訓練的來源；產生器會根據上述規則遍歷。

## 條件子局：`conditional_subgames.json`

`subgames` 是明確指定的局面清單，適合要加強某種實戰常見狀態時使用，例如指定街別、公共牌、位置、投入額與範圍。它不是全域 grid 的替代品，也不會在空清單時產生資料。
