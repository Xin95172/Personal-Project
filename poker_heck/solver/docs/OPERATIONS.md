# 操作手冊

## 安裝

```powershell
python -m pip install -e ".[dev]"
```

## 產生訓練工作並建立策略庫

以下三份網格是專案唯一保留的訓練入口。它們描述要遍歷的有效籌碼、位置、前序動作、範圍設定與下注尺寸；產生器會建立 pack 與 manifest，再由資料庫建立器逐一訓練。

```powershell
# 8-Max preflop
poker-generate-preflop-packs configs\preflop_solution_grid.json
poker-build-db artifacts\generated\preflop_manifest.json

# heads-up postflop（flop、turn、river）
poker-generate-heads-up-packs configs\heads_up_solution_grid.json
poker-build-db artifacts\generated\heads_up_manifest.json

# 8-Max multiway postflop（flop）
poker-generate-packs configs\multiway_solution_grid.json
poker-build-db artifacts\generated\multiway_manifest.json
```

`poker-build-db` 會依 manifest 的優先順序連續執行工作；每個工作依設定週期寫入 checkpoint，完成後將平均策略存入 SQLite。三份網格的 `board_source` 皆使用 `canonical`：它會遍歷花色同構後的代表牌面，而不是固定 demo 牌面。完整網格的工作量很大，適合長時間執行。

### 限制 preflop 起手牌範圍

`preflop_solution_grid.json` 的 `range_profiles` 可使用 `top_percent`，由程式依可重現的 preflop 牌力排序選取前段 combo；不需要列出特定花色牌。例如：

```json
"position_aware": {
  "kind": "top_percent",
  "percent": 25,
  "percent_by_position": {"utg": 15, "mp": 20, "co": 30, "btn": 45, "sb": 40, "bb": 50}
}
```

未列出的位置使用 `percent` 的預設值。`all_combos` 則是每個位置完整的 1,326 combos。這個百分比是 range abstraction，不是已求得的 GTO opening range；若要採用特定資料來源的 range，下一步可加入 hand-class／權重表。

## 啟動 API

```powershell
poker-serve artifacts\checkpoints\river.pkl
```

打開 `http://127.0.0.1:8000/docs` 可用 Swagger 測試 API。

## 測試

```powershell
python -m pytest --cov=poker_solver --cov-branch -q
```

## 清理可再生資料

```powershell
# 只刪除產生的 pack 與 manifest；不會刪除 checkpoint 或 SQLite 策略庫
poker-clean --generated

# 清除快取與空資料夾
poker-clean --cache --empty
```
