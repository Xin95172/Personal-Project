# Poker Solver API

啟動服務：

```powershell
poker-serve artifacts\checkpoints\river.pkl --strategy-db artifacts\data\river_strategies.sqlite3
```

Swagger 文件位於 `http://127.0.0.1:8000/docs`。

## `GET /health`

確認服務、已載入的 solver 類型、迭代次數與資訊集數量。

## `POST /v1/river/strategy`

從載入中的 river checkpoint 查詢策略。輸入包含五張公共牌、hero 位置與手牌、底池大小與有效籌碼；大小一律使用 BB。回傳目前資訊集的動作機率，下注動作以底池比例表示。

## `POST /v1/river/database-strategy`

從 SQLite 策略庫查詢已完成訓練工作的 river 策略。請求需提供完整的策略鍵，例如牌面、位置、手牌、底池、有效籌碼、範圍設定與 solver 版本；找不到完全相符的工作會回傳 404，而不是假裝回傳 GTO。

## `POST /v1/multiway-postflop/root-strategy`

查詢目前載入 checkpoint 的多人 postflop 根節點策略。`actions` 為已發生的抽象動作序列：`check`、`call`、`fold`、`all_in` 不帶比例；`bet` 與 `raise` 帶 `pot_ratio`。

實際可用欄位與範例以 Swagger 頁面的 schema 為準。
