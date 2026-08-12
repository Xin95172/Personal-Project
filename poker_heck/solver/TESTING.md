# 測試

執行核心引擎、求解器與 API 的完整測試與 coverage：

```powershell
python -m pytest --cov=poker_solver.engine --cov=poker_solver.solver_core --cov=poker_solver.api --cov-branch -p no:cacheprovider -q
```

coverage 門檻為 90%，設定於 `pyproject.toml`。CLI 是薄薄的命令列轉接層，會由 integration test 驗證其工作流程，但不計入核心 coverage 門檻。

- `tests/unit/`：規則、狀態轉換、下注 abstraction、牌面同構化與求解器元件。
- `tests/integration/`：checkpoint、策略庫、API、pack 產生器與 manifest 流程。

若要快速確認某個訓練網格可被解析，可先執行對應產生器；完整 canonical 牌面網格可能需要大量時間與磁碟空間。
