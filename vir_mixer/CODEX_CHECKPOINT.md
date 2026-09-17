# VirMixer Codex Checkpoint

## 最後更新／當前階段
2026-09-17，Asia/Taipei。安裝前準備已完成；停在使用者明確指定的 Windows 系統變更批准邊界。
專案：C:\Users\UUU\Documents\GitHub\Personal-Project\vir_mixer。
最新要求：持續自主完成所有不需系統變更的 source/build/package/test 工作，只有真正需安裝／安全設定時停下取得批准。
CODEX_HANDOFF.md 不存在。整個 vir_mixer 在父 Git repository 尚未追蹤（git status: ?? ./）。勿重設／清理使用者檔案；勿修改 sibling voice/RVC。

## 已完成與證據
- Debug、Release x64 皆 compile/link 成功，0 warnings / 0 errors；ApiValidator: Universal。
- 獨立 InfVerif /u: VALID。Inf2Cat: no errors / warnings，CAT 生成成功。
- 正確 SYS 名為 VirMixerAudio.sys，禁止誤用舊 TabletAudioSample.sys。
- 新增可重跑 package.ps1：prepare/build → 新目錄 → InfVerif → Inf2Cat → SHA256 manifest → offline package verification。
- offline 檢查 SYS x64 PE、三個檔案 hash、硬體 ID、服務與端點 INF 定義通過。
- 200,000 次原生 C++ 隨機測試通過：小環形＋實際 19,200/3,840 bytes 容量／預填、overflow、underrun、reset、wrap、frame 資料正確性。
- verify_source.py 通過（含析構順序）、test_prepare.py 1 test 通過（可重現＋防覆寫）。
- test_verify_driver.py 4 tests 通過；拒絕靜音、錯聲道、反相、錯音量、時序錯誤、短 capture。
- PowerShell 全部 script parser 通過；target.ps1 未帶 -Apply 確實拒絕，未更動系統。
- verify_driver.py --run --repeats 1 實際執行：預期 FAIL，VirMixer Input found 0；driver/out/runtime-report.json 記錄 passed:false。此結果絕不可稱為 PCM runtime 通過。

## 最後成功 Build / Package
- driver/out/logs/08-lifetime-debug.log：生命週期修正後 Debug build/API/INF/CAT 全通過。
- driver/out/logs/09-lifetime-release.log：同上 Release 全通過。
- driver/out/logs/10-final-package.log：最終 Debug package pipeline 與 offline checks 通過。
- 最新 Debug package：driver/out/packages/Debug-b709c92583834fc3adfd9293c39ac1bf
- 最新 Release package：driver/out/packages/Release-7d527993818c4d5f9c9c4997bb53df19
- 亦可讀 driver/out/latest-package-Debug.txt / latest-package-Release.txt。
- Debug SYS 219648 bytes，SHA256 8C962A0A062BCD7990C0C0284EEC6E78DE968B25D184481557B38CD22B4C2CEE。
- Release SYS 117760 bytes，SHA256 A09794B0B6270BEB125196C4CF458BF19A4330DA029D0D54C0A9642C959297A7。
- 兩者均未簽章，CAT 亦未簽章；未建立測試憑證。

## 本輪修改
- driver/prepare.py：先 ExDeleteTimer(TRUE,TRUE) / KeFlushQueuedDpcs，才 miniport Release/DPC free；更新舊 sine/save-file 註解。永久修改在 generator，不直接手改生成檔。
- driver/package.ps1：新增完整套件流程及每次獨立目錄。
- driver/target.ps1：新增明確分離的 Sign/Trust/EnableTestMode/Install/Uninstall/RemoveTrust/DisableTestMode/VerifierOn/VerifierOff 操作。全部需管理員＋-Apply，尚未執行。
- driver/tests/ring_test.cpp：新增 production-size priming differential test。
- driver/tests/verify_source.py：新增先停止計時器才釋放物件的斷言。
- driver/tests/verify_package.py：新增 offline hash/PE/INF identity 檢查。
- driver/tests/verify_driver.py：PCM16、shared/exclusive、獨立聲道 correlation/gain/alignment、連續性、重複開關、producer 不存在時靜音、持續長測、磁碟暫存、JSON PASS/FAIL 報告。
- driver/tests/test_verify_driver.py：驗證分析器能拒絕錯音訊。
- driver/RUNTIME_TESTING.md：完整精確命令、系統影響、rollback、sleep/wake、Verifier、Discord/OBS 與 static review。
- driver/README.md、README.md、本檔：更新狀態。

## Static review / 設計限制
每 adapter 一個 nonpaged ring，spin lock 同步 read/write/reset；bridge 無配置、無檔案 I/O。DMA helper 非分頁、frame 對齊；單一底層 render/capture stream。Stream 持有 miniport；miniport 到 adapter 保留 SysVAD parent-managed weak ref。析構順序風險已修正，但 PnP/power/surprise remove 仍須實機驗證。
任何串流 state transition 會清 ring，另一側可能短暫斷音並重新預填；不是 gapless。
原生測試证明 overflow/underrun 演算法；runtime 無 producer silence 可觀察，強制核心 ring overflow 的實際觸發仍需 target debugger/instrumentation，不能用 PortAudio overflow 冒充。

## 當前第一個 Blocker／為何要批准
核心 PCM path 只有真正載入驅動才能驗證。使用者明確禁止未批准的 test mode、憑證 store、Driver Store、root devnode、kernel driver 安裝或 reboot。
到目前為止沒有執行任何上述變更，也沒有執行 Verifier、沒有修改 Secure Boot、沒有對 Discord 通話送音。
沒有 compiler/package blocker。需要用戶批准 test signing + trust + test mode + install 才能進下一步；Secure Boot 更改及 Driver Verifier 是另外批准事項。

## 批准後的精確命令（管理員 PowerShell，repository root）
```powershell
Set-Location 'C:\Users\UUU\Documents\GitHub\Personal-Project\vir_mixer'
$package = (Get-Content .\driver\out\latest-package-Debug.txt -Raw).Trim()
.\driver\target.ps1 -Action Sign -Package $package -Apply
.\driver\target.ps1 -Action Trust -Apply
.\driver\target.ps1 -Action EnableTestMode -Apply
```
Sign：在 CurrentUser\My 建立專用測試憑證，簽 SYS、重建並簽 CAT，輸出全新 Signed-* 套件。Trust：加入 LocalMachine Root/TrustedPublisher。EnableTestMode：bcdedit /set testsigning on。
需要 restart 才能套用 test mode；腳本不自動 reboot。若 Secure Boot 阻擋，停止並說明，不自行關閉。
批准並完成 restart 後：
```powershell
.\driver\target.ps1 -Action Install -Apply
& 'C:\Users\UUU\anaconda3\envs\xin\python.exe' driver/tests/verify_driver.py --run --repeats 10 --report driver/out/runtime-shared.json
& 'C:\Users\UUU\anaconda3\envs\xin\python.exe' driver/tests/verify_driver.py --run --exclusive --repeats 100 --report driver/out/runtime-exclusive.json
& 'C:\Users\UUU\anaconda3\envs\xin\python.exe' driver/tests/verify_driver.py --run --exclusive --repeats 1 --long-seconds 3600 --report driver/out/runtime-hour.json
```
Install 驗 hashes/signatures/catalog membership 後，devcon install <signed INF> Root\VirMixerAudio，新增 root devnode、Driver Store package 與 VirMixerAudio service，拒絕 duplicate devnode。可能需 restart，會回報不自動執行。
最壞風險：擴大測試憑證信任、可載入測試簽章核心碼，kernel bug 可造成 hang/BSOD/音效失效甚至無法正常開機。建議可回復測試機／VM，保存工作與恢復金鑰。

## Rollback
先關閉使用 VirMixer 的應用程式，讀 driver/out/target-state/installed.json 或 pnputil /enum-drivers 的實際 oemNN.inf：
```powershell
.\driver\target.ps1 -Action Uninstall -PublishedInf oemNN.inf -Apply
.\driver\target.ps1 -Action RemoveTrust -Apply
.\driver\target.ps1 -Action DisableTestMode -Apply
```
必須換成真實 OEM INF。腳本檢查 provider=VirMixer Project、原始 INF=VirMixerAudio.inf，避免移除其他驅動。移除憑證只依保存的 thumbprint。最後 restart，驗證 devnode/service/package 不再載入、憑證已移除、testsigning off。
若另批准啟用 Verifier：用 verifier /reset 後 restart；該命令清除全部 Verifier 設定，先保存原設定。無法正常啟動時從 Safe Mode/recovery 回復，勿猜測刪除 driver 檔案。詳見 RUNTIME_TESTING.md。

## 不要重做／可直接使用
MSB8020、sideband guards、TargetName 組態覆寫、x86 驗證工具載入錯誤皆已解決。不要重裝工具鏈或還原 sideband。
MSBuild：C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe。
SDK/WDK：10.0.28000.0；InfVerif 在 Tools/<version>/x64，Inf2Cat 在 bin/<version>/x86（工具正常可用）。
Python：C:\Users\UUU\anaconda3\envs\xin\python.exe。
Zig：C:\Users\UUU\Documents\Codex\2026-09-16\referenced-chatgpt-conversation-this-is-an-2\work\driver-toolchain\ziglang\zig.exe。
Source pinned upstream：97429c5623590d52f001249460daf43e6749d777。

## 下一 session 第一個 action
讀本檔、RUNTIME_TESTING.md、最新使用者是否批准；若有批准，依上述 staged commands 進行，不再研究已解決工具鏈；若沒有批准，不执行系統變更。安裝／端點出現不是成功，必須實測 PCM 並修問題。睡眠、Verifier、Discord/OBS 按文件逐項記錄證據，維持 checkpoint。正式 Microsoft signing、Steam 整合不阻擋目前測試準備。
