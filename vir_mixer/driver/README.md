> **更新 2026-09-17：Debug / Release、Universal API、獨立 InfVerif、Inf2Cat、套件 SHA256 驗證均通過。**
> 新套件請依 `out/latest-package-Debug.txt` / `out/latest-package-Release.txt`，不要使用舊二進位。
> 已修正串流 teardown 順序；200,000 次原生隨機測試及 PCM 分析器測試見 checkpoint。
> 尚未簽章／安裝；真正 PCM runtime 未驗證。後續操作、精確命令和 rollback 見 [RUNTIME_TESTING.md](RUNTIME_TESTING.md)。
# VirMixer Virtual Audio — 開發原型

## 目標接法

```text
Mixer / Windows 應用程式
  → VirMixer Input（播放端）
  → 本驅動的 PCM 緩衝
  → VirMixer Output（錄音端）
  → Discord / OBS / 錄音程式
```

驅動以 Microsoft SysVAD 的 WaveRT 時鐘、DMA、PnP／電源管理架構為基礎。
固定來源版本：`97429c5623590d52f001249460daf43e6749d777`。
這是 SysVAD 衍生開發，不是 VB-CABLE 的更名或包裝，亦沒有複製其驅動。

## 已實作

- 獨立 `Root\VirMixerAudio` 硬體 ID 與 `VirMixerAudio` 服務。
- 自訂播放端／錄音端名稱與名稱 GUID，只註冊一組端點。
- 48,000Hz、PCM16、雙聲道的硬體格式，兩端相同。
  Windows shared-mode audio engine 負責應用端的 float／其他取樣率轉換；
  不支援其他格式的 exclusive-mode stream。
- 每個 adapter 自己持有非分頁環形緩衝；以 spin lock 保護跨串流存取。
- 100ms 容量，20ms 初始／欠載恢復預填；不足補零，超出容量丟棄最舊資料。
- 串流狀態改變會清除緩衝，避免重新播放時讀到先前音訊。
- 移除範例的 capture 正弦波，render 路徑改為写入音訊橋接；不在核心寫 WAV 檔。
- 限制一個底層 render stream 與一個底層 capture stream；一般多應用程式使用
  Windows shared mode，由 audio engine 混合／分送。未提供硬體 offload 與 render loopback pin。
- 移除未實作的 capture 硬體音量／靜音／電平節點；未加入 APO 或藍牙／USB sideband。

這些是原始碼的設計與實作範圍，端點實際名稱、格式協商及音訊品質仍需在 Windows 上驗證。
不包含任意應用程式的多聲卡路由、ASIO、多取樣率核心重採樣或正式驅動發行。

## 檔案

- `core/AudioRing.h`：不依賴 Windows 的 PCM 環形緩衝。
- `core/VirtualCable.h`：Windows 核心 spin lock 包裝。
- `prepare.py`：取得固定 Microsoft 版本並產生我們的修改版。
- `build-source/audio/sysvad/`：已產生的完整驅動原始碼，可直接檢視。
- `package/VirMixerAudio.inx`：自訂驅動套件描述。
- `build.ps1`：檢查工具鏈並編譯，不會安裝驅動、加入憑證或改安全設定。
- `test-core.ps1`、`tests/ring_test.cpp`：原生 C++ 測試。
- `tests/verify_source.py`：檢查產生後的關鍵程式碼與專案設定。
- `tests/verify_driver.py`：驅動安裝後的真實雙聲道回傳驗證。

`upstream/` 與 `build-source/` 是可重建內容，已從 Git 排除。Microsoft 原始碼授權保留於
`upstream/LICENSE` 和 `build-source/LICENSE-Microsoft`。在乾淨 checkout 執行
`python driver/prepare.py` 可重新生成，需 Git 與網路。
生成目錄包含 SHA256 manifest；沒有手動修改生成來源時，可重複執行普通 prepare。
若生成來源被手改或缺少 manifest，預設拒絕覆寫；`--refresh-generated` 會覆蓋其中的生成檔案，先保留自己的修改。
永久修正應放在 prepare.py 或 core，而非只改 build-source。

## 建置與測試

1. 此電腦的 Visual Studio 2026／MSVC／SDK／WDK 已正常，無需重裝。新機器才需安裝相容版本。
   請依 [Microsoft 的 WDK 安裝與版本搭配說明](https://learn.microsoft.com/en-us/windows-hardware/drivers/download-the-wdk)
   選擇匹配版本，包含 WindowsKernelModeDriver10.0 工具集與 C++ Spectre libraries。
2. 從 `vir_mixer` 目錄執行：

```powershell
python driver/prepare.py  # 可重複生成；會保護手動修改過的生成來源
powershell -File driver/build.ps1
```

編譯目標為 x64。先建 EndpointsCommon 靜態程式庫，再建 `VirMixerAudio.sys`。
原始 SysVAD solution 包含其他不使用的 APO 範例，請使用本專案的 build.ps1。
build.ps1 使用 amd64 MSBuild 與 x64 工具，避免誤用 x86 驗證工具。
本機 Debug / Release build 均已成功；獨立 InfVerif、Inf2Cat 也已通過。測試簽章／安裝另行處理，build.ps1 不執行安裝。

原生緩衝測試（在 x64 Native Tools PowerShell，有 `cl.exe`）：

```powershell
powershell -File driver/test-core.ps1
python driver/tests/verify_source.py
python driver/tests/test_prepare.py -v
```

也可指定獨立 Zig 編譯器：`powershell -File driver/test-core.ps1 -Zig C:\path\to\zig.exe`。
測試涵蓋 100,000 次隨機讀寫、wrap、超載丟棄、欠載補零、重設、預填與 stereo frame 對齊。

## 尚未完成的安裝驗證

完整驅動需要先通過 WDK 編譯、InfVerif、Inf2Cat 與簽章檢查，再在獨立測試機／VM
驗證安裝與移除、睡眠／喚醒、多應用程式 shared mode、長時間播放、串流啟停、
Driver Verifier 與音訊延遲。測試機需要能使用相應的測試簽章設定。
driver/target.ps1 已備妥分開的測試簽章、信任、安裝及回復操作；必須取得批准並明確指定 -Apply，尚未執行。不會自動關閉 Secure Boot。

Root-enumerated 裝置還需要建立相符的 root devnode；只執行 `pnputil /add-driver ... /install`
未必會建立該 devnode。請使用 WDK 的測試部署流程，對 `Root\VirMixerAudio` 建立裝置。
不要在尚未編譯／簽章成功前執行安裝。

安裝到測試機後，用**新開啟**的 Python 程序執行：

```powershell
python driver/tests/verify_driver.py --run
```

它會向 VirMixer 播放端送出兩個不同的低音量測試訊號，從 VirMixer 錄音端接收並檢查
每聲道相關性與輸入欠載。這可區分「真正傳送 PCM」與「只有裝置名稱／產生範例正弦波」。
此項目前尚未執行成功，因為沒有已安裝的 VirMixer 驅動。

正式發行還需要符合 Microsoft 的驅動簽章要求；開發測試簽章不等於正式發行簽章。
參考：[SysVAD](https://learn.microsoft.com/en-us/samples/microsoft/windows-driver-samples/sysvad-virtual-audio-device-driver-sample/)、
[驅動簽章](https://learn.microsoft.com/en-us/windows-hardware/drivers/install/driver-signing)、
[音訊端點命名](https://learn.microsoft.com/en-us/windows-hardware/drivers/audio/friendly-names-for-audio-endpoint-devices)。


