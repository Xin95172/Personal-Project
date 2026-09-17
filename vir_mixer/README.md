# Audio Mixer · File Console

自研虛擬聲卡開發原型位於 `driver/`；請先閱讀 `driver/README.md`。
Debug / Release x64 驅動、INF / catalog / 套件驗證已通過；尚未簽章／安裝或驗證核心音訊通路。安裝準備、PCM 測試與回復方式見 `driver/RUNTIME_TESTING.md`。

獨立的多音檔混音台，位於 `vir_mixer`，不匯入或修改旁邊的 `voice` / RVC 專案。
此版本依目前需求專注於音檔混音，不是錄音室 DAW 或硬體混音器的完整替代品。

## 啟動

PowerShell 可從任何資料夾執行：

```powershell
& C:\Users\UUU\anaconda3\envs\xin\python.exe -X utf8 C:\Users\UUU\Documents\GitHub\Personal-Project\vir_mixer\run_mixer.py
```

也可在本資料夾執行 `start.ps1`。預設加入 `p0002-7.mp3`，但不自動播放。
新環境安裝：`python -m pip install -r requirements.txt`。

## 使用混音台

1. 按「加入音檔」選取多個檔案，或拖放 MP3 / WAV / FLAC / OGG 到視窗。
   同一個音檔可以加入多次，每一軌都有自己的播放位置。最多 32 軌。
2. 每軌獨立播放、暫停、停止；停止會回到開頭，播完後再按播放會從頭開始。
   上方「全部播放／暫停／停止」控制已載入的全部音軌。
3. 拖曳進度條並放開即可跳轉；勾選「循環播放」會反覆播放該音檔。
4. 垂直推桿控制音量（−∞ 至 +12 dB），「0 dB」恢復原始倍率。
   最低位置完全靜音。BAL 控制立體聲左右平衡；中央保留原本 L/R。
5. MUTE 靜音；任何音軌 SOLO 時，只輸出所有 SOLO 音軌。MUTE 優先。
   暫停或播完的 SOLO 音軌仍維持 SOLO，需取消 SOLO 才會恢復其他音軌。
6. MASTER 控制總音量（−∞ 至 +6 dB）與總靜音。
   電平表顯示 L/R 峰值，單位 dBFS；音軌電平在推桿後，MASTER 電平在安全截幅前。
   CLIP 亮起表示主混音超過 0 dBFS，應降低推桿；點擊 CLIP 可重設指示燈。
7. 點擊「移除」會停止並移除該軌。音軌多時可水平捲動，小視窗亦支援垂直捲動。

改變音量、Mute、Solo、BAL 會在一個 10ms block 內漸變，減少控制切換的爆音。
主輸出另有 [-1, 1] 硬截幅保護，並不是壓縮器或專業限幅器。

## 主輸出與 Discord

- 「輸入／輸出」按鈕可開啟完整装置清單，包含可被音訊系統偵測的麥克風、喇叭、耳機、
  HDMI／螢幕音訊與虛擬裝置。支援搜尋與輸入／輸出篩選，顯示聲道數、預設取樣率與系統預設。
  相同硬體以不同音訊介面出現時會保留各個端點，不會誤合併成同一列。
- 每 5 秒用獨立程序重新偵測，亦可按「更新裝置」或「立即重新掃描」。背景掃描不會中斷播放。
  主輸出啟動前會在沒有開啟音訊串流時更新裝置並依名稱／介面重新定位，避免使用過期索引。
  已拔除、停用或驅動未公開的裝置不保證出現在清單；列出不代表硬體一定能成功開啟。
  輸入清單目前僅供偵測與檢視，尚未實作麥克風輸入音軌。

- 預設使用 Windows 預設輸出；可選實體喇叭、耳機或已安裝的虛擬音訊裝置。
- 播放或錄音會開啟音訊輸出。要換裝置，先按「關閉輸出」。
  「關閉輸出」會停止所有音軌並完成錄音；「全部停止」只停止音軌，錄音會繼續錄下靜音。
- Discord 接法：本程式主輸出選 **CABLE Input**，Discord 麥克風選 **CABLE Output**，
  Discord 喇叭仍選實體耳機。必須先安裝 VB-CABLE；程式不會安裝驅動程式。
- 此版只有一個主輸出，未提供主輸出加耳機的雙裝置監聽，也不收取麥克風或桌面音效。
- 音訊裝置必須支援 48kHz；會先檢查單／雙聲道輸出能力。
  如果裝置開啟失敗，狀態列會顯示原因，可以改選其他裝置重試。
  新安裝或插拔裝置後若清單未更新，請重開程式。

## 錄音

「開始錄音」會讓你選擇新 WAV 檔名，格式為 **48kHz、24-bit、雙聲道**。
錄下送往主輸出的混音（含主推桿、Mute、Solo、BAL 和截幅），不是每軌各自錄檔。
錄音不會自動播放音軌，可以先開始錄音再按播放。暫停音軌期間仍持續錄製。
錄音時間顯示於 MASTER 區域；按「停止錄音」後背景會排空待寫資料並完成檔案。
狀態列顯示完整存檔位置。為保留原錄音，必須使用新檔名，程式不覆蓋現有錄音。
關閉視窗會等待錄音收尾。

## 設定存取

「儲存設定」建立 JSON 場景，保存音檔路徑、音軌名稱、音量、平衡、Mute、Solo、
循環、總音量與輸出裝置名稱。載入前先關閉輸出。
設定不內嵌音檔、不保存播放位置、不自動播放或錄音；搬移音檔後需要重新加入。
裝置用名稱與音訊介面辨識，不依賴容易變動的裝置索引；裝置遺失時需手動重新選擇。
程式不會自動儲存未存的設定，離開前請按「儲存設定」。

## 原有 CLI

在專案資料夾執行：

```powershell
python run_mixer.py --list-devices
python run_mixer.py --cli --seconds 3 --gain 0.5 --master-gain 0.5
python run_mixer.py --cli example.wav --device 3
```

CLI 保留單檔測試用途；`--gain`、`--master-gain` 是線性倍率，只在 CLI 生效。
GUI 請使用介面上的 dB 推桿。CLI 按 Ctrl+C 停止。

## 驗證

```powershell
python -m unittest test_mixer test_gui test_devices -v
```

核心測試涵蓋 PCM、多軌獨立播放、循環尾塊、重採樣、Solo/Mute、平衡、電平、音量漸變、
WAV 錄音內容與設定驗證；GUI 測試預設使用模擬輸出，會打開短暫測試視窗。

若要實際喇叭測試，PowerShell 執行：

```powershell
$env:MIXER_REAL_AUDIO = '1'
python -m unittest test_gui -v
Remove-Item Env:MIXER_REAL_AUDIO
```

## 架構與目前限制

- `audio_source.py`：背景解碼與必要的完整重採樣，統一 48kHz float32。
- `mixer_engine.py`：只依赖 PCM 的 gain / mute / solo / balance / master 與電平運算。
- `transport.py`：每軌播放位置、暫停、停止、循環、跳轉與尾塊補零。
- `audio_output.py`：裝置檢查、單／雙聲道適配、blocking 音訊輸出。
- `devices.py`、`device_panel.py`：完整輸入／輸出清單、背景重新偵測、搜尋與裝置定位。
- `runtime.py`：專用音訊執行緒、命令佇列與獨立解碼執行緒。
- `recorder.py`：有界佇列與獨立寫檔執行緒；磁碟寫入異常會停止錄音並回報。
- `session.py`：JSON 場景驗證與存取。
- `mixer_gui.py`：多軌控制台、推桿、電平表與錄音介面。

目前整檔解碼到記憶體，再以 480-frame / 10ms block 混音，並非磁碟串流解碼。
多個長音檔會佔用較多 RAM；32 軌為介面上限，不代表所有硬體都能無欠載地播放 32 軌。
輸出延遲依 Windows 裝置緩衝而定，沒有專業硬即時保證。
錄音保存引擎輸出的 samples，不會補償外部裝置欠載造成的實際播放時間差。
尚未包含麥克風、desktop loopback、EQ、壓縮器、AUX／雙輸出、VST、RVC 或虛擬聲卡驅動。

實作參考：[sounddevice stream API](https://python-sounddevice.readthedocs.io/en/latest/api/streams.html)、
[Qt 執行緒與訊號](https://doc.qt.io/qtforpython-6/overviews/qtdoc-threads-synchronizing.html)。

