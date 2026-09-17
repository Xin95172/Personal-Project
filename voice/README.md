# Voice Changer：學習型 RVC MVP

## 2026-09-16：真實模型轉換已驗證

目前已有真正的 `vec-768-layer-12.onnx` 與 `maobailing_rvc.pth`。已由後者重新匯出 `models/voice.dynamic.onnx`，`models/config.json` 現在指向此修正版；原本 `voice.onnx` 保留，未覆寫。

先前匯出器的 `attentions.py` 在 relative attention 中使用 `int(length)` 等 Python 整數轉換，讓部分 padding/reshape 長度在 ONNX trace 時固定，導致非匯出範例長度的輸入出現 Reshape 錯誤。已將這些計算保留為動態長度，再以 FP32、eval 模式匯出。新增 `export_rvc.py` 可重現匯出，需搭配本機已修正的 `rvc-export` 原始碼、torch 與 onnx。

```powershell
& C:\Users\UUU\anaconda3\envs\xin\python.exe -X utf8 export_rvc.py models/maobailing_rvc.pth models/voice.dynamic.onnx
```

匯出腳本先寫 pending 檔，通過 ONNX checker 及 T=128、200、900、174 的實際模型推論後才交付目標檔；預期輸出分別為 51200、80000、360000、69600 samples。保留六輸入契約，並寫入 config/version metadata。

已使用真正的 ContentVec 與修正版 RVC，完成原始 `p0002-7.mp3` 全長轉換：9 段，Pitch=0、Speed=1、speaker_id=0，輸入 70.5195625 秒，輸出 2820782 samples / 40000 Hz = 70.51955 秒；輸出有限、非全零，peak 約 0.8721。原有 9 個自動化測試也重新通過。

這次已驗證真實權重端到端執行。是否符合預期音色、發音自然度和個人聽感，仍需聆聽比較；不是主觀音質評分。下方早期測試紀錄中的人工 ONNX graph 測試仍保留，不能與這次真實模型測試混為一談。


完整技術交接與 ChatGPT 上下文請讀 [README_CHATGPT.md](README_CHATGPT.md)。

保留 `app.py` + `audio_engine.py` 與原本五個操作（Pitch、Speed、Play Original、Play Converted、Stop）。增加 Load Audio、模式選擇、模型設定 JSON，實作離線音檔轉換。這版尚未做麥克風即時串流。

## 啟動

這台電腦已在原本的 `xin` conda 環境安裝 ONNX Runtime，原有 PySide6/librosa 套件沿用。

```powershell
cd C:\Users\UUU\Documents\GitHub\Personal-Project\voice
.\start.ps1
```

或：

```powershell
& C:\Users\UUU\anaconda3\envs\xin\python.exe -X utf8 app.py
```

其他電腦建議 Python 3.10，安裝 `python -m pip install -r requirements.txt` 後 `python app.py`。不要誤用這台電腦預設的 Python 3.14；啟動腳本已指定原本的 xin 環境。

1. 啟動會載入原有 `p0002-7.mp3`；也可按 Load Audio 選 WAV/MP3/FLAC/OGG。
2. 尚無模型時，可選 **DSP 比較模式**，立即使用原本的音高與速度功能。
3. **AI Voice Conversion** 需要兩個相容 ONNX 模型，位置與格式見 `models/README.md`。
4. Play Converted 在背景執行，狀態列顯示目前步驟，完成後播放。
5. Stop 立即停止播放並要求取消。單次 ONNX/pYIN 呼叫不能中途強制終止，結束該步驟後才停止；取消後不會自動播放舊結果。轉換中關閉視窗會等該步驟結束。

## 每個檔案負責什麼

| 檔案 | 職責 |
|---|---|
| `app.py`（修改） | PySide6 控制項、檔案選擇、QThread 工作、進度與錯誤、播放與取消 |
| `audio_engine.py`（修改） | 保留 load_audio、pitch_shift、change_speed、convert_voice、play_audio、stop_audio；convert_voice 明確為原本 DSP 練習 |
| `voice_converter.py`（新增） | VoiceConverter 統一入口、設定驗證、選擇 AI/DSP、模型快取、速度處理與輸出峰值控制；回傳 waveform + sample rate |
| `pipeline.py`（新增） | 重採樣、前處理、分段、呼叫特徵/F0/模型、接合 waveform |
| `features.py`（新增） | ContentVec 神經網路內容特徵、pYIN F0、半音調整與 RVC pitch bins |
| `rvc_model.py`（新增） | ONNX session、模型介面檢查、輸入型別、真正的 session.run 模型推論 |
| `models/config.json`（新增） | 相對於 JSON 所在目錄的模型路徑、原生取樣率與 speaker ID |
| `models/README.md`（新增） | 模型取得、匯出格式、配置範例與限制 |
| `convert_file.py`（新增） | 不開 GUI 也能跑同一條轉換流程並輸出 WAV |
| `start.ps1`（新增） | 使用原有 xin 環境啟動 |
| `requirements.txt`（修改） | 新增 onnxruntime==1.23.2 |
| `requirements-test.txt`、`tests/test_conversion.py`、`tests/test_gui.py`（新增） | 模型介面、F0、分段接合、取消與 GUI 測試 |
| `.gitignore`、`README.md`（新增） | 不追蹤模型權重、使用與學習說明 |

原本 notebook、MP3、output.wav 不需修改。

## Play Converted 到模型 inference 的完整資料流

```text
Load Audio / 啟動載入 → audio_engine.load_audio()
  → librosa 讀檔、混成 mono float waveform，保留來源 sr
app.py: VoiceChanger.play_converted()
  → 讀取 Pitch、Speed、模式、JSON 路徑
  → ConversionWorker.run() [背景 QThread]
  → VoiceConverter.convert(audio, sr, pitch, speed, backend='rvc')
  → ModelConfig.load()：檢查權重與取樣率
  → RVCPipeline(config) [第一次載入；同設定重用 session]
  → RVCPipeline.run()
      → librosa.resample：來源 sr → 16000 Hz
      → 峰值控制 + 48 Hz 高通
      → 8 秒分段，100 ms 重疊，左右 500 ms 反射 padding
      ├→ ContentEncoder.extract()
      │    → ContentVec ONNX session.run([1,1,samples])
      │    → [1,T,256/768] 語音內容特徵
      │    → 每格複製兩次：20 ms/frame → 10 ms/frame
      └→ extract_f0()
           → librosa.pyin：50–1100 Hz，10 ms/frame，無聲段 F0=0
           → F0 × 2 ** (Pitch / 12)
           → pitchf [1,T]：Hz；pitch [1,T]：1–255 的 mel bins
      → RVCModel.infer(content, pitch, pitchf)
           → phone [1,T,256/768]
           → phone_lengths [1]
           → pitch [1,T]、pitchf [1,T]
           → ds [1]：speaker ID
           → rnd [1,192,T]：生成器噪聲
           → RVC ONNX session.run(feed)  ← 真正的音色模型推論
           → 模型原生 32/40/48 kHz waveform
      → 去 padding、100 ms crossfade / overlap-add
  → change_speed() [僅 Speed != 1 時，保持音高的 time stretch]
  → 檢查有限值、超過 0.99 時縮小峰值
  → ConvertedAudio(audio, sample_rate)
  → converted Signal 回主執行緒
  → VoiceChanger.on_converted() [若已 Stop 則丟棄]
  → audio_engine.play_audio(result.audio, result.sample_rate)
  → sounddevice → Windows 預設喇叭
```

AI 模式的 Pitch 不會先對輸入跑 librosa pitch_shift，而是調整模型的 F0 conditioning。pYIN 是傳統音高估計器；音色轉換本身由 ContentVec + RVC 神經網路完成。這版沒有 FAISS `.index` 檢索、RMVPE、GPU 設定或訓練功能，先讓核心資料流保持可讀。

模型的取樣率與原始音檔可能不同，所以不可再固定 `play_audio(converted, self.sr)`。這是此次引入 ConvertedAudio 的原因。

## 命令列與測試

```powershell
# 安裝執行環境（本機已安裝）
& C:\Users\UUU\anaconda3\envs\xin\python.exe -m pip install -r requirements.txt
# 有模型後輸出真正 RVC 變聲 WAV
& C:\Users\UUU\anaconda3\envs\xin\python.exe -X utf8 convert_file.py input.wav converted.wav --config models/config.json --pitch 3
# 無權重也可測試原本的 DSP
& C:\Users\UUU\anaconda3\envs\xin\python.exe -X utf8 convert_file.py input.wav converted.wav --backend dsp --pitch 3 --speed 1.1
# 測試依賴 onnx 只用來建立小型測試圖；執行 GUI 不需要它
& C:\Users\UUU\anaconda3\envs\xin\python.exe -m pip install -r requirements-test.txt
& C:\Users\UUU\anaconda3\envs\xin\python.exe -X utf8 -m unittest discover -s tests -v
```

已通過 9 個測試：F0 倍頻/靜音、輸入驗證/取消/缺檔、真 ONNX Runtime 執行合成測試圖、跨段與尾段接合、取樣率不符拒絕、DSP 速度、GUI DSP 播放入口、GUI AI 按鈕到 ONNX 推論與 40 kHz 播放、缺模型後恢復、Stop/關閉丟棄結果（部分為同一測試的多個斷言）。

**早期驗證紀錄：最初僅用人工測試圖驗證接線。現在已另外完成真實模型全長轉換，見本文頂端的更新。人工測試圖本身仍不能當成聲音模型。** 放入相容的權重後，請先用 3–10 秒乾淨人聲測試；目前是 CPU 離線處理，不保證即時速度。

## 參考

- 分層設計思路：https://github.com/w-okada/voice-changer
- RVC 官方六輸入 ONNX 匯出：https://github.com/RVC-Project/Retrieval-based-Voice-Conversion/blob/develop/rvc/modules/onnx/export.py
- ContentVec/RVC ONNX 介面參考：https://github.com/dev6699/rvc-onnx

僅參考架構與模型介面，未複製 w-okada 大型專案，也未安裝其 server/UI。

實機驗證：原有 70.52 秒 / 48 kHz MP3 成功載入；擷取 1 秒人聲以 Pitch=+3、Speed=1.1 轉換後，透過 sounddevice 實際播放成功。PySide6 視窗成功顯示並擷取檢視。預設音訊裝置支援模型常用的 40 kHz；pip check 無相依衝突。


