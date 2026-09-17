# 模型放置與相容性

本機已有 ContentVec、原始 PTH、舊 voice.onnx 與修正版 voice.dynamic.onnx；目前設定使用修正版。AI 模式缺檔會顯示錯誤，絕不自動改用 DSP。

## 最小目錄

```text
voice/
  models/
    config.json
    voice.onnx                 # 你要使用的 RVC 音色模型
    vec-768-layer-12.onnx       # RVC v2 的內容編碼器
```

`config.json` 的所有相對路徑均相對於這個 JSON，與你從哪個資料夾啟動無關：

```json
{
  "voice_model": "voice.dynamic.onnx",
  "contentvec": "vec-768-layer-12.onnx",
  "sample_rate": 40000,
  "speaker_id": 0
}
```

`sample_rate` 必須等於訓練模型的原生取樣率（32000、40000、48000），不是輸入音檔的 sr。單一音色通常用 speaker_id=0；多說話者模型需依模型說明選 ID。若 ONNX 有 config metadata 會交叉檢查；沒有則以輸出長度檢查取樣率。

## 取得什麼模型

1. **音色模型**：使用你已有或可使用的 RVC v1/v2、啟用 F0 的音色權重，透過 RVC 的 ONNX exporter 匯出。此程式不直接讀 `.pth`，也不能僅改副檔名。
   官方 exporter：https://github.com/RVC-Project/Retrieval-based-Voice-Conversion/blob/develop/rvc/modules/onnx/export.py
2. **ContentVec**：v1 使用 `vec-256-layer-9.onnx`；v2 使用 `vec-768-layer-12.onnx`。可在 https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/tree/main 找到這兩個檔案；分別約 293 MB / 378 MB，請依該模型頁的授權使用。
3. 這版 F0 使用已安裝 librosa 的 pYIN，所以**不需要 rmvpe.onnx**。

可以在 GUI 選擇另一份 JSON 切換音色。替換同一路徑的模型檔後請重啟程式，使 ONNX session 重新載入。

## 嚴格支援的 ONNX 介面

ContentVec：單輸入 `[1,1,samples]` float waveform，16 kHz；輸出 `[1,T,256]` 或 `[1,T,768]`。

RVC：六輸入、動態時間長度的匯出：

| 名稱 | Shape | 意義 |
|---|---|---|
| phone | [1,T,256/768] | 內容特徵 |
| phone_lengths | [1] | T |
| pitch | [1,T] int64 | F0 mel bins |
| pitchf | [1,T] float | F0 Hz |
| ds | [1] int64 | speaker ID |
| rnd | [1,192,T] float | 生成器噪聲 |

CPU 建議 FP32。程式也依輸入型別轉成 FP16，但能否执行取決於個別 ONNX graph / runtime 的 CPU operator 支援。優先使用 FP32 匯出。

這不等於支援所有副檔名為 ONNX 的檔案：w-okada 的 `feats/p_len/sid` 五輸入匯出、其他 vocoder、無 F0 模型、固定 T 匯出、一般 HuBERT PyTorch 權重皆不相容。錯誤會顯示在狀態列。`.index` 是可選的檢索加強，此 MVP 尚未實作；不需準備它。

第一個測試建議短、乾淨、單人說話 WAV，先 Pitch=0、Speed=1，再逐步調整。音質與模型訓練內容、Pitch 選擇、pYIN 的估計品質都有關。
