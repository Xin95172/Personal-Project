# Voice Changer — 給 ChatGPT 的完整架構與程式交接 README

## 2026-09-16：真實模型轉換已驗證

目前已有真正的 `vec-768-layer-12.onnx` 與 `maobailing_rvc.pth`。已由後者重新匯出 `models/voice.dynamic.onnx`，`models/config.json` 現在指向此修正版；原本 `voice.onnx` 保留，未覆寫。

先前匯出器的 `attentions.py` 在 relative attention 中使用 `int(length)` 等 Python 整數轉換，讓部分 padding/reshape 長度在 ONNX trace 時固定，導致非匯出範例長度的輸入出現 Reshape 錯誤。已將這些計算保留為動態長度，再以 FP32、eval 模式匯出。新增 `export_rvc.py` 可重現匯出，需搭配本機已修正的 `rvc-export` 原始碼、torch 與 onnx。

```powershell
& C:\Users\UUU\anaconda3\envs\xin\python.exe -X utf8 export_rvc.py models/maobailing_rvc.pth models/voice.dynamic.onnx
```

匯出腳本先寫 pending 檔，通過 ONNX checker 及 T=128、200、900、174 的實際模型推論後才交付目標檔；預期輸出分別為 51200、80000、360000、69600 samples。保留六輸入契約，並寫入 config/version metadata。

已使用真正的 ContentVec 與修正版 RVC，完成原始 `p0002-7.mp3` 全長轉換：9 段，Pitch=0、Speed=1、speaker_id=0，輸入 70.5195625 秒，輸出 2820782 samples / 40000 Hz = 70.51955 秒；輸出有限、非全零，peak 約 0.8721。原有 9 個自動化測試也重新通過。

這次已驗證真實權重端到端執行。是否符合預期音色、發音自然度和個人聽感，仍需聆聽比較；不是主觀音質評分。下方早期測試紀錄中的人工 ONNX graph 測試仍保留，不能與這次真實模型測試混為一談。


> 文件日期：2026-09-16。本文依照目前專案原始碼整理，可單獨上傳給 ChatGPT 作為專案背景。這是實作現況的說明，不是未來功能規格。若之後程式有修改，以最新原始碼為準。

## 1. 先理解這個專案目前做到了哪裡

這是一個 **Python + PySide6 的離線音檔 voice conversion 學習專案**。使用者原本只有 `app.py` 與 `audio_engine.py`，可以載入音檔、用 librosa 調整音高與速度，再播放。現在在保留這個學習結構的前提下，加入獨立的 VoiceConverter、Pipeline、內容特徵、F0 與 ONNX 模型推論模組。

設計參考 w-okada/voice-changer 將 GUI、音訊流程、特徵處理與模型分開的思路，沒有複製或啟動它的大型 server/client 專案。

目前有兩個明確分開的模式：

| 模式 | 轉換方法 | 是否需要權重 | 輸出取樣率 |
|---|---|---|---|
| `dsp` | librosa pitch_shift + time_stretch | 不需要 | 原始音檔取樣率 |
| `rvc` | ContentVec ONNX + pYIN F0 + RVC ONNX 音色模型 | 需要兩個 ONNX 檔 | 音色模型的原生取樣率 |

**目前本機已有真正的 ContentVec 與角色音色權重，並完成真實模型全長轉換（詳見文件開頭更新）。** 原始人工測試圖仍只用來驗證程式契約，真實音質尚需使用者聆聽驗收。沒有權重時選 AI 模式會顯示錯誤，不會默默降級成 DSP。

這版是「讀入整個音檔 → 分段推論 → 全部完成 → 播放」。不是麥克風即時變聲，也不是一邊生成一邊播放。

## 2. 學習背景與需要保留的結構

使用者正在學 waveform、sample rate、STFT、F0、harmonics、formant、spectral envelope，以及 voice conversion 的程式結構。因此這個專案優先考量可讀性與能追蹤的資料流。

現有 GUI 操作包括：

- Load Audio：選音檔。
- 模式選單：AI Voice Conversion / DSP 比較模式。
- 選擇模型設定 JSON。
- Pitch：-12 到 +12 半音。
- Speed：0.50x 到 1.50x，初始 1.00x。
- Play Original、Play Converted、Stop。

目前沒有 Formant 滑桿，也沒有獨立的 formant shifting 演算法。AI 音色變化依賴載入的聲音模型，不是用某個 formant 參數冒充角色模型。

## 3. 專案位置與檔案地圖

目前電腦上的專案根目錄：

```text
C:\Users\UUU\Documents\GitHub\Personal-Project\voice
```

以下樹狀圖中的檔名皆相對於這個目錄：

```text
voice/
├── app.py                    # GUI、工作執行緒、進度、播放與取消
├── audio_engine.py           # 音檔 I/O、原本 DSP 功能、播放/停止
├── voice_converter.py        # 統一轉換 API、設定、模式選擇、結果型別
├── pipeline.py               # RVC 前處理、分段、特徵/F0/模型串接、接合
├── features.py               # ContentEncoder、extract_f0
├── rvc_model.py              # ONNX Runtime session 與 RVC tensor 介面
├── convert_file.py           # 不經 GUI 的命令列轉換與 WAV 寫出
├── start.ps1                 # 使用既有 xin Python 環境啟動 GUI
├── requirements.txt          # 執行環境與原有 notebook 相依套件
├── requirements-test.txt     # 額外加入 onnx，供測試建立人工模型圖
├── models/
│   ├── config.json           # 已存在，僅設定，不含權重
│   ├── README.md             # 模型取得與相容格式說明
│   ├── voice.dynamic.onnx    # 已驗證：修正動態長度後的音色模型
│   └── vec-768-layer-12.onnx  # 已提供：v2 內容編碼器
├── tests/
│   ├── test_conversion.py    # F0、模型契約、分段、速率與驗證
│   └── test_gui.py           # Qt 工作流程、按鈕、取消與關閉
├── README.md                 # 一般使用說明
├── README_CHATGPT.md         # 本文件
├── .gitignore               # 排除模型檔與 .venv
├── main.ipynb                # 原本的學習 notebook
├── p0002-7.mp3               # 原本的示範音檔
└── output.wav               # 原本既有輸出；不是新模型的驗收結果
```

本次擴充沒有修改原本 notebook 或音檔。repository 裡可能有使用者既有的未提交修改，不能把全部 git diff 都解讀成此次功能造成的變更。

## 4. 架構總覽

```text
                          app.py / PySide6
                      VoiceChanger.play_converted()
                                  │
                        ConversionWorker / QThread
                                  │
                       VoiceConverter.convert()
                         │                  │
                      backend=dsp       backend=rvc
                         │                  │
                 audio_engine         ModelConfig.load()
                 .convert_voice()           │
                         │             RVCPipeline.run()
                  pitch_shift               │
                  change_speed        16 kHz + 高通 + 分段
                         │                  │
                         │          ┌───────┴─────────┐
                         │          │                 │
                         │   ContentEncoder      extract_f0
                         │   ContentVec ONNX     librosa.pyin
                         │   內容特徵            F0 + Pitch
                         │          │                 │
                         │          └───────┬─────────┘
                         │             RVCModel.infer()
                         │             RVC ONNX graph
                         │                  │
                         │           去 padding + 接合
                         │                  │
                         │           change_speed()
                         └────────┬─────────┘
                          float32 + 峰值限制
                         ConvertedAudio(audio, sr)
                                  │
                         Qt converted Signal
                                  │
                        on_converted() / 主執行緒
                                  │
                         play_audio() → sd.play()
```

圖中的 ContentVec 和 F0 是兩個概念分支，但目前程式依序執行，沒有平行跑這兩個步驟。

## 5. 「模型架構」與「應用程式架構」要分開理解

本專案自己實作的是推論管線和 ONNX 介面，沒有在 Python 裡重新定義完整 RVC 神經網路。

| 元件 | 本專案知道/控制的部分 | 本專案沒有提供的部分 |
|---|---|---|
| ContentVec | 載入 ONNX、16 kHz waveform 輸入、256/768 維特徵輸出 | 訓練資料、模型內部完整層數與權重 |
| pYIN | librosa 的音高估計參數、voiced 判定、F0 後處理 | 不涉及神經網路權重 |
| RVC 音色模型 | 六個輸入 tensor、speaker ID、輸出 waveform | 訓練流程、訓練資料、個別模型內部 graph 的詳細組成 |

概念上，內容特徵攜帶語音內容相關的表示；F0 指示基頻與有聲/無聲區域；RVC 音色模型根據這些條件合成 waveform。內容特徵不是文字，也不保證完全去除原說話者資訊。F0 不等於音色或 formant。

如果要解說實際載入模型的 encoder、flow、decoder、vocoder、層數、參數量或訓練 loss，必須另外檢視該模型的 ONNX graph、匯出程式及模型文件。不能單憑本專案的 `RVCModel` wrapper 宣稱已確認這些內部細節。這裡也沒有外接一個獨立 vocoder 模組：應用程式從 RVC ONNX 的第一個輸出直接取得 waveform。

## 6. 主要 API 與物件生命週期

### VoiceConverter.convert

目前呼叫介面：

```python
convert(
    audio, sr, pitch=0, speed=1.0,
    backend="rvc",
    config_path=ROOT / "models/config.json",
    progress=lambda text: None,
    cancel=None,
) -> ConvertedAudio
```

- `audio`：一維 mono waveform，進入函式後轉成 NumPy float32。
- `sr`：輸入音訊取樣率，GUI 來源為 librosa.load 回傳值。
- `pitch`：半音，允許 -12 到 +12；GUI 給整數，CLI 可給浮點數。
- `speed`：0.5 到 1.5，大於 1 會縮短音訊。
- `progress`：接收字串的 callback；GUI 連接 Qt Signal，CLI 使用 print。
- `cancel`：具有 `is_set()` 的事件；GUI 使用 threading.Event。

輸入驗證包括：一維、非空、有限數值；sr > 0；Pitch 和 Speed 的有限值與範圍。sr 目前沒有完整的型別/有限值檢查，直接呼叫 API 的人應提供正常的正整數取樣率。

回傳型別：

```python
@dataclass
class ConvertedAudio:
    audio: np.ndarray
    sample_rate: int
```

waveform 與取樣率一起回傳，是為了避免把 40 kHz 的模型輸出誤用原檔 48 kHz 播放，造成速度與音高錯誤。

### 模型快取

每個 GUI 視窗建立一個 VoiceConverter。VoiceConverter 保留最近一個 `_pipeline` 與 `_config`，設定相同時重用 ONNX session；設定不同才重新建立。

ModelConfig 是 frozen dataclass，包含解析後的兩個檔案路徑、取樣率與 speaker ID。比較的是這些欄位，沒有比較檔案內容、mtime 或 hash。因此同一路徑的模型被替換後，需重啟程式才會確保重新載入。這不是多模型快取，也沒有提供共用實例同時多路轉換的同步保證。

## 7. 從 Load Audio 到 Play Converted 的逐步呼叫

1. GUI 建構時若 `p0002-7.mp3` 存在，呼叫 `open_audio()`；也可從檔案選擇器載入。
2. `audio_engine.load_audio(path)` 呼叫 `librosa.load(path, sr=None, mono=True)`，保留原始 sr、混成 mono，並檢查空資料/非有限值。
3. GUI 保存 `self.audio` 與 `self.sr`。載入新音檔成功後停止先前播放；載入失敗會顯示錯誤，舊音訊仍保留。
4. `play_converted()` 確認已有音訊、目前沒有另一個 worker，停止播放，停用可改變輸入的控制項。
5. 將目前 waveform、sr、Pitch、Speed、backend、JSON 路徑組成 arguments。waveform 是同一個陣列的參照，不是為 worker 額外做完整拷貝；轉換期間 GUI 不允許重新載入。
6. 建立 ConversionWorker，連接 progress、converted、failed、finished，再 start。
7. worker.run 呼叫 `VoiceConverter.convert()`。
8. DSP 模式走舊的 librosa 音效流程；RVC 模式讀設定、載入或重用 RVCPipeline，然後 run。
9. 得到結果後，若未取消，worker 發送 converted Signal。
10. 主執行緒的 `on_converted()` 再檢查一次取消狀態，呼叫 `play_result()`。
11. `play_result()` 將 result.audio 和 result.sample_rate 交給 `audio_engine.play_audio()`，內部呼叫 sounddevice.play。
12. worker 結束時 `on_finished()` 清除 worker 並恢復控制項。轉換完成不等於播放完成；sounddevice.play 是非阻塞的。

音檔載入仍在 GUI 主執行緒。背景 QThread 主要解決的是特徵處理與轉換卡住 GUI，並未把所有 I/O 都非同步化。

## 8. RVC 前處理與取樣率

`RVCPipeline.run()` 先做：

```python
audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
audio = audio / max(1.0, max(abs(audio)) / 0.95)
```

峰值超過 0.95 才衰減，不會將安靜音訊自動放大。接著當音訊長度 > 32 samples 時，使用五階 Butterworth、48 Hz 高通，透過 sosfiltfilt 做雙向濾波。這是離線前處理，不是可直接套用到即時串流的因果濾波器。

管線有三個不同概念：

| 符號 | 意義 | 例子 |
|---|---|---|
| 輸入 sr | 原始音檔的取樣率 | 48000 Hz |
| 分析 sr | ContentVec 和 pYIN 使用的取樣率 | 固定 16000 Hz |
| target_sr | RVC 合成 waveform 的取樣率 | 32000 / 40000 / 48000 Hz |

取樣率轉換不等同於 Speed 調整：重採樣的目的在保持時間長度下改變 samples 的密度。

## 9. 分段、padding 與時間對齊

以 16 kHz 分析 waveform 為基準：

| 參數 | samples | 時間 |
|---|---:|---:|
| chunk | 128000 | 8.00 秒 |
| overlap | 1600 | 0.10 秒 |
| stride = chunk - overlap | 126400 | 7.90 秒 |
| 左 padding | 8000 | 0.50 秒 |
| 右 padding | 8320 | 0.52 秒 |

右邊除了 0.5 秒外還多 320 samples（20 ms），為編碼器邊界與時間格數預留空間。原本的簡短說明把它概括成左右 500 ms；此表是目前程式的精確值。

每段取 `core = audio[start:start + 128000]`，再將這一段本身做反射 padding。這個 context 是段內反射，不是從原始音檔額外取得真正的前後文；core 只有一個 sample 時改用 edge padding。

```python
context = np.pad(core, (8000, 8320), mode="reflect")
content = encoder.extract(context)
frames = min(content.shape[1], len(context) // 160)
```

最後一段可以比 8 秒短。管線不依靜音位置切段、不做 VAD，也沒有用 SOLA 尋找相位最佳對齊點。

## 10. ContentVec：內容特徵分支

`ContentEncoder` 接收已建立的 ONNX session。

目前輸入契約：

```text
mono waveform [N]
    ↓ 增加 batch/channel 維度
[1, 1, N]，16 kHz，float32 或依模型輸入轉為 float16
    ↓ session.run(None, {input_name: waveform})[0]
[1, Tc, C]，C 必須是 256 或 768
    ↓ np.repeat(features, 2, axis=1)
[1, 2*Tc, C]
```

`input_name` 從模型輸入 metadata 取得，沒有硬編碼 ContentVec 的輸入名稱。建構時只檢查輸入數量為 1、rank 為 3；真正是否符合 `[1,1,N]` 仍須由相容模型與 runtime 共同保證。

常見配對是 RVC v1 使用 256 維、v2 使用 768 維。此程式本身沒有讀取獨立的 version 欄位，而是檢查 encoder 輸出的維度，並在 RVC phone 最後一維為靜態整數時檢查它們相符。

時間上假設 ContentVec 約 20 ms/frame，而 RVC 需要 10 ms/frame。因此將每一格特徵重複兩次，這是 nearest-repeat 對齊，不是新的神經網路層，也沒有對內容特徵做線性插值。

這裡的 Tc 必須採用實際模型輸出的時間長度，不能保證永遠等於 N/320；編碼器的邊界行為可能影響格數。

## 11. F0：音高分支

`extract_f0(context, frames, semitones)` 使用：

```python
librosa.pyin(
    context,
    sr=16000,
    fmin=50,
    fmax=1100,
    frame_length=1024,
    hop_length=160,
    fill_na=0.0,
)
```

- frame_length=1024：每個分析視窗長 64 ms。
- hop_length=160：相鄰分析格間隔 10 ms。
- `voiced` 判斷為 false 的 F0 設成 0 Hz。
- F0 格數少於 frames 時尾端補 0，多於 frames 時裁切。
- 不需要 RMVPE 權重；目前也沒有 RMVPE 呼叫。

音高調整：

```text
F0_shifted = F0_original × 2^(semitones / 12)
```

例：220 Hz，Pitch=+12 → 440 Hz；Pitch=-12 → 110 Hz。F0=0 的無聲格乘上倍率後仍為 0。

RVC 同時需要兩種 pitch 表示：

```text
pitchf：連續 F0，Hz，shape [1,T]
pitch ：量化的 mel bins，int64，shape [1,T]
```

量化公式依照目前程式：

```text
mel(f) = 1127 × ln(1 + f/700)
low = mel(50)
high = mel(1100)
pitch = round(clip((mel(F0_shifted)-low) × 254/(high-low) + 1, 1, 255))
```

無聲格會得到 pitch=1、pitchf=0。50–1100 Hz 是估計階段範圍；Pitch 調整後的連續 pitchf 沒有再次限制到這個範圍，只有量化 bins 會 clip。極端 Pitch 的實際品質要用真實模型驗證。

## 12. RVCModel：真正的聲音模型推論

### Session 設定

`open_session()` 使用 ONNX Runtime：

```python
options.intra_op_num_threads = 4
providers = ["CPUExecutionProvider"]
```

ContentVec 與 RVC 分別建立 session。這個 4 是每個 session 的算子內部執行緒設定，不是四個同時執行的音訊工作。即使機器有 NVIDIA GPU，這份程式仍指定 CPU，沒有自動切 CUDA。

### 六個必要輸入

模型輸入名稱集合必須**完全等於**下表，順序則不重要，因為透過字典按名稱傳入：

| 名稱 | Shape | 一般型別 | 內容 |
|---|---|---|---|
| phone | [1,T,C] | float32/float16 | ContentVec 特徵，C=256 或 768 |
| phone_lengths | [1] | int64 | 值為 T |
| pitch | [1,T] | int64 | 1–255 的 F0 bins |
| pitchf | [1,T] | float32/float16 | F0 Hz，無聲=0 |
| ds | [1] | int64 | speaker_id |
| rnd | [1,192,T] | float32/float16 | 標準常態隨機值 |

各 tensor 最後依模型 metadata 宣告的輸入型別轉換。目前型別 mapping 只處理 tensor(float)、tensor(float16)、tensor(int64)。未知型別沒有自訂的相容處理。

`rnd` 每次由新的 NumPy default_rng 生成，沒有固定 seed，所以同一段音訊重跑也不保證逐 sample 相同。它是模型生成過程使用的噪聲條件，不是直接加到喇叭輸出的雜訊效果。

核心推論只有這個呼叫：

```python
audio = session.run(None, feed)[0]
audio = np.asarray(audio, dtype=np.float32).reshape(-1)
```

程式取第一個模型輸出並攤平成一維。它假設該輸出是單一音訊 waveform，沒有額外的多聲道或多 batch 語意檢查。

### 輸出長度檢查

RVC 的時間格頻率假設為 100 frames/s，因此：

```text
expected_samples = T × sample_rate // 100
```

若實際輸出與預期相差超過 sample_rate//100，也就是一格 10 ms，會報錯。這可抓出常見取樣率配置錯誤，但不是完整的模型相容性或音質驗證。

如果模型 metadata 有 `config`，程式另假設它是 JSON 序列：最後一個值是取樣率、倒數第三個值是說話者數量，據此核對設定。沒有這份 metadata 時無法預先驗證 speaker ID 的上界，runtime 仍可能因超出範圍而失敗。

## 13. 去 padding、crossfade 與輸出後處理

每段推論後，以 target_sr 對應的時間位置寫回最終陣列：

```text
left  = round(start × target_sr / 16000)
right = min(total, round((start + len(core)) × target_sr / 16000))
trim  = target_sr // 2
converted = waveform[trim : trim + (right-left)]
```

這會跳過左側 500 ms 合成結果，保留核心區間的長度；右側 padding 與額外 20 ms 不寫回輸出。若保留下來的長度不足就報錯。

Crossfade 用的是線性權重：非第一段的開頭 100 ms 從 0 升到 1；非最後段的結尾 100 ms 從 1 降到 0。程式累加：

```text
output[left:right]  += converted × window
weights[left:right] += window
final = output / maximum(weights, 1e-8)
```

這是 overlap-add，不是等功率 crossfade，也沒有做相位對齊。分段可以限制每次模型推論的長度，但整份輸入、輸出與權重陣列仍在記憶體裡，因此不是固定記憶體大小的串流處理。

回到 VoiceConverter 後：

1. Speed != 1 時呼叫 librosa.effects.time_stretch，作用在合成後的整段 waveform。
2. 轉成 float32，拒絕空結果、NaN/Inf。
3. 峰值 > 0.99 時整段乘上 0.99/peak；不會把安靜音訊放大。
4. 回傳 ConvertedAudio。

沒有 RMS matching、降噪模型、響度標準化或額外靜音保護。pYIN 會把無聲 F0 設成 0，但這不代表管線強制將模型對應區段輸出設成全零。

## 14. 具體的 shape / 時間範例

假設輸入為 1 秒、48 kHz、mono，模型是 40 kHz、768 維：

```text
輸入 waveform                    [48000]
重採樣 16 kHz                    [16000]
加左右 padding：8000 + 8320       [32320]
ContentVec 輸入                  [1,1,32320]
ContentVec 輸出                  [1,Tc,768]（Tc 由模型決定）
repeat ×2                        [1,2*Tc,768]
T = min(2*Tc, 32320//160)         min(2*Tc,202)
phone                            [1,T,768]
pitch、pitchf                    [1,T]
rnd                              [1,192,T]
RVC 原始輸出                     約 T*400 samples
裁掉左 20000 samples，保留 core   [40000]
Speed=1.25                       約 [32000]
播放 sr                          40000 Hz
最終播放時間                     約 0.8 秒
```

不能把示例裡的 Tc 或 T 寫死。測試模型的 AveragePool 與真正 ContentVec 的邊界行為也可能不同。

## 15. DSP 模式與 AI 模式的精確差別

原本的 `audio_engine.convert_voice()` 仍只做 DSP：

```text
如果 Pitch != 0 → librosa.effects.pitch_shift
如果 Speed != 1 → librosa.effects.time_stretch
```

新的 `VoiceConverter.convert()` 是模式選擇入口。兩者名稱相近但不是同一個函式，解說時要寫清楚模組名稱。

AI 模式不先對輸入 waveform 做 pitch_shift；Pitch 改的是送進 RVC 的 F0。兩種模式的 Speed 都使用 time_stretch。Speed 調整的目標是改時長而盡量保持音高，不是直接把播放 sr 乘上倍率。

AI 模式即使 Pitch=0、Speed=1，仍會經過神經網路重新合成，不代表等同於 Play Original。

## 16. 執行緒、Stop、關閉與錯誤

### 控制項與工作狀態

開始轉換後，停用 Load Audio、模式、設定、Pitch、Speed、Play Original 和 Play Converted。Stop 保持可用。同一視窗每次只允許一個 worker。

### 合作式取消

Stop 先呼叫 sounddevice.stop，再設置 worker.cancel。`check_cancel()` 在管線若干階段檢查 Event，成立時丟出 ConversionCancelled。

目前檢查點包括：VoiceConverter 分支前；每個 chunk 開始；ContentVec 後；F0 後；整條 pipeline 結束；Speed 前後的相關邊界。不是每一行都有檢查，也不會中止正在執行的 session.run、pYIN、time_stretch 或模型載入。

worker 對 ConversionCancelled 不顯示錯誤，只停止送出結果。即使 converted Signal 已排入 Qt 事件佇列，主執行緒的 on_converted 仍再次檢查 cancel，避免 Stop 後又開始播放。

### 關閉視窗

若正在轉換，closeEvent 設 closing=True、要求停止、暫時 ignore 關閉事件。worker 完成後 on_finished 再次呼叫 close。這避免直接銷毀仍在執行的 QThread，但慢的模型步驟會延後真正關閉。

### 錯誤顯示

一般轉換例外會變成 `類別名稱: 訊息`，經 failed Signal 顯示在狀態列。播放裝置錯誤在 play_result 捕捉。沒有錯誤日誌檔或 GUI stack trace 面板。

播放完成沒有獨立的 GUI 通知；狀態文字可能仍停留在「播放中」，直到下一個操作。這是目前 UI 行為，不應描述成完整的播放狀態機。

## 17. 模型設定與相容性

目前 models/config.json：

```json
{
  "voice_model": "voice.dynamic.onnx",
  "contentvec": "vec-768-layer-12.onnx",
  "sample_rate": 40000,
  "speaker_id": 0
}
```

相對路徑以 JSON 所在目錄為基準。GUI 預設 JSON 位於專案 models 目錄，也可選其他 JSON。

必要欄位為 voice_model、contentvec、sample_rate。speaker_id 缺省為 0。程式檢查檔案存在、副檔名為 .onnx、sample_rate 在 32000/40000/48000 之中、speaker_id 非負。類別欄位雖有 sample_rate=40000 的預設，但 ModelConfig.load 的 JSON 讀取仍要求提供 sample_rate。

權重準備方式：

- 音色模型：提供符合六輸入契約的 RVC v1/v2、F0-enabled ONNX。若只有 .pth，需要用相容 RVC 匯出器真正匯出，不能只改檔名。
- v1 內容編碼器：vec-256-layer-9.onnx。
- v2 內容編碼器：vec-768-layer-12.onnx。
- 本版不需要 rmvpe.onnx，不讀 .index。

參考位置（不是已下載或已完成相容驗收的模型）：

- [RVC 官方 ONNX 匯出器](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion/blob/develop/rvc/modules/onnx/export.py)
- [ContentVec ONNX 模型檔案頁](https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/tree/main)
- [ONNX 介面參考實作](https://github.com/dev6699/rvc-onnx)
- [w-okada 架構參考](https://github.com/w-okada/voice-changer)

不是所有 ONNX 都相容。五輸入 feats/p_len/sid 格式、無 F0 模型、其他 vocoder、固定時間長度 graph、不同 encoder 輸出排列，不在目前支援契約內。部分不相容情形由我們的檢查提前擋下，其他情形仍會在 runtime.run 時報錯。

## 18. 環境、啟動與不經 GUI 的使用

目前已使用的環境是 Windows、conda `xin`、Python 3.10。這台電腦預設 python 是另一個 Python 3.14，因此本機命令使用完整路徑。

```powershell
cd C:\Users\UUU\Documents\GitHub\Personal-Project\voice
.\start.ps1
```

start.ps1 先切到腳本目錄，優先使用使用者目錄下 anaconda3/envs/xin/python.exe；找不到才退回 PATH 的 python。

明確啟動方式：

```powershell
& C:\Users\UUU\anaconda3\envs\xin\python.exe -X utf8 app.py
```

主要套件版本：PySide6 6.11.2、librosa 0.11.0、sounddevice 0.5.6、soundfile 0.14.0、numpy 2.2.6、onnxruntime 1.23.2。原有 requirements 同時包含 matplotlib 3.10.9 與 ipykernel 7.3.0。scipy 由 librosa 的相依關係安裝，pipeline 直接使用其濾波功能。測試另需 onnx 1.20.1。

以下命令從專案根目錄執行，input.wav/converted.wav 是示例檔名：

```powershell
& C:\Users\UUU\anaconda3\envs\xin\python.exe -m pip install -r requirements.txt

# 真實 RVC：先準備相容的模型
& C:\Users\UUU\anaconda3\envs\xin\python.exe -X utf8 convert_file.py input.wav converted.wav --config models/config.json --pitch 3 --speed 1.0

# 不需權重的 DSP 比較
& C:\Users\UUU\anaconda3\envs\xin\python.exe -X utf8 convert_file.py input.wav converted.wav --backend dsp --pitch 3 --speed 1.1

# 自動化測試
& C:\Users\UUU\anaconda3\envs\xin\python.exe -m pip install -r requirements-test.txt
& C:\Users\UUU\anaconda3\envs\xin\python.exe -X utf8 -m unittest discover -s tests -v
```

CLI 與 GUI 共用 VoiceConverter。CLI 以 soundfile.write 儲存模型回傳的取樣率；它沒有 GUI 的 Event 取消流程或友善例外面板。GUI 目前只有播放，沒有 Save Converted 按鈕。

## 19. 測試到底驗證了什麼

上一輪實作驗證已有 9 個 unittest 通過，本輪是依原始碼撰寫文件，沒有把文件修改當成一次新的模型音質驗收。

| 測試函式 | 驗證內容 |
|---|---|
| test_f0_pitch_and_silence | 220 Hz 正弦波、+12 半音倍頻、bins 範圍、靜音 F0 |
| test_validation_missing_weights_and_cancel | 空輸入、Speed=0、缺設定、開始前已取消 |
| test_real_onnx_runtime_with_synthetic_contract_graphs | 真正 ORT 執行人工 graph，確認結果長度/取樣率，拒絕錯誤取樣率 |
| test_overlap_add_and_partial_tail | 7.95、8.01、16.05 秒的接合與尾段；F0 被替換為常數以隔離接合測試 |
| test_dsp_speed_and_rate | DSP Speed=1.25 後的長度與原始 sr |
| test_dsp_button_and_original | GUI 原音/轉換按鈕、控制項恢復、播放呼叫 |
| test_missing_model_is_visible_and_recovers | 缺 JSON 顯示錯誤並恢復操作 |
| test_ai_button_reaches_onnx_and_plays_model_sample_rate | GUI 按鈕到人工 ONNX graph，再用 40 kHz 呼叫播放 |
| test_stop_discards_result_and_close_waits | 延遲工作下 Stop/關閉後不播放過期結果 |

人工測試 graph 的內容：

- 假 ContentVec：AveragePool → Transpose → Tile，產生 shape 正確的 768 維資料，沒有學到語音內容。
- 假 RVC：將 pitchf 每格重複 400 次並乘以 0.001，產生預期 40 kHz 長度，沒有真正生成音色。
- 即使 graph 宣告六個輸入，也不表示它在內部實際使用全部輸入。因此通過測試證明的是介面與流程，不能證明真實模型所有 conditioning 的行為。
- GUI 測試把 play_audio 替換成 mock，不會真的從喇叭播放；另有一次獨立實機播放驗證。

實機驗證紀錄：PySide6 視窗成功顯示；原有 70.52 秒/48 kHz MP3 成功載入；擷取一秒人聲，以 DSP Pitch=+3、Speed=1.1 處理並降低音量後，sounddevice 播放完成；預設裝置的 40 kHz 設定檢查通過；pip check 無相依衝突。這些是早期實機紀錄；另已完成本文開頭記載的真實 RVC 全長轉換，尚無主觀音質評分。

## 20. 限制、容易誤解之處與後續擴充位置

已知界線：

- 已完成真實權重端到端執行，但尚無主觀音質驗收與正式效能基準。
- 沒有訓練、模型下載器、.pth 直接推論、FAISS 檢索、RMVPE、GPU 切換。
- 沒有麥克風錄音、虛擬音效裝置、即時串流、SOLA 或按靜音切段。
- 沒有音訊內容轉文字步驟，沒有 ASR/TTS。
- 沒有獨立 Formant 控制；不能把 F0 倍率說成 formant shifting。
- 只接受 mono 轉換；讀檔時會混合聲道。
- 第一次載入模型、pYIN 初始化或推論可能較慢，沒有延遲保證。
- Stop 不是立即中斷計算；只是立即停止播放並要求在檢查點取消。
- 不支援任意 ONNX 模型，CPU 上 FP16 是否可用仍取決於個別算子。
- 模型快取不偵測檔案內容替換；輸出隨機性尚未提供 seed 控制。

合理的擴充位置（尚未實作）：

| 想新增什麼 | 最直接的修改位置 |
|---|---|
| 換 F0 演算法，例如 RMVPE | features.py，保持 pitch/pitchf 回傳契約 |
| 換相容的 RVC 權重 | models/config.json；相同路徑換檔需重啟 |
| 新的 ONNX tensor 命名或模型格式 | rvc_model.py，先明確定義另一個 adapter |
| 可選 FAISS index retrieval | pipeline.py，在內容特徵與 RVCModel.infer 之間 |
| 儲存轉換結果 | app.py；CLI 已有 soundfile.write 範例 |
| GPU provider | rvc_model.open_session 及設定，不需把細節塞到 GUI 轉換函式 |
| 即時模式 | 需要新增 audio stream、緩衝與低延遲設計，不能只把檔案讀取換成麥克風 |

## 21. 建議 ChatGPT 如何使用這份文件

這份文件可獨立提供專案上下文，但不是完整原始碼的替代品。解說目前流程時，可按這個順序閱讀：

```text
app.py: play_converted
→ ConversionWorker.run
→ voice_converter.py: VoiceConverter.convert
→ pipeline.py: RVCPipeline.run
→ features.py: ContentEncoder.extract / extract_f0
→ rvc_model.py: RVCModel.infer
→ app.py: on_converted / play_result
→ audio_engine.py: play_audio
```

後續討論時應持續區分：已存在的程式行為、外部模型需滿足的假設、人工 graph 測試結果，以及尚未驗證的真實音質。需要直接修改程式時，應取得相關原始碼的最新版本，特別是上述六個主要模組與 models/config.json。

可在上傳本文件後附上這段需求：

> 這是我目前的 voice conversion 學習專案。請先依 README 重建現有架構，分清楚 DSP、F0、ContentVec、RVC ONNX 各自負責什麼，再沿著 Play Converted 的實際資料流逐層教我。請區分目前已實作的功能與未來建議；如果要判斷真正神經網路的內部層或音質，先指出還需要哪些模型檔或原始碼。

