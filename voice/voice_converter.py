"""GUI-independent entry point: select a backend and return audio WITH its rate."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent


class ConversionCancelled(Exception):
    pass


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise ConversionCancelled()


@dataclass(frozen=True)
class ModelConfig:
    voice_model: Path
    contentvec: Path
    sample_rate: int = 40000
    speaker_id: int = 0

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        config = cls(
            (path.parent / data["voice_model"]).resolve(),
            (path.parent / data["contentvec"]).resolve(),
            int(data["sample_rate"]), int(data.get("speaker_id", 0)),
        )
        if config.sample_rate not in (32000, 40000, 48000):
            raise ValueError("sample_rate 必須是模型的 32000、40000 或 48000。")
        if config.speaker_id < 0:
            raise ValueError("speaker_id 不可為負數。")
        for model in (config.voice_model, config.contentvec):
            if not model.is_file():
                raise FileNotFoundError(f"找不到模型：{model}\n請參考 models/README.md。")
            if model.suffix.lower() != ".onnx":
                raise ValueError("此版本需要 ONNX 模型，不能直接載入 .pth。")
        return config


@dataclass
class ConvertedAudio:
    audio: np.ndarray
    sample_rate: int


class VoiceConverter:
    def __init__(self):
        self._pipeline = None
        self._config = None

    def convert(self, audio, sr, pitch=0, speed=1.0, backend="rvc",
                config_path=ROOT / "models/config.json", progress=lambda text: None,
                cancel=None):
        from audio_engine import convert_voice, change_speed
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError("音訊必須是非空、有限數值的 mono waveform。")
        if sr <= 0 or not np.isfinite(speed) or not 0.5 <= speed <= 1.5:
            raise ValueError("取樣率必須為正值，Speed 必須介於 0.5–1.5。")
        if not np.isfinite(pitch) or not -12 <= pitch <= 12:
            raise ValueError("Pitch 必須介於 -12–12 半音。")
        check_cancel(cancel)
        if backend == "dsp":
            progress("DSP 比較模式：音高與速度處理")
            result = convert_voice(audio, sr, pitch, speed)
            output_sr = sr
        elif backend == "rvc":
            from pipeline import RVCPipeline
            config = ModelConfig.load(config_path)
            if self._pipeline is None or config != self._config:
                progress("載入 ContentVec 與 RVC 模型…")
                self._pipeline = RVCPipeline(config)
                self._config = config
            result = self._pipeline.run(audio, sr, pitch, progress, cancel)
            output_sr = config.sample_rate
            check_cancel(cancel)
            if speed != 1:
                progress("調整輸出速度…")
                result = change_speed(result, speed)
        else:
            raise ValueError(f"未知模式：{backend}")
        check_cancel(cancel)
        result = np.asarray(result, dtype=np.float32)
        if result.size == 0 or not np.isfinite(result).all():
            raise ValueError("模型輸出空白或非有限數值。")
        peak = np.max(np.abs(result))
        if peak > 0.99:
            result *= 0.99 / peak
        return ConvertedAudio(result, output_sr)
