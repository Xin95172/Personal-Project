"""Offline RVC pipeline: resample -> content/F0 -> neural synthesis -> overlap-add."""
import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt
from features import ContentEncoder, extract_f0
from rvc_model import RVCModel, open_session
from voice_converter import check_cancel


class RVCPipeline:
    def __init__(self, config):
        self.config = config
        self.model = RVCModel(config)
        self.encoder = ContentEncoder(open_session(config.contentvec))

    def run(self, audio, sr, semitones, progress, cancel):
        progress("輸入轉為 16 kHz，移除低頻雜訊…")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audio = audio / max(1.0, float(np.max(np.abs(audio))) / 0.95)
        if len(audio) > 32:
            audio = sosfiltfilt(butter(5, 48, "highpass", fs=16000, output="sos"), audio)
        target_sr = self.config.sample_rate
        total = round(len(audio) * target_sr / 16000)
        output = np.zeros(total, dtype=np.float32)
        weights = np.zeros(total, dtype=np.float32)
        # Eight-second chunks, 100 ms crossfade, 500 ms reflected context.
        chunk, overlap, pad = 128000, 1600, 8000
        starts = list(range(0, max(1, len(audio) - overlap), chunk - overlap))
        for number, start in enumerate(starts, 1):
            check_cancel(cancel)
            core = audio[start:start + chunk]
            context = np.pad(core, (pad, pad + 320), mode="reflect" if len(core) > 1 else "edge")
            progress(f"{number}/{len(starts)}：ContentVec 內容特徵…")
            content = self.encoder.extract(context)
            check_cancel(cancel)
            frames = min(content.shape[1], len(context) // 160)
            progress(f"{number}/{len(starts)}：pYIN F0 / Pitch…")
            pitch, hz = extract_f0(context, frames, semitones)
            check_cancel(cancel)
            progress(f"{number}/{len(starts)}：RVC 模型 inference…")
            waveform = self.model.infer(content[:, :frames], pitch, hz)
            left = round(start * target_sr / 16000)
            right = min(total, round((start + len(core)) * target_sr / 16000))
            trim = target_sr // 2
            converted = waveform[trim:trim + right - left]
            if len(converted) != right - left:
                raise ValueError("模型輸出長度不符；請檢查 sample_rate 與 ONNX 匯出格式。")
            window = np.ones(len(converted), dtype=np.float32)
            fade = min(target_sr // 10, len(window))
            if start:
                window[:fade] *= np.linspace(0, 1, fade)
            if start + chunk < len(audio):
                window[-fade:] *= np.linspace(1, 0, fade)
            output[left:right] += converted * window
            weights[left:right] += window
            if start + chunk >= len(audio):
                break
        check_cancel(cancel)
        return output / np.maximum(weights, 1e-8)
