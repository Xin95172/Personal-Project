"""Two independent inputs to RVC: content features and fundamental frequency."""
import librosa
import numpy as np


def extract_f0(audio_16k, frames, semitones):
    """pYIN: 10 ms frames, unvoiced=0 Hz; no additional pitch-model weights."""
    f0, voiced, _ = librosa.pyin(
        audio_16k, sr=16000, fmin=50, fmax=1100,
        frame_length=1024, hop_length=160, fill_na=0.0,
    )
    f0 = np.where(voiced, f0, 0.0)
    f0 = np.pad(f0, (0, max(0, frames - len(f0))))[:frames]
    f0 = (f0 * 2 ** (semitones / 12)).astype(np.float32)
    # RVC expects both continuous Hz and integer mel bins in [1, 255].
    mel = 1127 * np.log1p(f0 / 700)
    low, high = 1127 * np.log1p(np.array([50, 1100]) / 700)
    bins = np.rint(np.clip((mel - low) * 254 / (high - low) + 1, 1, 255))
    return bins.astype(np.int64)[None, :], f0[None, :]


class ContentEncoder:
    def __init__(self, session):
        self.session = session
        inputs = session.get_inputs()
        if len(inputs) != 1 or len(inputs[0].shape) != 3:
            raise ValueError("ContentVec 需要單一 [1,1,samples] 輸入；請使用 models/README.md 指定的匯出格式。")
        self.input = inputs[0]

    def extract(self, audio_16k):
        dtype = np.float16 if self.input.type == "tensor(float16)" else np.float32
        features = self.session.run(None, {
            self.input.name: audio_16k[None, None, :].astype(dtype)
        })[0]
        if features.ndim != 3 or features.shape[0] != 1 or features.shape[2] not in (256, 768):
            raise ValueError(f"ContentVec 輸出應為 [1,T,256/768]，收到 {features.shape}。")
        # ContentVec is 20 ms/frame; RVC is 10 ms/frame.
        return np.repeat(features, 2, axis=1)
