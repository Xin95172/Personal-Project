"""File decoding is isolated from the PCM mixer."""
from pathlib import Path
import numpy as np
import soundfile as sf


class FileAudioSource:
    def __init__(self, path, sample_rate=48000):
        self.path = Path(path)
        self.sample_rate = sample_rate
        self.pcm, original_rate = sf.read(self.path, dtype="float32", always_2d=True)
        if not self.pcm.size or not np.isfinite(self.pcm).all():
            raise ValueError("Audio is empty or contains invalid samples")
        if original_rate != sample_rate:
            import librosa
            self.pcm = librosa.resample(
                self.pcm, orig_sr=original_rate, target_sr=sample_rate, axis=0
            )
        self.pcm = np.ascontiguousarray(self.pcm, dtype=np.float32)

    def blocks(self, block_size=480, max_frames=None):
        """Decode once, then yield small PCM blocks (last block may be shorter).

        Whole-file resampling avoids discontinuities at block boundaries.
        Playback pacing is supplied by the output device's blocking write.
        """
        if block_size <= 0:
            raise ValueError("Block size must be positive")
        end = len(self.pcm) if max_frames is None else min(len(self.pcm), max_frames)
        for start in range(0, end, block_size):
            yield self.pcm[start:min(start + block_size, end)]
