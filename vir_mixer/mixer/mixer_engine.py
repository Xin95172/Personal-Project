"""Mix frame-major float32 PCM blocks independently of their sources."""
from dataclasses import dataclass
from typing import Mapping
import numpy as np


def validate_gain(value):
    if not np.isfinite(value) or value < 0:
        raise ValueError("Gain must be finite and non-negative")


@dataclass
class Channel:
    gain: float = 1.0
    mute: bool = False
    solo: bool = False
    pan: float = 0.0


class MixerEngine:
    def __init__(self, channels=None, master_gain=1.0, smoothing=False):
        self.channels = dict(channels or {})
        self.master_gain = master_gain
        self.master_mute = False
        self.smoothing = smoothing
        self._previous = {}
        self._master_previous = None
        self.meters = {}
        self.master_peak = np.zeros(2, dtype=np.float32)

    def _apply_gain(self, pcm, target, previous):
        if self.smoothing and previous is not None and len(pcm):
            ramp = np.linspace(0, 1, len(pcm), dtype=np.float32)[:, None]
            return pcm * (previous + ramp * (target - previous))
        return pcm * target

    def process(self, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        """Inputs share (frames, speakers) shape and sample rate; never mutate them.

        A logical mixer channel can contain stereo PCM. Missing inputs contribute
        silence. Caller pads unequal final blocks before mixing multiple sources.
        """
        validate_gain(self.master_gain)
        if not inputs:
            raise ValueError("At least one PCM block is required")
        first = next(iter(inputs.values()))
        if first.ndim != 2:
            raise ValueError("PCM must have shape (frames, speakers)")
        result = np.zeros(first.shape, dtype=np.float32)
        solo_active = any(ch.solo for ch in self.channels.values())
        self.meters = {}
        for name, pcm in inputs.items():
            if pcm.shape != result.shape or pcm.dtype != np.float32:
                raise ValueError("PCM blocks must have equal shapes and float32 dtype")
            if not np.isfinite(pcm).all():
                raise ValueError("PCM contains non-finite samples")
            channel = self.channels[name]
            validate_gain(channel.gain)
            if not np.isfinite(channel.pan) or not -1 <= channel.pan <= 1:
                raise ValueError("Pan must be between -1 and 1")
            target = np.full(pcm.shape[1], channel.gain, dtype=np.float32)
            # Stereo balance: center preserves L/R, either edge attenuates the opposite side.
            if pcm.shape[1] == 2:
                target[0] *= 1 - max(0, channel.pan)
                target[1] *= 1 + min(0, channel.pan)
            if channel.mute or (solo_active and not channel.solo):
                target[:] = 0
            mixed = self._apply_gain(pcm, target, self._previous.get(name))
            self._previous[name] = target
            self.meters[name] = np.max(np.abs(mixed), axis=0) if len(mixed) else np.zeros(pcm.shape[1])
            result += mixed
        target = np.full(first.shape[1], 0 if self.master_mute else self.master_gain, dtype=np.float32)
        result = self._apply_gain(result, target, self._master_previous)
        self._master_previous = target
        self.master_peak = np.max(np.abs(result), axis=0) if len(result) else np.zeros(first.shape[1])
        # Basic hard clipping protection; this is not a compressor/limiter.
        np.clip(result, -1.0, 1.0, out=result)
        return result
