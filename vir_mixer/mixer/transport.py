"""Device-independent file transports and synchronized stereo mixing."""
from dataclasses import dataclass, field
import numpy as np
from .mixer_engine import Channel, MixerEngine
from .audio_output import adapt_channels


@dataclass
class Track:
    pcm: np.ndarray
    channel: Channel = field(default_factory=Channel)
    cursor: int = 0
    playing: bool = False
    loop: bool = False

    def play(self):
        if self.cursor >= len(self.pcm):
            self.cursor = 0
        self.playing = True

    def stop(self):
        self.playing = False
        self.cursor = 0

    def seek(self, fraction):
        self.cursor = int(np.clip(fraction, 0, 1) * len(self.pcm))

    def pull(self, frames):
        block = np.zeros((frames, 2), dtype=np.float32)
        offset = 0
        while self.playing and offset < frames:
            if self.cursor >= len(self.pcm):
                if self.loop and len(self.pcm):
                    self.cursor = 0
                else:
                    self.playing = False
                    break
            count = min(frames - offset, len(self.pcm) - self.cursor)
            block[offset:offset + count] = self.pcm[self.cursor:self.cursor + count]
            self.cursor += count
            offset += count
        if self.cursor == len(self.pcm) and not self.loop:
            self.playing = False
        return block


class TransportMixer:
    def __init__(self):
        self.tracks = {}
        self.engine = MixerEngine(master_gain=0.5, smoothing=True)

    def add(self, key, pcm):
        if pcm.dtype != np.float32 or pcm.ndim != 2 or not len(pcm) or not np.isfinite(pcm).all():
            raise ValueError("Expected nonempty finite float32 PCM")
        self.tracks[key] = Track(adapt_channels(pcm, 2))
        self.engine.channels[key] = self.tracks[key].channel

    def remove(self, key):
        self.tracks.pop(key, None)
        self.engine.channels.pop(key, None)
        self.engine._previous.pop(key, None)

    def render(self, frames=480):
        if frames <= 0:
            raise ValueError("frames must be positive")
        if not self.tracks:
            self.engine.meters = {}
            self.engine.master_peak = np.zeros(2)
            return np.zeros((frames, 2), dtype=np.float32)
        return self.engine.process({key: track.pull(frames) for key, track in self.tracks.items()})
