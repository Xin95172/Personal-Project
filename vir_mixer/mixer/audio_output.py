"""PortAudio output with explicit mono/stereo adaptation."""
import numpy as np
import sounddevice as sd


def adapt_channels(pcm, channels):
    if pcm.shape[1] == channels:
        return np.ascontiguousarray(pcm)
    if pcm.shape[1] == 1 and channels == 2:
        return np.repeat(pcm, 2, axis=1)
    if pcm.shape[1] == 2 and channels == 1:
        return np.ascontiguousarray(pcm.mean(axis=1, keepdims=True))
    raise ValueError("MVP supports mono and stereo sources only")


class AudioOutput:
    def __init__(self, device=None, sample_rate=48000, block_size=480):
        self.info = sd.query_devices(device, "output")
        self.channels = min(2, int(self.info["max_output_channels"]))
        if self.channels < 1:
            raise ValueError("Selected device has no output channels")
        sd.check_output_settings(device=device, samplerate=sample_rate,
                                 channels=self.channels, dtype="float32")
        self.stream = sd.OutputStream(device=device, samplerate=sample_rate,
                                      channels=self.channels, dtype="float32",
                                      blocksize=block_size, latency="high")
        self.underflows = 0

    def __enter__(self):
        self.stream.__enter__()
        return self

    def write(self, pcm):
        self.underflows += bool(self.stream.write(adapt_channels(pcm, self.channels)))

    def __exit__(self, *args):
        return self.stream.__exit__(*args)
