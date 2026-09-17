import librosa
import sounddevice as sd
import numpy as np


def load_audio(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    if y.size == 0 or not np.isfinite(y).all():
        raise ValueError("音檔為空或包含無效數值。")
    return y, sr

def pitch_shift(y, sr, semitones):
    return librosa.effects.pitch_shift(
        y,
        sr=sr,
        n_steps=semitones
    )

def change_speed(y, rate):
    return librosa.effects.time_stretch(
        y,
        rate=rate
    )

def convert_voice(y, sr, pitch=0, speed=1.0):
    """Original DSP exercise. AI conversion is in VoiceConverter.convert()."""
    converted = y

    if pitch != 0:
        converted = pitch_shift(
            y,
            sr,
            pitch
        )

    if speed != 1.0:
        converted = change_speed(
            converted,
            speed
        )

    return converted

def play_audio(y, sr):
    sd.play(y, sr)

def stop_audio():
    sd.stop()
