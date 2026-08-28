import librosa
import sounddevice as sd


def load_audio(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr


def pitch_shift(y, sr, semitones):
    return librosa.effects.pitch_shift(
        y,
        sr=sr,
        n_steps=semitones
    )


def convert_voice(y, sr, pitch=0):
    converted = pitch_shift(
        y,
        sr,
        pitch
    )

    return converted


def play_audio(y, sr):
    sd.play(y, sr)


def stop_audio(self):
    sd.stop()