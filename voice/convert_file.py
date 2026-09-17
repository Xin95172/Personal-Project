"""Run the very same converter without Qt, useful for debugging model inference."""
import argparse
import soundfile as sf
from audio_engine import load_audio
from voice_converter import VoiceConverter, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--config", default=str(ROOT / "models/config.json"))
    parser.add_argument("--backend", choices=("rvc", "dsp"), default="rvc")
    parser.add_argument("--pitch", type=float, default=0)
    parser.add_argument("--speed", type=float, default=1)
    args = parser.parse_args()
    audio, sr = load_audio(args.input)
    result = VoiceConverter().convert(audio, sr, args.pitch, args.speed,
                                     backend=args.backend, config_path=args.config,
                                     progress=print)
    sf.write(args.output, result.audio, result.sample_rate)
    print(f"Saved {args.output}: {result.sample_rate} Hz")


if __name__ == "__main__":
    main()
