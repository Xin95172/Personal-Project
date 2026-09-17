"""Play a file through the independent PCM mixer."""
import argparse
from pathlib import Path
import math
import sounddevice as sd
from mixer import Channel, MixerEngine
from mixer.audio_source import FileAudioSource
from mixer.audio_output import AudioOutput


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", action="store_true", help="Use command-line playback instead of GUI")
    parser.add_argument("file", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "p0002-7.mp3")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", help="Output device index or unique name")
    parser.add_argument("--gain", type=float, default=1.0, help="Linear input gain")
    parser.add_argument("--master-gain", type=float, default=0.5)
    parser.add_argument("--mute", action="store_true")
    parser.add_argument("--block-size", type=int, default=480)
    parser.add_argument("--seconds", type=float, help="Limit playback for a smoke test")
    args = parser.parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return
    if not args.cli:
        from mixer.mixer_gui import launch_gui
        return launch_gui(args.file)
    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be finite and positive")
    for gain in (args.gain, args.master_gain):
        if not math.isfinite(gain) or gain < 0:
            parser.error("gains must be finite and non-negative")
    device = int(args.device) if args.device and args.device.isdecimal() else args.device
    source = FileAudioSource(args.file)
    engine = MixerEngine({"file": Channel(args.gain, args.mute)}, args.master_gain)
    limit = max(1, int(args.seconds * source.sample_rate)) if args.seconds else None
    blocks = frames = 0
    with AudioOutput(device, source.sample_rate, args.block_size) as output:
        print(f"Output: {output.info['name']} | 48000 Hz | {output.channels} channels")
        for pcm in source.blocks(args.block_size, limit):
            output.write(engine.process({"file": pcm}))
            blocks += 1
            frames += len(pcm)
    print(f"Played {frames} frames in {blocks} blocks; underflows: {output.underflows}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Playback stopped.")
    except (OSError, ValueError, RuntimeError, sd.PortAudioError) as exc:
        raise SystemExit(f"Mixer error: {exc}")
