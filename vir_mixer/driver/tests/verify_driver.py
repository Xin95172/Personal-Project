"""Explicit installed-device tests: PCM16 stereo, restart, silence and sustained I/O.
No installation or system-setting changes. Reports are evidence, not certification.
"""
import argparse
import json
import tempfile
import threading
import time
from pathlib import Path
import numpy as np
import sounddevice as sd
from scipy.signal import correlate

RATE = 48000

def analyze(recorded, reference, threshold=.98):
    if len(recorded) < len(reference):
        raise AssertionError('Capture shorter than reference')
    scores, offsets, gains = [], [], []
    for c in range(2):
        ref = reference[:, c].astype(np.float64)
        wave = recorded[:, c].astype(np.float64)
        offset = int(np.argmax(correlate(wave, ref, mode='valid', method='fft')))
        segment = wave[offset:offset + len(ref)]
        if np.std(segment) == 0 or np.std(ref) == 0:
            raise AssertionError('Silence or constant signal cannot prove PCM transmission')
        score = float(np.corrcoef(segment, ref)[0, 1])
        gain = float(np.dot(segment, ref) / np.dot(ref, ref))
        if not np.isfinite(score) or score < threshold or not .90 <= gain <= 1.10:
            raise AssertionError(f'Channel {c}: correlation={score}, gain={gain}')
        scores.append(score)
        gains.append(gain)
        offsets.append(offset)
    if abs(offsets[0] - offsets[1]) > 1:
        raise AssertionError(f'Stereo timing mismatch: {offsets}')
    return dict(correlation=scores, gain=gains, offset_frames=offsets)


def find_devices():
    hosts = sd.query_hostapis()
    devices = list(sd.query_devices())
    def find(name, direction):
        matches = [i for i, d in enumerate(devices) if name in d['name']
                   and d[direction] >= 2 and hosts[d['hostapi']]['name'] == 'Windows WASAPI']
        if len(matches) != 1:
            raise RuntimeError(f'Need one WASAPI {name}; found {len(matches)}. Install the approved signed driver first.')
        return matches[0]
    return find('VirMixer Input', 'max_output_channels'), find('VirMixer Output', 'max_input_channels')


def trial(render, capture, seconds, seed, exclusive=False, silence=False):
    # Blocking capture in its own worker; no disk access in a PortAudio callback.
    # A temporary disk spool bounds memory even for the one-hour continuous test.
    statuses, errors = [], []
    stop = threading.Event()
    settings = sd.WasapiSettings(exclusive=exclusive)
    kwargs = dict(samplerate=RATE, channels=2, dtype='int16', blocksize=480,
                  extra_settings=settings)
    with tempfile.TemporaryDirectory(prefix='virmixer-test-') as folder:
        path = Path(folder) / 'capture.raw'
        with sd.InputStream(device=capture, **kwargs) as inp:
            def reader():
                try:
                    with path.open('wb') as f:
                        while not stop.is_set():
                            data, overflow = inp.read(480)
                            if overflow:
                                statuses.append('capture overflow')
                            f.write(data.tobytes())
                except Exception as exc:
                    errors.append(str(exc))
            worker = threading.Thread(target=reader, daemon=True)
            worker.start()
            try:
                if silence:
                    time.sleep(seconds)
                else:
                    rng = np.random.default_rng(seed)
                    with sd.OutputStream(device=render, **kwargs) as out:
                        for data in [np.zeros((RATE // 2, 2), dtype=np.int16)]:
                            if out.write(data): statuses.append('render underflow')
                        for _ in range(seconds):
                            data = rng.integers(-3000, 3001, (RATE, 2), dtype=np.int16)
                            if out.write(data): statuses.append('render underflow')
                        if out.write(np.zeros((RATE // 2, 2), dtype=np.int16)):
                            statuses.append('render underflow')
                    time.sleep(.25)
            finally:
                stop.set()
                worker.join(timeout=5)
                if worker.is_alive():
                    inp.abort()
                    worker.join(timeout=2)
                    raise RuntimeError('Capture worker did not stop normally')
        if errors or statuses:
            raise AssertionError(dict(errors=errors, glitches=statuses))
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError('No captured PCM')
        recorded = np.memmap(path, dtype=np.int16, mode='r').reshape(-1, 2)
        try:
            if silence:
                peak = int(np.max(np.abs(recorded[RATE // 2:].astype(np.int32))))
                if peak > 2: raise AssertionError(f'Capture with no producer is not silence: peak={peak}')
                return dict(silence_peak=peak)
            rng = np.random.default_rng(seed)
            results, next_start = [], 0
            for second in range(seconds):
                reference = rng.integers(-3000, 3001, (RATE, 2), dtype=np.int16)
                # Initial latency up to 2 seconds; then bounded alignment search.
                lo = max(0, next_start - RATE // 4)
                hi = min(len(recorded), next_start + RATE * (3 if second == 0 else 2))
                result = analyze(recorded[lo:hi], reference)
                absolute = lo + result['offset_frames'][0]
                if second and abs(absolute - next_start) > 2:
                    raise AssertionError(f'Discontinuous PCM at second {second}: {absolute - next_start} frames')
                next_start = absolute + RATE
                results.append(result)
            return dict(seconds=seconds, minimum_correlation=min(min(r['correlation']) for r in results),
                        initial_offset_frames=results[0]['offset_frames'])
        finally:
            recorded._mmap.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--seconds', type=int, default=2)
    parser.add_argument('--long-seconds', type=int, default=0)
    parser.add_argument('--exclusive', action='store_true')
    parser.add_argument('--report', type=Path, default=Path('driver/out/runtime-report.json'))
    args = parser.parse_args()
    if not 1 <= args.repeats <= 1000 or not 1 <= args.seconds <= 60 or not 0 <= args.long_seconds <= 3600:
        parser.error('repeats 1..1000; seconds 1..60; long-seconds 0..3600')
    report = dict(passed=False, mode='exclusive' if args.exclusive else 'shared', format='48000 PCM16 stereo', tests=[])
    try:
        render, capture = find_devices()
        report['devices'] = dict(render=dict(sd.query_devices(render)), capture=dict(sd.query_devices(capture)))
        report['tests'].append(trial(render, capture, 2, 0, args.exclusive, silence=True))
        for repeat in range(args.repeats):
            report['tests'].append(trial(render, capture, args.seconds, 735 + repeat, args.exclusive))
            print(f'PASS restart {repeat + 1}/{args.repeats}', flush=True)
        if args.long_seconds:
            report['tests'].append(trial(render, capture, args.long_seconds, 973, args.exclusive))
        report['tests'].append(trial(render, capture, 2, 0, args.exclusive, silence=True))
        report['passed'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PASS: real stereo PCM, restart and silence; see report for tested duration')

if __name__ == '__main__':
    main()
