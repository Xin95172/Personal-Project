"""Opt-in encoded-PCM experiment and offline boundary analysis. Never installs a driver."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import threading
import time

import numpy as np

RATE = 48000


def signal(first, frames):
    if first < 0 or first + frames > 0x10000000:
        raise ValueError('28-bit source identity range exceeded')
    return (np.arange(first, first + frames, dtype='<u4') | np.uint32(0xA0000000)).view('<i2').reshape(-1, 2)


def analyze_capture(path, expected_frames=None):
    if path.stat().st_size % 4 or not path.stat().st_size:
        raise ValueError('Capture must contain complete PCM16 stereo frames')
    words = np.memmap(path, dtype='<u4', mode='r')
    transitions, previous, invalid, encoded, zeros = [], None, 0, 0, 0
    first_id = last_id = first_position = last_position = None
    gaps = reordered = transition_count = 0
    for start in range(0, len(words), RATE):
        block = words[start:start + RATE]
        valid = (block & 0xF0000000) == 0xA0000000
        positions = np.flatnonzero(valid)
        zeros += int(np.count_nonzero(block == 0))
        invalid += int(np.count_nonzero(~valid & (block != 0)))
        encoded += len(positions)
        offsets = (block[valid] & 0x0FFFFFFF).astype(np.int64) - (positions + start)
        if not len(offsets):
            continue
        ids = (block[valid] & 0x0FFFFFFF).astype(np.int64)
        if first_id is None:
            first_id, first_position = int(ids[0]), int(positions[0] + start)
        steps = np.diff(ids) if last_id is None else np.diff(np.r_[last_id, ids])
        gaps += int(np.sum(steps[steps > 1] - 1))
        reordered += int(np.count_nonzero(steps <= 0))
        last_id, last_position = int(ids[-1]), int(positions[-1] + start)
        changes = np.flatnonzero(np.r_[previous is None or offsets[0] != previous, np.diff(offsets) != 0])
        transition_count += len(changes)
        for index in changes:
            value = int(offsets[index])
            if len(transitions) < 10000:
                transitions.append(dict(capture_frame=int(start + positions[index]),
                                        source_frame=int(block[positions[index]] & 0x0FFFFFFF),
                                        offset=value, delta=None if previous is None else value - previous))
            previous = value
        previous = int(offsets[-1])
    result = dict(frames=len(words), encoded_frames=encoded, zero_frames=zeros,
                  invalid_frames=invalid, transitions=transitions, transition_limit=10000,
                  transition_count=transition_count, transitions_truncated=transition_count > 10000,
                  first_source_frame=first_id, last_source_frame=last_id,
                  first_encoded_capture_frame=first_position, last_encoded_capture_frame=last_position,
                  source_gap_frames=gaps, repeated_or_reordered_transitions=reordered)
    if expected_frames is not None:
        result['expected_frames'] = expected_frames
        result['pcm_passed'] = bool(encoded == expected_frames and first_id == 0
                                    and last_id == expected_frames - 1 and not invalid
                                    and not gaps and not reordered and transition_count == 1)
    del words
    return result


def analyze_log(path):
    rows, summaries, pending, complete_groups = [], [], [], []
    raw = path.read_bytes()
    text = raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig', errors='replace')
    for line in text.splitlines():
        if 'VirMixer: FRAME ' in line:
            row = {k: int(v) for k, v in re.findall(r'(\w+)=(-?\d+)', line)}
            rows.append(row)
            pending.append(row)
        elif 'VirMixer: FRAME_END ' in line:
            summary = {k: int(v) for k, v in re.findall(r'(\w+)=(-?\d+)', line)}
            summaries.append(summary)
            end = summary.get('seq', -1)
            complete_groups.append(end >= 0 and [row.get('seq') for row in pending] == list(range(max(1, end - 255), end + 1)))
            pending = []
    return dict(records=rows, summaries=summaries,
                anomalies=[row for row in rows if row.get('reason')],
                complete=bool(rows) and bool(summaries) and all(complete_groups) and not pending,
                warning='Compare matching epoch/frame across stages; output order is observation order, QPC is callback-entry time. No automatic root-cause verdict.')


def run(seconds, path, *, exclusive=True, read_frames=480,
        capture_stall_ms=0, render_stall_ms=0, stall_after_seconds=2, capture_api='portaudio'):
    import sounddevice as sd
    from verify_driver import find_devices, qpc_now, qpc_frequency

    render, capture = find_devices()

    # Same one-second pre-generated writes as the existing long exclusive test.
    blocks = [signal(second * RATE, RATE) for second in range(seconds)]
    digest = hashlib.sha256()
    for block in blocks:
        digest.update(block.tobytes())

    report = dict(
        format='48000 PCM16 stereo',
        seconds=seconds,
        source_sha256=digest.hexdigest(),
        qpc_frequency=qpc_frequency(),
        devices=dict(render=render, capture=capture),
        input_overflows=[],
        output_underflows=[],
        errors=[],
        writes=[], stalls=[], mode='exclusive' if exclusive else 'shared',
        read_frames=read_frames, capture_stall_ms=capture_stall_ms,
        render_stall_ms=render_stall_ms, stall_after_seconds=stall_after_seconds,
        sounddevice_version=sd.__version__, portaudio_version=sd.get_portaudio_version(),
        portaudio_sha256=hashlib.sha256(Path(sd._libname).read_bytes()).hexdigest(),
        capture_api=capture_api
    )

    stop = threading.Event()
    settings = sd.WasapiSettings(exclusive=exclusive)

    capture_stream = None
    render_stream = None
    raw = None
    worker = None
    tap = None

    try:
        # IMPORTANT:
        # Create both WASAPI exclusive streams on the main thread.
        # Opening them from a Python worker thread fails with PortAudio -9996
        # on this test environment.
        capture_stream = sd.RawInputStream(
            device=capture,
            samplerate=RATE,
            channels=2,
            dtype='int16',
            blocksize=480,
            extra_settings=settings
        )

        render_stream = sd.OutputStream(
            device=render,
            samplerate=RATE,
            channels=2,
            dtype='int16',
            blocksize=480,
            extra_settings=settings
        )

        raw = path.open('wb')

        if capture_api == 'wasapi':
            from wasapi_capture import WasapiCapture
            tap = WasapiCapture(capture_stream, sd)

        capture_stream.start()
        render_stream.start()

        def capture_worker():
            try:
                if tap is not None:
                    tap.open()
                captured, stalled = 0, False
                while not stop.is_set():
                    if capture_stall_ms and not stalled and captured >= stall_after_seconds * RATE:
                        begin = qpc_now()
                        time.sleep(capture_stall_ms / 1000)
                        report['stalls'].append(dict(side='capture', frame=captured, begin_qpc=begin, end_qpc=qpc_now()))
                        stalled = True
                    if tap is not None:
                        data = tap.read(qpc_now)
                        overflow = False  # WASAPI flags are preserved separately.
                        if data is None:
                            stop.wait(0.001)
                            continue
                    else:
                        data, overflow = capture_stream.read(read_frames)
                    raw.write(data)
                    captured += len(data) // 4

                    if overflow:
                        report['input_overflows'].append(qpc_now())

            except Exception as exc:
                if not stop.is_set():
                    report['errors'].append(repr(exc))
            finally:
                if tap is not None:
                    try:
                        tap.close()
                    except Exception as exc:
                        report['errors'].append('WASAPI cleanup: ' + repr(exc))

        worker = threading.Thread(target=capture_worker, daemon=True)
        worker.start()

        # 0.5 s leading silence.
        if render_stream.write(np.zeros((RATE // 2, 2), dtype=np.int16)):
            report['output_underflows'].append(qpc_now())

        # Encoded source.
        for second, block in enumerate(blocks):
            if render_stall_ms and second == stall_after_seconds:
                begin = qpc_now()
                time.sleep(render_stall_ms / 1000)
                report['stalls'].append(dict(side='render', frame=second * RATE, begin_qpc=begin, end_qpc=qpc_now()))
            begin = qpc_now()
            underflow = render_stream.write(block)

            report['writes'].append(dict(
                first=second * RATE,
                frames=RATE,
                begin_qpc=begin,
                end_qpc=qpc_now()
            ))

            if underflow:
                report['output_underflows'].append(qpc_now())

        # 0.5 s trailing silence.
        if render_stream.write(np.zeros((RATE // 2, 2), dtype=np.int16)):
            report['output_underflows'].append(qpc_now())

        time.sleep(0.25)

    except Exception as exc:
        report['errors'].append(repr(exc))

    finally:
        stop.set()

        # Stopping capture releases a worker blocked in read().
        if capture_stream is not None and tap is None:
            try:
                capture_stream.stop()
            except Exception as exc:
                report['errors'].append(repr(exc))

        if worker is not None:
            worker.join(10)
            if worker.is_alive():
                report['errors'].append(
                    'Capture worker did not stop; evidence incomplete'
                )
        if capture_stream is not None and tap is not None:
            try:
                capture_stream.stop()
            except Exception as exc:
                report['errors'].append(repr(exc))
            report['wasapi_packets'] = tap.trace.report()
            if worker is None or not worker.is_alive():
                try:
                    tap.close_pending()
                except Exception as exc:
                    report['errors'].append('WASAPI marshal cleanup: ' + repr(exc))

        if render_stream is not None:
            try:
                render_stream.stop()
            except Exception as exc:
                report['errors'].append(repr(exc))

        if raw is not None:
            raw.close()

        if capture_stream is not None:
            try:
                capture_stream.close()
            except Exception as exc:
                report['errors'].append(repr(exc))

        if render_stream is not None:
            try:
                render_stream.close()
            except Exception as exc:
                report['errors'].append(repr(exc))

        if (
            path.exists()
            and (worker is None or not worker.is_alive())
            and path.stat().st_size
        ):
            report['capture'] = analyze_capture(path, seconds * RATE)

        report['passed'] = bool(report.get('capture', {}).get('pcm_passed')
                                and not report['errors'] and not report['input_overflows']
                                and not report['output_underflows'])

        path.with_suffix('.json').write_text(
            json.dumps(report, indent=2),
            encoding='utf-8'
        )

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--read-frames', type=int, default=480)
    parser.add_argument('--capture-stall-ms', type=float, default=0)
    parser.add_argument('--render-stall-ms', type=float, default=0)
    parser.add_argument('--stall-after-seconds', type=int, default=2)
    parser.add_argument('--capture-api', choices=['portaudio', 'wasapi'], default='portaudio')
    parser.add_argument('--portaudio-library', type=Path)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 3600:
        parser.error('seconds must be 1..3600')
    if not 1 <= args.read_frames <= 48000 or not 0 <= args.capture_stall_ms <= 1000 or not 0 <= args.render_stall_ms <= 1000:
        parser.error('read-frames 1..48000; stall-ms 0..1000')
    if (args.capture_stall_ms or args.render_stall_ms) and not 0 <= args.stall_after_seconds < args.seconds:
        parser.error('stall-after-seconds must fall inside the source duration')
    if args.shared and args.capture_api == 'wasapi':
        parser.error('Direct WASAPI diagnostic reader is exclusive PCM16 only')
    if args.portaudio_library:
        from portaudio_override import load
        load(args.portaudio_library)
    if args.run:
        if args.capture.exists() or args.capture.with_suffix('.json').exists():
            parser.error('Use a new capture path; existing evidence is never overwritten')
        args.capture.parent.mkdir(parents=True, exist_ok=True)
        result = run(args.seconds, args.capture, exclusive=not args.shared, read_frames=args.read_frames,
                     capture_stall_ms=args.capture_stall_ms, render_stall_ms=args.render_stall_ms,
                     stall_after_seconds=args.stall_after_seconds, capture_api=args.capture_api)
    else:
        result = dict(capture=analyze_capture(args.capture))
    if args.log:
        result['kernel'] = analyze_log(args.log)
    if args.report:
        args.report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    summary = {key: value for key, value in result.items() if key not in ('writes', 'kernel', 'wasapi_packets')}
    if 'capture' in summary:
        summary['capture'] = dict(summary['capture'])
        summary['capture']['transitions'] = summary['capture']['transitions'][:12]
    print(json.dumps(summary, indent=2))
    if result.get('errors') or result.get('input_overflows') or result.get('output_underflows'):
        raise SystemExit('Run contains transport errors: retain evidence, do not treat as a clean reproduction')
    if args.run and not result.get('passed'):
        raise SystemExit('PCM identity validation failed; retained report distinguishes injected stalls from an unperturbed run')


if __name__ == '__main__':
    main()
