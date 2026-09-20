"""Bounded client-stall experiment on an already installed driver. No deployment."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--delays-ms', type=float, nargs='+', default=[0, 4, 8, 12, 20, 30])
    parser.add_argument('--side', choices=['capture', 'render'], default='capture')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--seconds', type=int, default=5)
    parser.add_argument('--capture-api', choices=['portaudio', 'wasapi'], default='portaudio')
    parser.add_argument('--portaudio-library', type=Path)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10 or not 3 <= args.seconds <= 60 or len(args.delays_ms) > 20 or any(not 0 <= d <= 1000 for d in args.delays_ms):
        parser.error('Bounded matrix: repeats 1..10, seconds 3..60, at most 20 delays in 0..1000ms')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'matrix.json').exists():
        parser.error('Existing matrix evidence will not be overwritten')
    results = []
    for repeat in range(args.repeats):
        for delay in args.delays_ms:
            name = f'{args.side}-{delay:g}ms-r{repeat + 1}'
            raw = args.output / (name + '.raw')
            command = [sys.executable, str(Path(__file__).with_name('frame_identity.py')),
                       '--run', '--seconds', str(args.seconds), '--capture', str(raw),
                       f'--{args.side}-stall-ms', str(delay), '--capture-api', args.capture_api]
            if args.portaudio_library:
                command += ['--portaudio-library', str(args.portaudio_library.resolve())]
            row = dict(name=name, command=command, started_ns=time.time_ns())
            with (args.output / (name + '.console.log')).open('w') as console:
                try:
                    process = subprocess.run(command, stdout=console, stderr=subprocess.STDOUT, timeout=args.seconds + 30)
                    row['exit_code'] = process.returncode
                except subprocess.TimeoutExpired:
                    row['timeout'] = True
            report = raw.with_suffix('.json')
            if report.exists():
                row['report'] = json.loads(report.read_text())
            row['finished_ns'] = time.time_ns()
            results.append(row)
            (args.output / 'matrix.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
            capture = row.get('report', {}).get('capture', {})
            print(name, 'exit=', row.get('exit_code'), 'gaps=', capture.get('source_gap_frames'),
                  'deltas=', [x['delta'] for x in capture.get('transitions', [])[:10]], flush=True)
            if row.get('timeout'):
                raise SystemExit('Case timed out and child was stopped; inspect evidence before further audio tests')


if __name__ == '__main__':
    main()
