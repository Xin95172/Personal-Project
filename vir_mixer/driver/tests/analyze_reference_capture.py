"""Exact stereo-word audit of legacy verify_driver deterministic captures.

Matches eight complete stereo frames at each resynchronization; never treats
correlation alone as proof. A bounded search failure is explicitly unresolved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(path, seconds, seed):
    rng = np.random.default_rng(seed)
    reference = np.concatenate([rng.integers(-3000, 3001, (48000, 2), dtype=np.int16)
                                for _ in range(seconds)]).view('<u4').reshape(-1)
    captured = np.memmap(path, dtype='<u4', mode='r')
    candidates = np.flatnonzero(captured[:240000] == reference[0])
    starts = [int(i) for i in candidates if np.array_equal(captured[i:i+8], reference[:8])]
    if len(starts) != 1:
        raise ValueError('Expected one exact initial eight-frame anchor in first five seconds')
    r, s = starts[0], 0
    matched, events, unresolved = 0, [], None
    while s < len(reference) and r < len(captured):
        size = min(4096, len(reference)-s, len(captured)-r)
        wrong = np.flatnonzero(reference[s:s+size] != captured[r:r+size])
        if not len(wrong):
            matched += size
            r += size
            s += size
            continue
        prefix = int(wrong[0])
        r += prefix
        s += prefix
        matched += prefix
        found = None
        lo, hi = max(0, s-8192), min(len(reference)-7, s+8192)
        for next_r in range(r, min(len(captured)-7, r+8192)):
            if captured[next_r] == 0:
                continue
            positions = np.flatnonzero(reference[lo:hi] == captured[next_r])
            exact = [lo+int(i) for i in positions if np.array_equal(reference[lo+i:lo+i+8], captured[next_r:next_r+8])]
            if len(exact) == 1:
                found = next_r, exact[0]
                break
        if found is None:
            unresolved = dict(capture_frame=r, expected_source_frame=s)
            break
        nr, ns = found
        events.append(dict(capture_frame=r, expected_source_frame=s, resumed_capture_frame=nr,
                           resumed_source_frame=ns, offset_change=(ns-nr)-(s-r),
                           unmatched_capture_frames=nr-r, source_advance=ns-s,
                           persistent_offset=ns-nr+starts[0]))
        r, s = nr, ns
    return dict(seed=seed, seconds=seconds, capture_frames=len(captured), initial_capture_frame=starts[0],
                expected_source_frames=len(reference), exact_matched_frames=matched,
                source_end_reached=s==len(reference), events=events, unresolved=unresolved,
                raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--seed', type=int, default=973)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.capture, args.seconds, args.seed)
    args.report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
